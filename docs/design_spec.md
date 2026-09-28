# BatchMatmulMaxSum 算子竞赛设计说明书

> 赛事：AtomGit 开放原子大赛【上合赛区】
> 赛题：BatchMatmulMaxSum Ascend-C 自定义算子
> 硬件平台：昇腾 NPU AI-Core（Cube + Vector），CANN 9.0.0
> 文档版本：v1.0

---

## 1. 项目背景

### 1.1 ColBERT Late-Interaction 晚交互打分

检索增强生成（RAG）系统的核心环节之一是 **Query 与候选文档的相关性打分**。ColBERT 提出的 Late-Interaction（晚交互）模型在 **token 粒度**上计算查询与文档的语义相似度，兼顾检索精度与效率：

1. 将 Query 的每个 token 向量与 Document 的全部 token 向量做点积，得到 token 级相似度矩阵；
2. 对每个 Query token，只保留其与所有 Document token 的**最大相似度**（MaxSim）；
3. 将所有 Query token 的 MaxSim **求和**，得到 Query-Document 最终相关性分数。

该流程天然包含 **批量矩阵乘 + N 维 Max 归约 + M 维 Sum 归约** 三段计算，且中间矩阵体量为 `[B, M, N]`，在长文档、多候选场景下 Global Memory（以下简称 GM）访存开销显著。将其融合为**单 AICore Kernel**，是本赛题的价值所在。

### 1.2 业务落地场景（未来对接，不含于本工程）

本工程是**完全独立的 NPU 算子项目**，仅包含算子 Kernel、Host 侧 Tiling 与精度验证脚本，**不包含任何上层应用业务源码**。

后续该算子可接入**运动健康 Agent**，作为其 RAG 检索链路的重排算子：对运动训练、饮食营养、运动康复等知识库文档进行 token-level 相关性打分，为用户输出高质量的运动训练计划、饮食与康复建议。上层 Agent 应用的开发不在本工程范围内。

---

## 2. 算子定义

### 2.1 算子原型

| 项目 | 内容 |
|---|---|
| 算子名 | `BatchMatmulMaxSum` |
| 输入 x1 | FP16 / BF16，逻辑 shape `[B, M, K]` |
| 输入 x2 | FP16 / BF16，逻辑 shape `[B, K, N]` |
| 输出 y | FP32，shape `[B]` |
| 属性 transposeX1 | bool，仅声明 x1 物理存储排布，默认 false |
| 属性 transposeX2 | bool，仅声明 x2 物理存储排布，默认 false |

### 2.2 物理存储排布（transpose 语义）

transpose 属性**只改变张量的物理存储排布**，不改变逻辑 shape：

| 属性 | 物理 shape |
|---|---|
| transposeX1 = false | x1：`[B, M, K]` |
| transposeX1 = true  | x1：`[B, K, M]` |
| transposeX2 = false | x2：`[B, K, N]` |
| transposeX2 = true  | x2：`[B, N, K]` |

> 无论属性取值，x1 逻辑 shape 恒为 `[B,M,K]`，x2 逻辑 shape 恒为 `[B,K,N]`。
> **禁止执行显式矩阵转置运算**，算子仅通过访存索引差异读取数据。

### 2.3 数学计算流程（顺序为评分红线，不可调换）

**第一步：BatchMatmul 批量矩阵乘**

$$
A[b,m,n] = \sum_{k=0}^{K-1} x1[b,m,k] \cdot x2[b,k,n], \qquad A \in \mathbb{R}^{B\times M\times N}
$$

- batch 维严格一一配对：第 b 组 x1 只与第 b 组 x2 计算；
- 禁止跨 batch 广播，禁止 batch 之间笛卡尔积混合。

**第二步：N 维 Max 归约**

$$
R[b,m] = \max_{n} A[b,m,n], \qquad R \in \mathbb{R}^{B\times M}
$$

**第三步：M 维 Sum 归约**

$$
y[b] = \sum_{m=0}^{M-1} R[b,m], \qquad y \in \mathbb{R}^{B}
$$

PyTorch Golden 参考实现：

```python
def golden_batch_matmul_max_sum(x1_logical, x2_logical):
    sim = torch.bmm(x1_logical.double(), x2_logical.double())  # FP64
    max_sim = torch.amax(sim, dim=-1)                          # N 维 Max
    out = torch.sum(max_sim, dim=-1)                           # M 维 Sum
    return out.float()
```

---

## 3. 赛题约束与设计对策总览

| 编号 | 赛题约束 | 本工程对策 |
|---|---|---|
| 1 | 单 AICore 融合 Kernel，三段计算不拆算子 | 单个 Kernel 内由 Matmul 高阶 API + Vector Reduce 完成；中间 A 不回写 GM |
| 2 | 支持 4 种 transpose 组合，不做显式转置 | MatmulType 使能转置读，SetTensorA/B 运行时传属性，仅变访存索引 |
| 3 | 动态 shape、非对齐尾块、M/N/K ≤ 8192 | Host 动态 Tiling；SetTail 处理尾行；尾块无效列填充负无穷后归约 |
| 4 | 精度达标、结果可复现 | FP32 片上累加；固定顺序归约，不使用原子加；固定种子生成数据 |
| 5 | Cube 负责乘、Vector 负责两级归约 | Matmul（Cube）+ ReduceMax/ReduceSum（Vector），MIX 模式 |
| 6 | Host 侧校验、分片、workspace 推导 | TilingFunc 全量合法性校验 + MultiCoreMatmulTiling + workspace 分区 |
| 7 | 工程可直接编译提交 | CMakeLists.txt + build.sh，产出 `libbatch_matmul_max_sum.so` |

维度边界：`1 ≤ B ≤ 64`，`1 ≤ M,N ≤ 8192`，`32 ≤ K ≤ 8192`（K 为 8 的倍数），且 `B·M·K ≤ 2²⁶`、`B·N·K ≤ 2²⁶`。

---

## 4. Host 侧 Tiling 设计

### 4.1 逻辑-存储映射

Host 侧拿到的是**物理 shape**，结合属性反推逻辑维度：

```
逻辑M  = transposeX1 ? x1物理D2 : x1物理D1
逻辑K1 = transposeX1 ? x1物理D1 : x1物理D2
逻辑N  = transposeX2 ? x2物理D1 : x2物理D2
逻辑K2 = transposeX2 ? x2物理D2 : x2物理D1
B      = x1物理D0（必须等于 x2物理D0）
```

### 4.2 合法性校验

TilingFunc 依次校验：

1. 两输入均为 3D Tensor；
2. batch 维相等（不支持 batch broadcast）；
3. 反推的两个 K 相等；
4. `B/M/N/K` 取值范围（含 K 为 8 的倍数）；
5. `B*M*K ≤ 2²⁶`、`B*N*K ≤ 2²⁶`；
6. 两输入 dtype 相同且仅为 FP16/BF16。

### 4.3 任务切分（多核并行）

- **行块常量**：`BASE_M = 64`，`BASE_N = 128`；
- **任务粒度**：一个任务对应 `(batch, m 行块)`，任务数：

$$
totalTasks = B \cdot \lceil M / 64 \rceil
$$

- **blockDim**：`min(GetCoreNumAic(), totalTasks)`；
- 各核按任务号轮转：`taskId = blockIdx, blockIdx+blockNum, ...`。

任务内通过 Matmul 的 `Iterate` 顺序遍历该 batch 的全部 N 分块；K 维累加由高阶 API 内部完成。该切分方式天然保证 batch 严格配对。

### 4.4 Cube 标准 Tiling（TCubeTiling）

通过 `MultiCoreMatmulTiling` 生成 Matmul 高阶 API 所需的标准 `TCubeTiling`：

```cpp
cubeTiling.SetDim(1);                                   // 单任务视角
cubeTiling.SetAType(GM, ND, dtype, transposeX1);        // 第4参=物理是否转置
cubeTiling.SetBType(GM, ND, dtype, transposeX2);
cubeTiling.SetCType(GM, ND, DT_FLOAT);                  // Kernel 侧实际输出到 VECIN
cubeTiling.SetBiasType(GM, ND, DT_FLOAT);
cubeTiling.SetOrgShape(64, N, K);
cubeTiling.SetShape(64, N, K);
cubeTiling.SetFixSplit(64, 128, -1);                    // baseM/baseN/baseK自动
cubeTiling.EnableBias(false);
cubeTiling.SetBufferSpace(-1, -1, -1);
```

生成的 `TCubeTiling` 以 `TILING_DATA_FIELD_DEF_STRUCT` 嵌入 TilingData，Kernel 侧通过 `REGIST_MATMUL_OBJ` 绑定。

### 4.5 Workspace 推导

workspace 分为两部分，由框架统一申请：

| 分区 | 大小 | 用途 |
|---|---|---|
| 用户区 | `align64(totalTasks * 4)` | 每任务一个 FP32 partial |
| 系统区 | `GetLibApiWorkSpaceSize()` | Matmul 高阶 API 内部使用，框架管理 |

末阶段由 0 号核按 **batch 顺序 + 行块顺序**归约 partial，避免原子加的浮点乱序，保证相同输入多次运行结果完全一致。

---

## 5. Kernel 融合方案

### 5.1 整体执行流

```text
每个 AI Core：
  ┌─ 任务轮转循环（taskId = blockIdx, +blockNum, ...）
  │    1. 由 taskId 反解 (batch, m行块) 及实际行数 actualRows
  │    2. 计算 x1/x2 子矩阵 GM 偏移（transpose 索引变换）
  │    3. SetTail(actualRows) + SetTensorA/B + DisableBias
  │    4. runningMax[64] 初值置负无穷
  │    ┌─ while (matmul.Iterate())          // 每轮一个 64×128 C 块
  │    │    GetTensorC(chunk)               // L0C → UB（ND，FP32）
  │    │    块内按行 ReduceMax(128列)        // N 维 Max
  │    │    Max(runningMax, blockMax)       // 跨 N 块在线更新
  │    └─
  │    5. ReduceSum(runningMax, actualRows) // M 维 Sum
  │    6. partial[taskId] = 标量（DataCopyPad）
  └─
  SyncAll()                                 // 所有核硬件栅栏汇合
  0 号核：按 batch 分组 ReduceSum partial → 写 y[b]
```

### 5.2 单 Kernel 融合与 GM 访存优化

- 相似度矩阵 A 的每个 64×128 分块在 **L0C 产生后直接搬到 UB**（C 类型声明为 `VECIN`），随即被 Vector 消费；
- **完整 A 矩阵不落 GM**：GM 中仅出现最终输出 y 与 tiny 的 partial 工作区；
- 对比"Matmul 算子 + Max 算子 + Sum 算子"三算子方案，省去 `[B,M,N]` 矩阵的一次 GM 写回与两次 GM 读入，显著降低 DDR 带宽压力。

### 5.3 transpose 访存索引处理（不做显式转置）

逻辑元素到物理地址的映射：

```text
x1 元素 (b, m, k)：
  transposeX1=false → b*(M*K) + m*K + k     // 物理[B,M,K]
  transposeX1=true  → b*(K*M) + k*M + m     // 物理[B,K,M]

x2 元素 (b, k, n)：
  transposeX2=false → b*(K*N) + k*N + n     // 物理[B,K,N]
  transposeX2=true  → b*(N*K) + n*K + k     // 物理[B,N,K]
```

实现上借助 **Cube 硬件原生的转置读能力**：

1. MatmulType 模板第 4 参置 `true`，声明"允许转置读"；
2. `SetTensorA/B` 时把属性作为 `isTranspose` 入参传入；
3. 硬件仅改变取数索引与分形排布，不插入任何转置指令/算子，完全符合"只改访存索引"的红线。

行块起点偏移：

```text
transposeX1=false：offsetA = b*(M*K) + rowStart*K   （行优先连续行）
transposeX1=true ：offsetA = b*(K*M) + rowStart     （物理平面内连续列）
```

### 5.4 非对齐尾块处理

**M 维尾行**：最后一个行块 `actualRows = M - rowStart < 64`。调用 `SetTail(actualRows, -1, -1)`，不改变 Tiling、仅重配本次单任务 M；满块任务显式传 64，保证对象跨任务复用恢复。

**N 维尾列**：Matmul Tiling 以真实 N 构建，最后一块的有效列数 `validN < 128`。FP32 向量指令一次处理 64 个元素，128 列拆前后两半：

- `validN ≥ 64`：前半满归约；后半归约 `validN-64` 列，不足 64 时将无效列填充为负无穷（`-3.4e38`）再按满 64 归约；
- `validN < 64`：填充 `[validN, 64)` 为负无穷后归约；
- 无效列填充逐行 `Duplicate`，仅发生在唯一一个尾块，开销可忽略。

**初值正确性**：行最大值 `runningMax` 初值必须为负无穷，**不能初值为 0**——当相似度全为负时，初值 0 会得到错误结果。

### 5.5 多核同步与末阶段归约

- 所有核任务结束后调用硬件 `SyncAll()` 栅栏，保证 partial 全部可见；
- 0 号核按固定顺序处理：每个 batch 一次 `DataCopyPad` 读入其全部行块 partial，`ReduceSum` 后经 `DataCopyPad` 写 `y[b]`；
- 不使用 `SetAtomicAdd`：浮点原子加的完成顺序随调度变化，可能引入运行间抖动；固定顺序归约严格可复现。

---

## 6. 片上资源预算（单核 UB）

| Buffer | 大小 | 说明 |
|---|---:|---|
| C 块队列 cQueue | 32 KB | 64×128 FP32，深度 1 |
| 行最大值区 maxBuf | 768 B | maxLo/maxHi/runningMax 各 64 FP32 |
| Reduce 工作区 workBuf | 256 B | ReduceMax/ReduceSum 内部缓冲 |
| 末阶段区 reduceBuf | 544 B | partial 组 128 FP32 + 标量 |
| **合计** | **≈ 34 KB** | 远小于 UB 容量，预留充足流水空间 |

---

## 7. 性能优化思路

1. **算子融合**：三段计算单 Kernel 完成，消除中间矩阵 GM 往返，这是最大的带宽收益；
2. **Cube + Vector 硬件分工**：矩阵乘走 Cube 高吞吐阵列，归约走 Vector，避免用 Vector 模拟矩阵乘；
3. **高阶 API 内置流水线**：Matmul 内部自动完成 GM→L1→L0A/L0B 的分块搬运与 K 维累加，自带 Double Buffer / Ping-Pong 流水，掩盖访存延迟；
4. **行块尺寸选择**：`BASE_M=64` 使 M 维 Sum 一次 ReduceSum 即可完成；`BASE_N=128` 兼顾 Cube 分形效率与 N 维归约次数；
5. **多核轮转**：任务数不少于核数时所有核均有任务，尾块仅影响最后一个行块任务；
6. **标量走 DataCopyPad**：4B partial/y 的读写不触发非对齐异常；
7. **性能分析**：可通过 msprof 采集 `op_summary`、`task_time`，重点观察 Cube/Vector 利用率与 MTE 搬送占比，后续可通过调整 FixSplit 或启用更深度流水进一步调优。

---

## 8. 精度保障

1. **FP32 累加**：Cube 输出 FP32，两级归约全程 FP32，降低大 K 求和误差；
2. **Golden 口径一致**：先将随机输入量化为 FP16/BF16 实际存储值，再于 FP64 下执行 bmm→amax→sum，避免"随机真值 vs 量化输入"的口径偏差；
3. **确定性归约**：固定任务顺序累加，无原子操作，多次运行逐位一致、无随机抖动；
4. **比对阈值**：FP16/BF16 采用 `rtol=atol=1e-3`，逐元素以相对误差或绝对误差通过判定。

---

## 9. 测试用例覆盖

`testcase/test_cases.json` 共 20 组用例，覆盖矩阵：

- **4 种 transpose 组合**：(F,F)、(T,F)、(F,T)、(T,T) 均有多个用例；
- **最小维度**：B=M=N=1，K=32；
- **非对齐尾块**：M=17/65/77/100，N=3/33/99/100/130 等，覆盖尾块落在前半 64 列与后半 64 列两类情形；
- **维度上限**：M=8192、N=8192、K=8192 各有专项用例；
- **典型业务规模**：ColBERT 常见 `B=8,M=128,N=512,K=128`；
- **两种 dtype**：FP16 与 BF16（含 BF16 双尾块用例）；
- **大 Batch**：B=64。

所有用例均通过元素总量上限校验，可直接由 `gen_golden.py` 生成数据。

---

## 10. 编译与验证流程

```bash
# 1. 编译（昇腾 Linux 服务器，已安装 CANN 9.0.0）
bash build.sh                 # 产出 build/libbatch_matmul_max_sum.so

# 2. 生成 Golden 数据（任意安装 torch/numpy 的环境）
python3 scripts/gen_golden.py

# 3. 算子执行后，比对输出与 Golden（输出文件命名为 y_output.npy）
python3 scripts/verify_result.py \
    --golden scripts/data/<case_id>/y_golden.npy \
    --output <path>/y_output.npy
```

验证脚本逐元素计算相对/绝对误差并输出校验报告，全部通过时退出码为 0。

---

## 11. 交付目录

```text
BatchMatmulMaxSum_Operator/
├── CMakeLists.txt
├── build.sh
├── msopgen_op.json
├── op_kernel/
│   └── batch_matmul_max_sum_kernel.cpp
├── op_host/
│   ├── batch_matmul_max_sum_tiling.h
│   ├── batch_matmul_max_sum_tiling.cpp
│   └── batch_matmul_max_sum_op.cpp
├── scripts/
│   ├── gen_golden.py
│   └── verify_result.py
├── testcase/
│   └── test_cases.json
├── docs/
│   ├── README.md
│   └── design_spec.md
└── .gitignore
```

本工程不包含任何鸿蒙 ArkTS、UI 或运动健康 App 业务源码；与运动健康 Agent 的关系仅为"未来可对接的 RAG 重排算子"，在文档层面描述。

---

## 12. 结论

本方案以"单 AICore 融合 Kernel"为核心：Cube 高阶 Matmul 完成批量矩阵乘，Vector 即时完成 N 维 Max 与 M 维 Sum，中间矩阵全程片上流转；transpose 属性完全通过硬件转置读与偏移索引实现，非对齐尾块动态处理；Host 侧完成全量校验、动态分片与 workspace 推导。工程可一键编译产出算子 so，并具备 Golden 生成与精度比对的完整闭环，满足 AtomGit 赛题初赛提交的全部要求。
