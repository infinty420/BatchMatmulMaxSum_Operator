# BatchMatmulMaxSum Ascend-C 自定义算子

> 2026 年 CANN 挑战赛 · **上合赛区初赛**赛题（权重 100%）独立算子工程
> 硬件平台：昇腾 NPU AI Core ｜ 语言：Ascend-C（Device）/ C++（Host）/ Python（精度验证）

## 一、项目简介

本工程基于昇腾 CANN 平台，使用 **Ascend-C** 原生实现 **BatchMatmulMaxSum** 单融合算子，在**同一个 AI Core Kernel** 内完成以下三级计算（顺序固定，不可交换）：

```
BatchMatMul（Cube 矩阵乘） → Max 沿 N 维归约（Vector） → Sum 沿 M 维归约（Vector）
```

- **输入 x1**：FP16 / BF16，逻辑 shape `[B, M, K]`
- **输入 x2**：FP16 / BF16，逻辑 shape `[B, K, N]`
- **输出 y**：FP32，shape `[B]`
- **属性 transposeX1 / transposeX2**：仅描述输入的物理存储布局，**Kernel 只改变访存索引，不执行显式矩阵转置**

赛题红线已全部落实：单 AICore 融合 Kernel、完整中间相似度矩阵 A **不回写 Global Memory**、四种 transpose 组合全支持、动态 shape 与非对齐尾块处理、FP32 精度累加。

## 二、业务背景与落地场景

该算子对应 **ColBERT Late-Interaction（晚交互）** 打分核心逻辑，用于 RAG（检索增强生成）系统中 Query 与知识库 Document 的 token 级相关性打分。

**未来对接方向（不在本工程内实现）**：本算子为完全独立开发的 NPU 算子项目，后续可作为推理加速内核接入「运动健康 Agent」，对运动训练、饮食营养、康复指导知识库文档进行 RAG 重排打分，为用户输出运动训练与饮食康复建议。本工程**不包含任何 Agent 应用、UI 或鸿蒙 ArkTS 代码**，仅交付 NPU 算子内核、Host 侧 tiling 与精度验证能力。

## 三、环境依赖

| 依赖 | 版本 / 说明 |
| :--- | :--- |
| CANN Toolkit | **9.0.0**（赛题指定），已 source `set_env.sh` |
| 编译器 | g++（支持 C++17）、ccec（随 Toolkit 发布） |
| CMake | ≥ 3.16 |
| Python | ≥ 3.8，依赖 `torch`、`numpy` |
| 目标硬件 | 昇腾 Atlas A2 训练系列（默认架构 `dav-c220`，可通过 `ASCEND_ARCH` 覆盖） |

## 四、编译步骤

```bash
# 1) 加载 CANN 环境（若尚未配置）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2) 一键编译
./build.sh

# 3) 清理
./build.sh clean
```

编译成功后产物为：`build/libbatch_matmul_max_sum.so`

## 五、Golden 数据生成与精度验证

```bash
# 生成 testcase/test_cases.json 中全部用例的输入（物理存储布局）与 golden 真值
python3 scripts/gen_golden.py --cases_file testcase/test_cases.json --output_dir scripts/data

# 只生成指定编号用例
python3 scripts/gen_golden.py --case case_basic_ff

# 将算子输出（FP32 npy，shape [B]）与 golden 比对，输出误差报告
python3 scripts/verify_result.py \
    --golden scripts/data/case_basic_ff/y_golden.npy \
    --output scripts/data/case_basic_ff/y_output.npy
```

Golden 真值按赛题要求在 **FP64 精度下完成 bmm → amax → sum，最后转换为 FP32**；判定阈值为相对误差、绝对误差均 `< 1e-3`。

## 六、工程目录

```
BatchMatmulMaxSum_Operator/
├── CMakeLists.txt                     # 主编译配置（g++ + ccec → so）
├── build.sh                           # 一键编译脚本
├── msopgen_op.json                    # msopgen 算子原型定义
├── op_kernel/
│   └── batch_matmul_max_sum_kernel.cpp   # Device 侧 Ascend-C 单融合 Kernel
├── op_host/
│   ├── batch_matmul_max_sum_tiling.h     # TilingData 结构定义
│   ├── batch_matmul_max_sum_tiling.cpp   # Host tiling：校验/分片/workspace
│   └── batch_matmul_max_sum_op.cpp       # 算子注册与框架适配
├── scripts/
│   ├── gen_golden.py                     # Golden 输入/真值生成
│   └── verify_result.py                  # 结果误差比对
├── testcase/
│   └── test_cases.json                   # 15+ 组用例，4 种 transpose 全覆盖
└── docs/
    ├── README.md                         # 本文件
    └── design_spec.md                    # 竞赛设计说明书（初赛提交材料）
```

## 七、赛事提交

1. 将本工程推送到 AtomGit / GitCode 参赛代码仓库；
2. 登录 CANNJudge 平台（`cannjudge.cn`）赛事页面，提交核函数工程进行评测打榜；
3. 赛题共 15 个测试点，**所有用例精度通过方会计分**，单点得分公式：`100 / (1 + log₁.₅(t/T))`。
