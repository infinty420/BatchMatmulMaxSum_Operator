// ============================================================
// BatchMatmulMaxSum Host 侧 Tiling 实现
// 职责：
//   1. 读取输入物理 shape、属性 transposeX1/transposeX2 与数据类型；
//   2. 完成赛题要求的全部合法性校验；
//   3. 由“物理存储 shape + 属性”反推逻辑 B/M/N/K（逻辑-存储映射）；
//   4. 计算多核任务切分与 blockDim；
//   5. 通过 MultiCoreMatmulTiling 生成 Cube 高阶 API 所需的标准 TCubeTiling；
//   6. 推导 workspace 大小（用户 partial 区 + 高阶 API 系统区）。
// ============================================================
#include <cstdint>
#include <string>

#include "batch_matmul_max_sum_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {

// ---------- 分块常量（对齐 Cube 分形要求，均为 16 的倍数） ----------
constexpr uint32_t BASE_M = 64;     // 每个任务处理 64 个 query token 行（一次 ReduceSum 即可归约）
constexpr uint32_t BASE_N = 128;    // 每次迭代处理 128 个 document token 列（fp32 下分两半做 ReduceMax）

// 维度上限与规模上限（赛题硬约束）
constexpr uint32_t MAX_BATCH = 64;
constexpr uint32_t MAX_DIM = 8192;
constexpr int64_t MAX_ELEMENTS = (1LL << 26);  // B*M*K、B*N*K 均不得超过 2^26
constexpr uint32_t MIN_K = 32;

// workspace 用户区对齐粒度（字节）
constexpr uint32_t WORKSPACE_ALIGN = 64;

/*
 * 错误上报辅助：直接返回失败状态（CANN 9.0 新式算子不依赖 ge_log 宏，
 * 框架会根据 graphStatus 记录错误）。
 */
static ge::graphStatus TilingFail() {
    return ge::GRAPH_FAILED;
}

/*
 * 向上对齐
 */
static uint32_t AlignUp(uint32_t value, uint32_t align) {
    return (value + align - 1) / align * align;
}

/*
 * Tiling 主函数：由框架在算子编译/下发阶段回调。
 */
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    // ---------------- 1. 读取输入物理存储 shape ----------------
    const gert::StorageShape *x1Shape = context->GetInputShape(0);
    const gert::StorageShape *x2Shape = context->GetInputShape(1);
    if (x1Shape == nullptr || x2Shape == nullptr) {
        return TilingFail();
    }

    const ge::StorageShape &x1Storage = x1Shape->GetStorageShape();
    const ge::StorageShape &x2Storage = x2Shape->GetStorageShape();

    if (x1Storage.GetDimNum() != 3 || x2Storage.GetDimNum() != 3) {
        return TilingFail();
    }

    // ---------------- 2. 读取属性 transposeX1 / transposeX2 ----------------
    bool transposeX1 = false;
    bool transposeX2 = false;
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs != nullptr) {
        // CANN 9.0 按属性索引取值（索引对应 OpDef 中 Attr 注册顺序）
        const bool *tp1 = attrs->GetBool(0);
        const bool *tp2 = attrs->GetBool(1);
        if (tp1 != nullptr) {
            transposeX1 = *tp1;
        }
        if (tp2 != nullptr) {
            transposeX2 = *tp2;
        }
    }

    // ---------------- 3. 逻辑-存储映射：由物理 shape + 属性反推逻辑维度 ----------------
    // x1 物理布局：false -> [B,M,K]，true -> [B,K,M]
    // x2 物理布局：false -> [B,K,N]，true -> [B,N,K]
    int64_t x1D0 = x1Storage.GetDim(0);
    int64_t x1D1 = x1Storage.GetDim(1);
    int64_t x1D2 = x1Storage.GetDim(2);
    int64_t x2D0 = x2Storage.GetDim(0);
    int64_t x2D1 = x2Storage.GetDim(1);
    int64_t x2D2 = x2Storage.GetDim(2);

    int64_t batchB = x1D0;
    int64_t logicalM = transposeX1 ? x1D2 : x1D1;
    int64_t logicalK1 = transposeX1 ? x1D1 : x1D2;
    int64_t logicalN = transposeX2 ? x2D1 : x2D2;
    int64_t logicalK2 = transposeX2 ? x2D2 : x2D1;

    // ---------------- 4. 合法性校验（赛题全部约束） ----------------
    if (x2D0 != batchB) {
        return TilingFail();
    }
    if (logicalK1 != logicalK2) {
        return TilingFail();
    }
    if (batchB < 1 || batchB > static_cast<int64_t>(MAX_BATCH)) {
        return TilingFail();
    }
    if (logicalM < 1 || logicalM > static_cast<int64_t>(MAX_DIM)) {
        return TilingFail();
    }
    if (logicalN < 1 || logicalN > static_cast<int64_t>(MAX_DIM)) {
        return TilingFail();
    }
    if (logicalK1 < static_cast<int64_t>(MIN_K) || logicalK1 > static_cast<int64_t>(MAX_DIM)) {
        return TilingFail();
    }
    if (logicalK1 % 8 != 0) {
        return TilingFail();
    }
    if (batchB * logicalM * logicalK1 > MAX_ELEMENTS) {
        return TilingFail();
    }
    if (batchB * logicalN * logicalK1 > MAX_ELEMENTS) {
        return TilingFail();
    }

    // 数据类型校验：两输入类型必须相同，且仅支持 FP16 / BF16
    ge::DataType x1Dtype = context->GetInputDesc(0)->GetDataType();
    ge::DataType x2Dtype = context->GetInputDesc(1)->GetDataType();
    if (x1Dtype != x2Dtype) {
        return TilingFail();
    }
    uint8_t dataTypeFlag = 0;
    matmul_tiling::DataType cubeDtype = matmul_tiling::DataType::DT_FLOAT16;
    if (x1Dtype == ge::DT_FLOAT16) {
        dataTypeFlag = 0;
        cubeDtype = matmul_tiling::DataType::DT_FLOAT16;
    } else if (x1Dtype == ge::DT_BF16) {
        dataTypeFlag = 1;
        cubeDtype = matmul_tiling::DataType::DT_BF16;
    } else {
        return TilingFail();
    }

    // ---------------- 5. 分块与多核任务切分 ----------------
    uint32_t b = static_cast<uint32_t>(batchB);
    uint32_t m = static_cast<uint32_t>(logicalM);
    uint32_t n = static_cast<uint32_t>(logicalN);
    uint32_t k = static_cast<uint32_t>(logicalK1);

    // 各维分块数（尾块不足一个分块也计一块，Kernel 中按实际长度处理）
    uint32_t numMTiles = (m + BASE_M - 1) / BASE_M;
    uint32_t numNTiles = (n + BASE_N - 1) / BASE_N;

    // 任务以 (b, m 行块) 为最小单位：同一任务内通过 Iterate 顺序遍历全部 N 分块，
    // K 维累加在 Matmul 高阶 API 内部完成；每个 query token 的 Max/Sum 全部片上结束，
    // 完整中间矩阵 A 不写回 GM。
    uint32_t totalTasks = b * numMTiles;

    // 获取硬件 AI Core（Cube 核）数量，blockDim 不超过任务数
    platform_ascendc::PlatformAscendC platformInfo(context->GetPlatformInfo());
    uint32_t coreNum = platformInfo.GetCoreNumAic();
    uint32_t blockNum = coreNum < totalTasks ? coreNum : totalTasks;
    if (blockNum < 1) {
        blockNum = 1;
    }

    // ---------------- 6. 生成 Cube 高阶 API 的标准 TCubeTiling ----------------
    // 单任务形状：[BASE_M, N, K]，SetDim(1) 表示该 tiling 为“单核单任务”视角，
    // 多核切分由本函数的任务轮转实现。
    matmul_tiling::MultiCoreMatmulTiling cubeTiling(platformInfo);
    cubeTiling.SetDim(1);
    // 第 4 个入参 isTranspose：物理布局是否转置——仅影响分块与访存，不插入转置算子。
    cubeTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        cubeDtype, transposeX1);
    cubeTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        cubeDtype, transposeX2);
    // C 在 Kernel 侧实际输出到 VECIN（片上 UB），Host 侧仍按 GM 声明，与官方融合样例一致。
    cubeTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                           matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetOrgShape(BASE_M, n, k);
    cubeTiling.SetShape(BASE_M, n, k);
    cubeTiling.SetFixSplit(BASE_M, BASE_N, -1);
    cubeTiling.EnableBias(false);
    cubeTiling.SetBufferSpace(-1, -1, -1);

    // ---------------- 7. 下发 TilingData ----------------
    TilingData tiling = TilingData();
    if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1) {
        return TilingFail();
    }

    // workspace 用户区：每个任务 1 个 FP32 partial，0 号核汇合后按固定任务顺序归约，
    // 不使用原子加，保证多次运行结果确定、无累加乱序。
    uint32_t userWorkspaceBytes = AlignUp(totalTasks * sizeof(float), WORKSPACE_ALIGN);

    tiling.set_batchNum(b);
    tiling.set_mDim(m);
    tiling.set_nDim(n);
    tiling.set_kDim(k);
    tiling.set_baseM(BASE_M);
    tiling.set_baseN(BASE_N);
    tiling.set_numMTiles(numMTiles);
    tiling.set_numNTiles(numNTiles);
    tiling.set_totalTasks(totalTasks);
    tiling.set_blockNum(blockNum);
    tiling.set_transposeX1(transposeX1 ? 1U : 0U);
    tiling.set_transposeX2(transposeX2 ? 1U : 0U);
    tiling.set_dataType(dataTypeFlag);
    tiling.set_userWorkspaceBytes(userWorkspaceBytes);

    // 动态 shape 统一使用 tiling key 0，数据类型通过 dataType 字段在 Kernel 内分支
    context->SetTilingKey(0);
    context->SetBlockDim(blockNum);
    context->SetTilingData(tiling);

    // workspace 总量 = 用户区 + Matmul 高阶 API 系统区（框架统一申请、分别管理）
    size_t systemWorkspaceSize = static_cast<size_t>(platformInfo.GetLibApiWorkSpaceSize());
    size_t workspaceSize = userWorkspaceBytes + systemWorkspaceSize;
    ge::graphStatus ret = context->SetWorkspaceSizes({{workspaceSize, nullptr}});
    if (ret != ge::GRAPH_SUCCESS) {
        return TilingFail();
    }

    return ge::GRAPH_SUCCESS;
}

/*
 * 算子 Tiling 入口注册：框架编译到 BatchMatmulMaxSum 时回调本函数。
 */
IMPL_OP_OPTILING(BatchMatmulMaxSum) {
    return TilingFunc(context);
}

}  // namespace optiling
