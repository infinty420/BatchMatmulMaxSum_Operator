#ifndef BATCH_MATMUL_MAX_SUM_TILING_H
#define BATCH_MATMUL_MAX_SUM_TILING_H

#include <cstdint>

#include "tiling/tiling_api.h"

/*
 * TilingData：Host 侧 Tiling 模块与 Device 侧 Kernel 之间的参数契约。
 * 该结构体通过 context->GetTilingData<TilingData>() 按内存布局直接传递给 Kernel。
 * 不使用 BEGIN_TILING_DATA_DEF 序列化宏，保持类型在当前 namespace 内可见。
 */
struct TilingData {
    uint32_t batchNum;              // Batch 数 B
    uint32_t mDim;                  // 逻辑 M（query token 数）
    uint32_t nDim;                  // 逻辑 N（document token 数）
    uint32_t kDim;                  // K（embedding 维 / 点积长度）

    uint32_t baseM;                 // M 维分块大小（行块，固定 64）
    uint32_t baseN;                 // N 维分块大小（固定 128）

    uint32_t numMTiles;             // M 维分块数（向上取整）
    uint32_t numNTiles;             // N 维分块数量
    uint32_t totalTasks;            // 总任务数 = B * numMTiles

    uint32_t blockNum;              // 实际启用的 AI Core 数

    uint8_t transposeX1;            // x1 物理布局：0=[B,M,K]，1=[B,K,M]
    uint8_t transposeX2;            // x2 物理布局：0=[B,K,N]，1=[B,N,K]
    uint8_t dataType;               // 0=float16, 1=bfloat16

    uint32_t userWorkspaceBytes;    // workspace 用户区字节数（partial 数组）

    // Cube 高阶 API（Matmul）所需的标准切分参数，由 Host 侧 MultiCoreMatmulTiling 生成。
    // 其形状固定为 [baseM, N, K]（单任务形状），尾行由 Kernel 侧 SetTail 重配。
    matmul_tiling::TCubeTiling cubeTilingData;
};

#endif  // BATCH_MATMUL_MAX_SUM_TILING_H
