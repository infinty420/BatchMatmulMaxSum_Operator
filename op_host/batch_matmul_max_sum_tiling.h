#ifndef BATCH_MATMUL_MAX_SUM_TILING_H
#define BATCH_MATMUL_MAX_SUM_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

/*
 * TilingData：Host 侧 Tiling 模块与 Device 侧 Kernel 之间的参数契约。
 * 该结构体会被框架序列化后通过 Kernel 入参 tiling 传递给 Device。
 */
BEGIN_TILING_DATA_DEF
    TILING_DATA_FIELD_DEF(uint32_t, batchNum);     // Batch 数 B
    TILING_DATA_FIELD_DEF(uint32_t, mDim);          // 逻辑 M（query token 数）
    TILING_DATA_FIELD_DEF(uint32_t, nDim);          // 逻辑 N（document token 数）
    TILING_DATA_FIELD_DEF(uint32_t, kDim);          // K（embedding 维 / 点积长度）

    TILING_DATA_FIELD_DEF(uint32_t, baseM);         // M 维分块大小（行块，固定 64）
    TILING_DATA_FIELD_DEF(uint32_t, baseN);         // N 维分块大小（固定 128）

    TILING_DATA_FIELD_DEF(uint32_t, numMTiles);     // M 维分块数（向上取整）
    TILING_DATA_FIELD_DEF(uint32_t, numNTiles);     // N 维分块数量
    TILING_DATA_FIELD_DEF(uint32_t, totalTasks);    // 总任务数 = B * numMTiles

    TILING_DATA_FIELD_DEF(uint32_t, blockNum);      // 实际启用的 AI Core 数

    TILING_DATA_FIELD_DEF(uint8_t, transposeX1);    // x1 物理布局：0=[B,M,K]，1=[B,K,M]
    TILING_DATA_FIELD_DEF(uint8_t, transposeX2);    // x2 物理布局：0=[B,K,N]，1=[B,N,K]
    TILING_DATA_FIELD_DEF(uint8_t, dataType);       // 0=float16, 1=bfloat16

    TILING_DATA_FIELD_DEF(uint32_t, userWorkspaceBytes);  // workspace 用户区字节数（partial 数组）

    // Cube 高阶 API（Matmul）所需的标准切分参数，由 Host 侧 MultiCoreMatmulTiling 生成。
    // 其形状固定为 [baseM, N, K]（单任务形状），尾行由 Kernel 侧 SetTail 重配。
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingData);
END_TILING_DATA_DEF;

// 将 TilingData 与算子类型绑定，供框架反射注册
REGISTER_TILING_DATA_CLASS(BatchMatmulMaxSum, TilingData)

}  // namespace optiling

#endif  // BATCH_MATMUL_MAX_SUM_TILING_H
