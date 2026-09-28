// ============================================================
// BatchMatmulMaxSum Device 侧单 AICore 融合 Kernel
//
// 赛题红线（本文件全部严格遵守）：
//   1. BatchMatmul + N维Max + M维Sum 在同一个 Kernel 内完成，不拆算子；
//   2. 完整中间相似度矩阵 A[B,M,N] 不写回 Global Memory，仅在片上逐块消费；
//   3. transposeX1/transposeX2 只改变访存索引：通过 Cube 硬件的转置读能力实现，
//      不执行任何显式矩阵转置运算；
//   4. 计算顺序固定：先 Matmul（K 累加），再按行取 N 维 Max，最后做 M 维 Sum。
//
// 硬件分工：
//   Cube 单元：Matmul 高阶 API 完成批量矩阵乘（FP16/BF16 输入，FP32 累加）；
//   Vector 单元：ReduceMax（N 维）、ReduceSum（M 维）两级归约。
// ============================================================
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "batch_matmul_max_sum_tiling.h"

using namespace matmul;
using namespace optiling;

// 与 Host 侧分块常量保持一致
constexpr uint32_t BASE_M = 64;     // 行块：每任务最多 64 个 query token
constexpr uint32_t BASE_N = 128;    // 列块：每块最多 128 个 document token

// FP32 单条向量指令一次处理 256B = 64 个元素，128 列需拆两半分别归约
constexpr uint32_t FP32_VEC_CAPACITY = 64;

// 行跨距（BASE_N 个 FP32）换算为 DataCopy 的 32B datablock 数：128*4/32 = 16
constexpr int32_t ROW_STRIDE_BLOCKS = BASE_N * sizeof(float) / 32;

// 片上“负无穷”初值：累加结果量级上限约 K*65504^2 < 4e13，该值远小于任何合法累加值
constexpr float NEG_INF_FP32 = -3.4028235e+38f;

template <typename xType>
class BatchMatmulMaxSumKernel {
public:
    // Matmul 高阶 API 对象。
    // 模板参数依次描述 A / B / C / Bias：
    //   A、B 位于 GM、ND 排布、第 4 个模板参数 true 表示“允许转置读”，
    //       运行时再通过 SetTensorA/B 的 isTranspose 入参决定本次是否转置；
    //   C 位于 VECIN：结果不回写 GM，直接送到 UB 供 Vector 归约（融合关键）；
    //   Bias 仅为满足模板要求，实际通过 EnableBias(false) / DisableBias 关闭。
    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, xType, true>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, xType, true>,
           MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, float>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>>
        matmulObj;

    __aicore__ inline BatchMatmulMaxSumKernel() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace,
                                const TilingData &tiling, AscendC::TPipe *pipe)
    {
        batchNum_ = tiling.batchNum;
        mDim_ = tiling.mDim;
        nDim_ = tiling.nDim;
        kDim_ = tiling.kDim;
        numMTiles_ = tiling.numMTiles;
        totalTasks_ = tiling.totalTasks;
        transposeX1_ = (tiling.transposeX1 != 0);
        transposeX2_ = (tiling.transposeX2 != 0);

        // 绑定整张物理输入张量（元素总数与转置无关，仅两维交换）
        x1Global_.SetGlobalBuffer(reinterpret_cast<__gm__ xType *>(x1),
                                  batchNum_ * mDim_ * kDim_);
        x2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ xType *>(x2),
                                  batchNum_ * nDim_ * kDim_);
        yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y), batchNum_);

        // workspace 用户区：每任务一个 FP32 partial，按任务号固定寻址
        partialGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace), totalTasks_);

        // C 块队列：1 级缓冲，单块 BASE_M*BASE_N 个 FP32（32KB）
        pipe->InitBuffer(cQueue_, 1, BASE_M * BASE_N * sizeof(float));
        // 行最大值工作区：maxLo(64) + maxHi(64) + runningMax(64)
        pipe->InitBuffer(maxBuf_, 3 * BASE_M * sizeof(float));
        // ReduceMax / ReduceSum 的内部缓冲（文档要求 ReduceMax 32 元素、ReduceSum 16 元素）
        pipe->InitBuffer(workBuf_, 64 * sizeof(float));
        // 末阶段 0 号核归约缓冲：partial 组(128) + 标量输出(8)
        pipe->InitBuffer(reduceBuf_, 136 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();

        // 多核轮转：任务号 = blockIdx, blockIdx+blockNum, ...
        // 任务与 (batch, m行块) 一一对应，天然保证 batch 严格配对、不跨 batch 混合。
        for (uint32_t taskId = blockIdx; taskId < totalTasks_; taskId += blockNum) {
            ProcessTask(taskId);
        }

        // 所有核任务全部完成（partial 已落 workspace）后硬件栅栏汇合
        SyncAll();

        // 0 号核按“batch 顺序 + 行块顺序”固定归约，避免原子加乱序，保证结果可复现
        if (blockIdx == 0) {
            FinalReduce();
        }
    }

private:
    /*
     * 单个任务：计算一个 (batch, m行块) 的 partial = sum_m max_n A[b,m,n]
     */
    __aicore__ inline void ProcessTask(uint32_t taskId)
    {
        // 由任务号反解 batch 与行块号
        uint32_t batchIdx = taskId / numMTiles_;
        uint32_t mTileIdx = taskId % numMTiles_;
        uint32_t rowStart = mTileIdx * BASE_M;

        // 实际行数：满块为 BASE_M，最后一个行块为 M 的非对齐尾块
        uint32_t actualRows = mDim_ - rowStart;
        if (actualRows > BASE_M) {
            actualRows = BASE_M;
        }

        // ---------- transpose 访存索引变换（重点） ----------
        // 逻辑元素 x1[b,m,k]：
        //   transposeX1=false，物理 [B,M,K]：地址 = b*(M*K) + m*K + k
        //   transposeX1=true ，物理 [B,K,M]：地址 = b*(K*M) + k*M + m
        // 本任务行范围 [rowStart, rowStart+actualRows)：
        uint32_t planeElemsMK = mDim_ * kDim_;
        uint32_t offsetA;
        if (!transposeX1_) {
            // 行优先：行块起点 = 平面号*平面 + 起始行*K
            offsetA = batchIdx * planeElemsMK + rowStart * kDim_;
        } else {
            // 物理 [K,M]：行块对应平面内连续的列，子视图从列号 rowStart 开始。
            // 后续由 Cube 按 isTransposeA 用“转置读”取出，不做显式转置。
            offsetA = batchIdx * planeElemsMK + rowStart;
        }

        // 逻辑元素 x2[b,k,n]：本任务遍历该 batch 全部 N，整平面参与：
        //   transposeX2=false，物理 [B,K,N]：平面起点 b*(K*N)
        //   transposeX2=true ，物理 [B,N,K]：平面起点 b*(N*K)
        uint32_t planeElemsNK = nDim_ * kDim_;
        uint32_t offsetB = batchIdx * planeElemsNK;

        // 取得片上张量视图
        AscendC::LocalTensor<float> maxAll = maxBuf_.Get<float>();
        AscendC::LocalTensor<float> maxLo = maxAll[0];
        AscendC::LocalTensor<float> maxHi = maxAll[BASE_M];
        AscendC::LocalTensor<float> runningMax = maxAll[BASE_M * 2];
        AscendC::LocalTensor<float> workLocal = workBuf_.Get<float>();
        AscendC::LocalTensor<float> reduceAll = reduceBuf_.Get<float>();
        AscendC::LocalTensor<float> sumScalar = reduceAll[128];

        // 非对齐尾行：不改变 tiling，仅把本次单任务 M 重配为 actualRows；
        // 满块显式传 BASE_M，确保同一对象跨任务复用时能从尾块配置恢复。
        matmulObj.SetTail(static_cast<int>(actualRows), -1, -1);

        // 传入 GM 子视图与转置标志：Cube 按物理布局自行调整取数索引
        matmulObj.SetTensorA(x1Global_[offsetA], transposeX1_);
        matmulObj.SetTensorB(x2Global_[offsetB], transposeX2_);
        matmulObj.DisableBias();

        // 每行 runningMax 初值为负无穷（不能初值为 0：相似度可能全为负）
        Duplicate<float>(runningMax, NEG_INF_FP32, BASE_M);

        // Iterate 每轮产出一个 BASE_M*BASE_N 的 C 块（K 维已在 API 内部完整累加），
        // 块顺序沿 N 维推进。
        uint32_t nStart = 0;
        while (matmulObj.template Iterate<true>()) {
            // L0C -> UB（ND），同步取数
            AscendC::LocalTensor<float> chunk = cQueue_.AllocTensor<float>();
            matmulObj.template GetTensorC<true>(chunk, false, true);
            cQueue_.EnQue(chunk);

            chunk = cQueue_.DeQue<float>();
            // 本块有效列数：满块 BASE_N，N 维非对齐尾块取剩余列
            uint32_t remain = nDim_ - nStart;
            uint32_t validN = remain > BASE_N ? BASE_N : remain;

            // 片上即时归约：块内按行取 N 维最大，并入 runningMax
            UpdateRowMax(chunk, actualRows, validN, maxLo, maxHi, runningMax, workLocal);

            cQueue_.FreeTensor(chunk);
            nStart += validN;
        }
        matmulObj.End();

        // M 维 Sum：对 actualRows 个行最大值求和，得到本任务 partial
        ReduceSum<float>(sumScalar, runningMax, workLocal, static_cast<int32_t>(actualRows));
        AscendC::PipeBarrier<PIPE_V>();

        // 单个 FP32（4B）不满足普通 DataCopy 的 32B 对齐，使用 DataCopyPad 写出
        AscendC::DataCopyExtParams padParams = {1, sizeof(float), 0, 0};
        DataCopyPad(partialGlobal_[taskId], sumScalar, padParams);
    }

    /*
     * 块内 N 维 Max：chunk 为 ND [rows 行 × BASE_N 列]，
     * 归约结果逐元素并入 runningMax（长度 rows）。
     */
    __aicore__ inline void UpdateRowMax(AscendC::LocalTensor<float> &chunk, uint32_t rows,
                                        uint32_t validN, AscendC::LocalTensor<float> &maxLo,
                                        AscendC::LocalTensor<float> &maxHi,
                                        AscendC::LocalTensor<float> &runningMax,
                                        AscendC::LocalTensor<float> &workLocal)
    {
        // ---------- 前 64 列 ----------
        if (validN < FP32_VEC_CAPACITY) {
            // 非对齐尾块：把各行无效列填成负无穷，再按满 64 归约，结果不受污染
            FillInvalidColumns(chunk, rows, validN, FP32_VEC_CAPACITY - validN);
        }
        // 高维 ReduceMax：每组连续 64 个元素求一个最大值，共 rows 组（每行一组），
        // 组间起点间隔 ROW_STRIDE_BLOCKS 个 32B block（即一行跨距）。
        ReduceMax<float>(maxLo, chunk, workLocal, static_cast<int32_t>(FP32_VEC_CAPACITY),
                         static_cast<uint8_t>(rows), ROW_STRIDE_BLOCKS);
        // runningMax = max(runningMax, maxLo)，按 rows 个连续元素计算
        Max<float>(runningMax, runningMax, maxLo, rows);

        // ---------- 后 64 列（validN > 64 时存在） ----------
        if (validN > FP32_VEC_CAPACITY) {
            uint32_t hiValid = validN - FP32_VEC_CAPACITY;
            if (hiValid < FP32_VEC_CAPACITY) {
                // 尾块后半不足 64：列起点为 64 + hiValid
                FillInvalidColumns(chunk, rows, FP32_VEC_CAPACITY + hiValid,
                                   FP32_VEC_CAPACITY - hiValid);
            }
            // chunk[64] 起每行后半段，行跨距仍为 16 blocks
            ReduceMax<float>(maxHi, chunk[FP32_VEC_CAPACITY], workLocal,
                             static_cast<int32_t>(FP32_VEC_CAPACITY),
                             static_cast<uint8_t>(rows), ROW_STRIDE_BLOCKS);
            Max<float>(runningMax, runningMax, maxHi, rows);
        }
    }

    /*
     * 将各行 [colStart, colStart+fillCount) 列填为负无穷。
     * 仅用于 N 维尾块，fillCount < 64，逐行 Duplicate。
     */
    __aicore__ inline void FillInvalidColumns(AscendC::LocalTensor<float> &chunk, uint32_t rows,
                                              uint32_t colStart, uint32_t fillCount)
    {
        for (uint32_t r = 0; r < rows; ++r) {
            Duplicate<float>(chunk[r * BASE_N + colStart], NEG_INF_FP32, fillCount);
        }
    }

    /*
     * 末阶段归约：按 batch 把该 batch 的行块 partial 顺序求和，写最终 y[b]。
     */
    __aicore__ inline void FinalReduce()
    {
        AscendC::LocalTensor<float> reduceAll = reduceBuf_.Get<float>();
        AscendC::LocalTensor<float> partialGroup = reduceAll[0];
        AscendC::LocalTensor<float> sumScalar = reduceAll[128];
        AscendC::LocalTensor<float> workLocal = workBuf_.Get<float>();

        for (uint32_t b = 0; b < batchNum_; ++b) {
            uint32_t taskStart = b * numMTiles_;

            // 一次读入该 batch 的全部行块 partial（numMTiles ≤ 128 个 FP32，
            // 长度可能不足 32B，使用 DataCopyPad；blockLen 单位为字节）
            AscendC::DataCopyExtParams loadParams = {
                1, numMTiles_ * sizeof(float), 0, 0};
            DataCopyPad(partialGroup, partialGlobal_[taskStart], loadParams);

            // M 维 Sum 后即得 y[b]
            ReduceSum<float>(sumScalar, partialGroup, workLocal,
                             static_cast<int32_t>(numMTiles_));
            AscendC::PipeBarrier<PIPE_V>();

            // 标量写回最终输出
            AscendC::DataCopyExtParams storeParams = {1, sizeof(float), 0, 0};
            DataCopyPad(yGlobal_[b], sumScalar, storeParams);
        }
    }

    AscendC::GlobalTensor<xType> x1Global_;
    AscendC::GlobalTensor<xType> x2Global_;
    AscendC::GlobalTensor<float> yGlobal_;
    AscendC::GlobalTensor<float> partialGlobal_;

    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> cQueue_;
    AscendC::TBuf<> maxBuf_;
    AscendC::TBuf<> workBuf_;
    AscendC::TBuf<> reduceBuf_;

    uint32_t batchNum_;
    uint32_t mDim_;
    uint32_t nDim_;
    uint32_t kDim_;
    uint32_t numMTiles_;
    uint32_t totalTasks_;
    bool transposeX1_;
    bool transposeX2_;
};

/*
 * Kernel 入口：参数顺序遵循传统算子工程约定
 *   输入 x1、x2 → 输出 y → workspace → tiling
 */
extern "C" __global__ __aicore__ void batch_matmul_max_sum(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    AscendC::TPipe pipe;
    if (tilingData.dataType == 0) {
        // FP16 输入
        BatchMatmulMaxSumKernel<half> kernel;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), kernel.matmulObj,
                          &tilingData.cubeTilingData);
        kernel.Init(x1, x2, y, workspace, tilingData, &pipe);
        kernel.Process();
    } else {
        // BF16 输入
        BatchMatmulMaxSumKernel<bfloat16_t> kernel;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), kernel.matmulObj,
                          &tilingData.cubeTilingData);
        kernel.Init(x1, x2, y, workspace, tilingData, &pipe);
        kernel.Process();
    }
}
