// ============================================================
// BatchMatmulMaxSum 算子注册与框架适配
// 职责：
//   1. 向图框架注册算子原型（输入/输出/属性及支持的数据类型）；
//   2. InferShape：推导输出 y 的 shape=[B]、dtype=float32、format=ND。
// 说明：msopgen 标准工程通常把 REG_OP 段放在同名 .h 中；本工程为
//       保持提交目录精简，将其与 InferShape 一并放在本文件（单一编译
//       单元，不违反 ODR）。msopgen_op.json 与其保持同源一致。
// ============================================================
#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

// ---------------- 算子原型注册 ----------------
REG_OP(BatchMatmulMaxSum)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16}))     // Query embedding
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16}))     // Document embedding
    .OUTPUT(y, TensorType({DT_FLOAT}))                // 相关性分数 [B]
    .OPTIONAL_ATTR(transposeX1, Bool, false)          // 仅声明 x1 物理布局
    .OPTIONAL_ATTR(transposeX2, Bool, false)          // 仅声明 x2 物理布局
    .OP_END_FACTORY_REG(BatchMatmulMaxSum)

// ---------------- 输出 Shape / DataType 推导 ----------------
IMPLEMT_COMMON_INFERFUNC(BatchMatmulMaxSumInfer) {
    // 无论 transpose 属性如何取值，物理第 0 维恒为 B
    Shape x1Shape = op.GetInputDesc("x1").GetShape();
    Shape yShape({x1Shape.GetDim(0)});

    TensorDesc yDesc = op.GetOutputDescByName("y");
    yDesc.SetShape(yShape);
    yDesc.SetDataType(DT_FLOAT);
    yDesc.SetFormat(FORMAT_ND);
    return op.UpdateOutputDesc("y", yDesc);
}

// InferShape 函数绑定到算子类型
COMM_INFER_FUNC_REG(BatchMatmulMaxSum, BatchMatmulMaxSumInfer);

}  // namespace ge
