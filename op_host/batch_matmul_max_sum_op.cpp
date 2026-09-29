// ============================================================
// BatchMatmulMaxSum 算子注册与框架适配（CANN 9.0 OpDef 风格）
// 职责：
//   1. 向图框架注册算子原型（输入/输出/属性及支持的数据类型）；
//   2. InferShape：推导输出 y 的 shape=[B]、dtype=float32、format=ND。
// 说明：CANN 9.0 推荐使用 OpDef 类 + register/op_def_registry.h 注册算子，
//       SetInferShape/SetInferDataType 完成推导，AICore().SetTiling 绑定 tiling。
// ============================================================
#include "register/op_def_registry.h"

namespace ge {

// ---------------- 输出 Shape 推导 ----------------
// 无论 transpose 属性如何取值，物理第 0 维恒为 B（赛题硬约束）。
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *x1Shape = context->GetInputShape(0);
    if (x1Shape == nullptr || x1Shape->GetDimNum() < 1) {
        return GRAPH_FAILED;
    }
    gert::Shape *yShape = context->GetOutputShape(0);
    if (yShape == nullptr) {
        return GRAPH_FAILED;
    }
    yShape->SetDimNum(1);
    yShape->SetDim(0, x1Shape->GetDim(0));
    return GRAPH_SUCCESS;
}

// ---------------- 输出 DataType 推导 ----------------
// y 恒为 FP32（赛题：Max/Sum 归约输出 FP32）。
static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, DT_FLOAT);
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class BatchMatmulMaxSum : public OpDef {
public:
    explicit BatchMatmulMaxSum(const char *name) : OpDef(name) {
        // 输入 x1：Query embedding，FP16/BF16
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        // 输入 x2：Document embedding，FP16/BF16
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        // 输出 y：相关性分数 [B]，FP32
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        // 属性 transposeX1 / transposeX2：仅改变物理存储排布，默认 false
        // 索引 0 -> transposeX1，索引 1 -> transposeX2
        this->Attr("transposeX1").AttrType(OPTIONAL).Bool(false);
        this->Attr("transposeX2").AttrType(OPTIONAL).Bool(false);

        // 推导函数绑定
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        // Tiling 绑定（optiling::TilingFunc 实现在 batch_matmul_max_sum_tiling.cpp）
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

// 注册算子到框架
OP_ADD(BatchMatmulMaxSum);

}  // namespace ops
