#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BatchMatmulMaxSum 算子 Golden 数据生成脚本

功能：
    1. 读取 testcase/test_cases.json 中定义的测试用例；
    2. 按用例随机生成逻辑张量 x1[B,M,K]、x2[B,K,N]；
    3. 严格按赛题要求计算 Golden：
         使用“输入实际存储值（先量化为 fp16/bf16）”在 FP64 精度下
         执行 bmm -> amax(N) -> sum(M)，最后转换为 FP32；
    4. 按 transposeX1/transposeX2 属性把张量排列成“物理存储布局”后落盘：
         transposeX1=false: x1 物理 shape [B,M,K]
         transposeX1=true : x1 物理 shape [B,K,M]
         transposeX2=false: x2 物理 shape [B,K,N]
         transposeX2=true : x2 物理 shape [B,N,K]
       —— 属性只改变物理排布，Golden 数学语义保持不变。

输出（每个用例一个子目录）：
    <output_dir>/<case_id>/x1.npy
    <output_dir>/<case_id>/x2.npy
    <output_dir>/<case_id>/y_golden.npy
    <output_dir>/<case_id>/meta.json

用法：
    python3 scripts/gen_golden.py                       # 生成全部用例
    python3 scripts/gen_golden.py --case case_basic_ff  # 生成指定用例
"""

import argparse
import json
import os

import numpy as np
import torch

# 工程根目录（scripts 的上一级），用于解析默认相对路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 随机输入取值范围（控制在 [-2, 2]，避免大 K 下累加值溢出 FP32）
RAND_LOW = -2.0
RAND_HIGH = 2.0


def load_test_cases(cases_file: str) -> list:
    """读取测试用例 json"""
    with open(cases_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["cases"]


def build_logical_inputs(case: dict) -> tuple:
    """
    按用例生成 FP64 逻辑张量，并量化为算子实际接收的低精度存储值。
    返回：
        x1_stored: 量化后的 torch 张量，逻辑 shape [B, M, K]
        x2_stored: 量化后的 torch 张量，逻辑 shape [B, K, N]
    """
    batch = case["B"]
    m = case["M"]
    n = case["N"]
    k = case["K"]
    dtype = case["dtype"]
    seed = int(case.get("seed", 0))

    # 使用固定种子，保证同一用例多次生成结果完全一致
    generator = torch.Generator().manual_seed(seed)

    # 先在 FP64 下生成均匀分布随机输入
    x1_logical = torch.rand(batch, m, k, generator=generator, dtype=torch.float64) \
        * (RAND_HIGH - RAND_LOW) + RAND_LOW
    x2_logical = torch.rand(batch, k, n, generator=generator, dtype=torch.float64) \
        * (RAND_HIGH - RAND_LOW) + RAND_LOW

    # 量化为算子实际接收的输入类型（fp16 / bf16）
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        raise ValueError("不支持的用例数据类型: {}".format(dtype))

    x1_stored = x1_logical.to(torch_dtype)
    x2_stored = x2_logical.to(torch_dtype)
    return x1_stored, x2_stored


def calc_golden(x1_stored: torch.Tensor, x2_stored: torch.Tensor) -> torch.Tensor:
    """
    按赛题规定计算 Golden：
        使用“输入实际存储值”在 FP64 下执行 bmm -> amax(dim=-1) -> sum(dim=-1)，
        最终输出 FP32。
    计算顺序固定为 MatMul -> Max(N) -> Sum(M)，不可交换。
    """
    x1_fp64 = x1_stored.to(torch.float64)
    x2_fp64 = x2_stored.to(torch.float64)

    similarity = torch.bmm(x1_fp64, x2_fp64)   # [B, M, N]
    max_sim = torch.amax(similarity, dim=-1)    # [B, M]：沿 N 维取 MaxSim
    y_golden = torch.sum(max_sim, dim=-1)       # [B]：沿 M 维求和
    return y_golden.to(torch.float32)


def to_physical_layout(tensor: torch.Tensor, transpose: bool) -> torch.Tensor:
    """
    根据 transpose 属性把逻辑张量排列为物理存储布局：
        transpose=false：保持逻辑布局
        transpose=true ：交换最后两个维度（物理上按转置后的 shape 连续存储）
    注意：这只是“准备物理输入数据”，算子 Kernel 内部不做转置运算，
         只按属性选择对应的访存索引。
    """
    if transpose:
        return tensor.transpose(-1, -2).contiguous()
    return tensor.contiguous()


def save_tensor_as_npy(tensor: torch.Tensor, path: str, dtype_name: str) -> None:
    """
    保存张量为 npy：
        fp16 直接使用 numpy float16；
        bf16 numpy 无原生支持，保存其 16bit 位模式（int16），
        由消费方按 meta.json 中的 dtype=bfloat16 还原。
    """
    if dtype_name == "float16":
        np.save(path, tensor.numpy().astype(np.float16))
    else:
        # bfloat16：取底层 16 位比特按 int16 保存，数值语义不变
        bits = tensor.contiguous().view(torch.int16).numpy()
        np.save(path, bits)


def gen_one_case(case: dict, output_dir: str) -> None:
    """生成单个用例的全部输入文件与 Golden 文件"""
    case_id = case["case_id"]
    dtype_name = case["dtype"]
    transpose_x1 = bool(case["transpose_x1"])
    transpose_x2 = bool(case["transpose_x2"])

    x1_stored, x2_stored = build_logical_inputs(case)
    y_golden = calc_golden(x1_stored, x2_stored)

    # 按属性转换为物理存储布局（仅数据排布变化）
    x1_physical = to_physical_layout(x1_stored, transpose_x1)
    x2_physical = to_physical_layout(x2_stored, transpose_x2)

    case_dir = os.path.join(output_dir, case_id)
    os.makedirs(case_dir, exist_ok=True)

    save_tensor_as_npy(x1_physical, os.path.join(case_dir, "x1.npy"), dtype_name)
    save_tensor_as_npy(x2_physical, os.path.join(case_dir, "x2.npy"), dtype_name)
    np.save(os.path.join(case_dir, "y_golden.npy"), y_golden.numpy().astype(np.float32))

    # 落盘用例元信息，供 Host 侧测试 / 结果比对使用
    meta = {
        "case_id": case_id,
        "description": case.get("description", ""),
        "B": case["B"],
        "M": case["M"],
        "N": case["N"],
        "K": case["K"],
        "dtype": dtype_name,
        "transpose_x1": transpose_x1,
        "transpose_x2": transpose_x2,
        "x1_physical_shape": list(x1_physical.shape),
        "x2_physical_shape": list(x2_physical.shape),
    }
    with open(os.path.join(case_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[OK] {:<24s} B={} M={} N={} K={:<5d} transpose=({},{}) dtype={}".format(
        case_id, case["B"], case["M"], case["N"], case["K"],
        transpose_x1, transpose_x2, dtype_name))


def main():
    parser = argparse.ArgumentParser(description="BatchMatmulMaxSum Golden 数据生成")
    parser.add_argument("--cases_file",
                        default=os.path.join(PROJECT_ROOT, "testcase", "test_cases.json"),
                        help="测试用例 json 路径")
    parser.add_argument("--output_dir",
                        default=os.path.join(PROJECT_ROOT, "scripts", "data"),
                        help="生成数据输出目录")
    parser.add_argument("--case", default="all",
                        help="指定用例 case_id，默认 all 生成全部用例")
    args = parser.parse_args()

    cases = load_test_cases(args.cases_file)
    if args.case != "all":
        cases = [c for c in cases if c["case_id"] == args.case]
        if not cases:
            raise FileNotFoundError("在用例文件中找不到 case_id={}".format(args.case))

    os.makedirs(args.output_dir, exist_ok=True)
    print("开始生成 {} 个用例，输出目录：{}\n".format(len(cases), args.output_dir))
    for case in cases:
        gen_one_case(case, args.output_dir)
    print("\n全部用例生成完成。")


if __name__ == "__main__":
    main()
