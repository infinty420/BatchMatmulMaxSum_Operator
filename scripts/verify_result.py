#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BatchMatmulMaxSum 算子结果精度校验脚本

功能：
    读取算子实际输出 y_output（FP32, shape [B]）与 Golden y_golden
    （FP64 计算后转 FP32），按赛题“双千分之一”精度要求逐元素比对，
    输出误差统计报告，并以退出码反映校验结论（0 通过 / 1 不通过）。

判定规则（逐元素，满足其一即视为该元素通过）：
    绝对误差 |out - golden| <= atol
    相对误差 |out - golden| / max(|golden|, eps) <= rtol
赛题阈值：FP16/BF16 用例 rtol=atol=1e-3；FP32 用例 1e-4。

用法：
    python3 scripts/verify_result.py \
        --golden scripts/data/case_basic_ff/y_golden.npy \
        --output scripts/data/case_basic_ff/y_output.npy
"""

import argparse
import os
import sys

import numpy as np

# 防除零小量
EPS = 1e-12


def load_output_tensor(path: str) -> np.ndarray:
    """加载 npy 张量并转为 float32 一维数组"""
    if not os.path.exists(path):
        raise FileNotFoundError("文件不存在: {}".format(path))
    arr = np.load(path).astype(np.float32)
    return arr.reshape(-1)


def verify(golden: np.ndarray, output: np.ndarray, rtol: float, atol: float) -> dict:
    """
    逐元素计算绝对/相对误差并生成校验报告数据。
    """
    if golden.shape != output.shape:
        raise ValueError("shape 不一致：golden {} vs output {}".format(golden.shape, output.shape))

    abs_diff = np.abs(output - golden)
    # 相对误差分母使用 |golden|，golden 为 0 时用 EPS 兜底
    rel_diff = abs_diff / np.maximum(np.abs(golden), EPS)

    # 逐元素：绝对误差或相对误差任一达标即通过（np.allclose 标准语义）
    abs_pass = abs_diff <= atol
    rel_pass = rel_diff <= rtol
    element_pass = abs_pass | rel_pass

    # 全负相似度等场景中 golden 恰为 0 时，只看绝对误差，避免相对误差虚高
    return {
        "total": int(golden.size),
        "passed": int(np.count_nonzero(element_pass)),
        "max_abs_err": float(np.max(abs_diff)) if abs_diff.size else 0.0,
        "mean_abs_err": float(np.mean(abs_diff)) if abs_diff.size else 0.0,
        "max_rel_err": float(np.max(rel_diff)) if rel_diff.size else 0.0,
        "mean_rel_err": float(np.mean(rel_diff)) if rel_diff.size else 0.0,
        "element_pass": element_pass,
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "golden": golden,
        "output": output,
    }


def print_report(case_name: str, report: dict, rtol: float, atol: float) -> None:
    """打印人类可读的校验报告"""
    total = report["total"]
    passed = report["passed"]
    all_pass = passed == total

    print("=" * 64)
    print("BatchMatmulMaxSum 精度校验报告  用例: {}".format(case_name))
    print("=" * 64)
    print("判定阈值        : rtol={}, atol={}".format(rtol, atol))
    print("元素总数        : {}".format(total))
    print("通过元素数      : {} / {}".format(passed, total))
    print("最大绝对误差    : {:.6e}".format(report["max_abs_err"]))
    print("平均绝对误差    : {:.6e}".format(report["mean_abs_err"]))
    print("最大相对误差    : {:.6e}".format(report["max_rel_err"]))
    print("平均相对误差    : {:.6e}".format(report["mean_rel_err"]))
    print("-" * 64)

    if all_pass:
        print("结论: PASS  全部元素满足双千分之一精度要求")
    else:
        fail_idx = np.flatnonzero(~report["element_pass"])
        print("结论: FAIL  有 {} 个元素超阈值，前若干个失败元素明细：".format(fail_idx.size))
        print("{:>6} {:>16} {:>16} {:>12} {:>12}".format(
            "index", "golden", "output", "abs_err", "rel_err"))
        for idx in fail_idx[:10]:
            print("{:>6} {:>16.6f} {:>16.6f} {:>12.3e} {:>12.3e}".format(
                idx, report["golden"][idx], report["output"][idx],
                report["abs_diff"][idx], report["rel_diff"][idx]))
    print("=" * 64)


def main():
    parser = argparse.ArgumentParser(description="BatchMatmulMaxSum 结果精度校验")
    parser.add_argument("--golden", required=True, help="Golden y_golden.npy 路径")
    parser.add_argument("--output", required=True, help="算子输出 y_output.npy 路径")
    parser.add_argument("--rtol", type=float, default=1e-3, help="相对误差阈值，默认 1e-3")
    parser.add_argument("--atol", type=float, default=1e-3, help="绝对误差阈值，默认 1e-3")
    args = parser.parse_args()

    golden = load_output_tensor(args.golden)
    output = load_output_tensor(args.output)

    case_name = os.path.basename(os.path.dirname(args.golden))
    report = verify(golden, output, args.rtol, args.atol)
    print_report(case_name, report, args.rtol, args.atol)

    sys.exit(0 if report["passed"] == report["total"] else 1)


if __name__ == "__main__":
    main()
