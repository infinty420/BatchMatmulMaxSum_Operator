#!/bin/bash
# ============================================================
# BatchMatmulMaxSum 算子一键编译脚本
# 用法：
#   ./build.sh          # 编译 Release 版本算子 so
#   ./build.sh clean    # 清理构建目录
# 依赖：
#   已安装 CANN Toolkit（本赛题为 CANN 9.0.0），
#   或已手动 source <toolkit>/set_env.sh
# ============================================================
set -e

# ---------- 1. 自动加载 CANN 环境变量 ----------
# 注意：必须完整 source set_env.sh，bisheng 编译依赖 LD_LIBRARY_PATH、
# ASCEND_OPP_PATH 等多个环境变量，仅设置 ASCEND_HOME_PATH 不够。
ENV_SCRIPT=""
for candidate in \
    "${ASCEND_HOME_PATH}/set_env.sh" \
    "/usr/local/Ascend/ascend-toolkit/set_env.sh" \
    "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh" \
    "/usr/local/Ascend/cann-9.0.0/set_env.sh" \
    "$HOME/Ascend/cann-9.0.0/set_env.sh"; do
    if [ -n "${candidate}" ] && [ -f "${candidate}" ]; then
        ENV_SCRIPT="${candidate}"
        break
    fi
done

if [ -z "${ENV_SCRIPT}" ]; then
    echo "[ERROR] 未找到 CANN set_env.sh，请先执行：source <CANN Toolkit>/set_env.sh"
    exit 1
fi

echo "[INFO] 加载 CANN 环境：${ENV_SCRIPT}"
source "${ENV_SCRIPT}"

if [ -z "${ASCEND_HOME_PATH}" ]; then
    echo "[ERROR] 未检测到 ASCEND_HOME_PATH，请检查 CANN 安装"
    exit 1
fi

# ---------- 2. 目标 AI Core 架构（默认 Atlas A2 / 910B，bisheng --npu-arch=dav-2201） ----------
export ASCEND_ARCH="${ASCEND_ARCH:-dav-2201}"

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"

# ---------- 3. 清理模式 ----------
if [ "$1" = "clean" ]; then
    rm -rf "${BUILD_DIR}"
    echo "[INFO] 已清理 ${BUILD_DIR}"
    exit 0
fi

# ---------- 4. CMake 配置 + 编译 ----------
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${PROJECT_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DASCEND_HOME_PATH="${ASCEND_HOME_PATH}" \
    -DASCEND_ARCH="${ASCEND_ARCH}"

make -j"$(nproc)"

echo ""
echo "[SUCCESS] 算子编译完成：${BUILD_DIR}/libbatch_matmul_max_sum.so"
