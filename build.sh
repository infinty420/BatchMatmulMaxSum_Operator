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
if [ -z "${ASCEND_HOME_PATH}" ]; then
    for env_script in \
        "/usr/local/Ascend/ascend-toolkit/set_env.sh" \
        "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh" \
        "/usr/local/Ascend/cann-9.0.0/set_env.sh"; do
        if [ -f "${env_script}" ]; then
            source "${env_script}"
            break
        fi
    done
fi

if [ -z "${ASCEND_HOME_PATH}" ]; then
    echo "[ERROR] 未检测到 CANN 环境，请先执行：source <CANN Toolkit>/set_env.sh"
    exit 1
fi

# ---------- 2. 目标 AI Core 架构（默认 Atlas A2 / 910B） ----------
export ASCEND_ARCH="${ASCEND_ARCH:-dav-c220}"

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
