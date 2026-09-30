# ============================================================
# BatchMatmulMaxSum Ascend-C 自定义算子主编译配置
# 产物：libbatch_matmul_max_sum.so
# 说明：Host 侧使用 g++ 编译，Device 侧 Kernel 使用 CANN 9.0 自带
#       bisheng 编译器编译，最终链接为同一个算子动态库。
# ============================================================
cmake_minimum_required(VERSION 3.16)
project(BatchMatmulMaxSum CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CMake >= 3.31 默认向链接器传递 --dependency-file，毕昇链接器不支持，显式关闭
set(CMAKE_LINK_DEPENDS_USE_LINKER FALSE)

# ---------- 1. 定位 CANN Toolkit ----------
if(NOT DEFINED ASCEND_HOME_PATH)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
    elseif(DEFINED ENV{ASCEND_TOOLKIT_HOME})
        set(ASCEND_HOME_PATH "$ENV{ASCEND_TOOLKIT_HOME}")
    else()
        set(ASCEND_HOME_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    endif()
endif()

if(NOT EXISTS "${ASCEND_HOME_PATH}/include")
    message(FATAL_ERROR "未找到 CANN 头文件目录：${ASCEND_HOME_PATH}/include，请先 source set_env.sh 或通过 -DASCEND_HOME_PATH 指定")
endif()
message(STATUS "ASCEND_HOME_PATH = ${ASCEND_HOME_PATH}")

# 系统架构前缀，如 aarch64-linux / x86_64-linux
set(SYSTEM_PREFIX "${CMAKE_SYSTEM_PROCESSOR}-linux")

# ---------- 2. AI Core 目标架构 ----------
# Atlas A2（910B/910_93）对应 bisheng --npu-arch=dav-2201
if(NOT DEFINED ASCEND_ARCH)
    set(ASCEND_ARCH "dav-2201" CACHE STRING "bisheng --npu-arch, e.g. dav-2201 for 910B")
endif()
message(STATUS "ASCEND_ARCH (bisheng --npu-arch) = ${ASCEND_ARCH}")

# ---------- 3. 头文件目录 ----------
set(HOST_INC_DIRS
    "${ASCEND_HOME_PATH}/include"
    "${ASCEND_HOME_PATH}/include/aclnn"
    "${ASCEND_HOME_PATH}/include/exe_graph"
)

# 不同 CANN 版本的 opp proto 头文件目录有差异，存在哪个用哪个
set(OPP_INC_CANDIDATES
    "${ASCEND_HOME_PATH}/opp/built-in/op_proto/inc"
    "${ASCEND_HOME_PATH}/opp/opp_proto/inc"
    "${ASCEND_HOME_PATH}/opp/op_proto/inc"
)
foreach(_opp_inc ${OPP_INC_CANDIDATES})
    if(EXISTS "${_opp_inc}")
        list(APPEND HOST_INC_DIRS "${_opp_inc}")
    endif()
endforeach()

# ---------- 4. Host 侧编译（tiling + 算子注册） ----------
set(HOST_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/op_host/batch_matmul_max_sum_tiling.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_host/batch_matmul_max_sum_op.cpp"
)

add_library(op_host_obj OBJECT ${HOST_SRCS})
target_include_directories(op_host_obj PRIVATE
    ${HOST_INC_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/op_host"
)
target_compile_options(op_host_obj PRIVATE -O2 -Wall -Wextra)
# CANN 9.0 Host 侧库使用 _GLIBCXX_USE_CXX11_ABI=0（旧 ABI）编译，
# 若不匹配会导致 std::string/std::list 等符号 undefined reference。
target_compile_definitions(op_host_obj PRIVATE _GLIBCXX_USE_CXX11_ABI=0)

# ---------- 5. Device 侧 Kernel 编译（bisheng） ----------
set(KERNEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/batch_matmul_max_sum_kernel.cpp")
set(KERNEL_OBJ "${CMAKE_BINARY_DIR}/batch_matmul_max_sum_kernel.o")

# bisheng 是 CANN 9.0 的 Ascend-C 编译器，替代旧版 ccec
find_program(BISHENG_BIN bisheng
    PATHS
        "${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin"
        "${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/ccec_compiler/bin"
        "${ASCEND_HOME_PATH}/bin"
    NO_DEFAULT_PATH
)
if(NOT BISHENG_BIN)
    set(BISHENG_BIN "bisheng")
endif()
message(STATUS "bisheng compiler = ${BISHENG_BIN}")

# Kernel 编译参数（注意：必须用 CMake 列表分号分隔成两个独立参数，
# 若写成一个带空格的字符串，VERBATIM 下会被整体引号包住，bisheng 无法解析）：
#   --npu-arch=dav-2201  指定 910B AICore 架构
#   -xasc                启用 Ascend-C 语言模式
set(KERNEL_COMPILE_FLAGS "--npu-arch=${ASCEND_ARCH}" "-xasc")

add_custom_command(
    OUTPUT ${KERNEL_OBJ}
    COMMAND ${BISHENG_BIN}
        -c -O2
        ${KERNEL_COMPILE_FLAGS}
        -I${ASCEND_HOME_PATH}/include
        -I${CMAKE_CURRENT_SOURCE_DIR}/op_host
        ${KERNEL_SRC}
        -o ${KERNEL_OBJ}
    DEPENDS ${KERNEL_SRC}
    COMMENT "使用 bisheng 编译 Ascend-C 融合 Kernel (npu-arch=${ASCEND_ARCH})"
    VERBATIM
)
add_custom_target(op_kernel_target DEPENDS ${KERNEL_OBJ})

# ---------- 6. 汇总链接为算子动态库 ----------
add_library(batch_matmul_max_sum SHARED
    $<TARGET_OBJECTS:op_host_obj>
    ${KERNEL_OBJ}
)
add_dependencies(batch_matmul_max_sum op_kernel_target)

target_link_directories(batch_matmul_max_sum PRIVATE "${ASCEND_HOME_PATH}/lib64")
# CANN 9.0 Host 侧框架库：
#   register   — OpDef / OP_ADD 算子注册
#   graph      — ge 命名空间（graphStatus、DT_FLOAT16 等）
#   gert       — gert 命名空间（TilingContext、Shape 等）
#   tiling_api — matmul_tiling / optiling::TCubeTiling
#   platform   — platform_ascendc::PlatformAscendC
#   ascendcl   — ACL 运行时
#   dl         — 动态加载
target_link_libraries(batch_matmul_max_sum PRIVATE
    register
    graph
    gert
    tiling_api
    platform
    ascendcl
    dl
)

# ---------- 7. 安装规则 ----------
install(TARGETS batch_matmul_max_sum
    LIBRARY DESTINATION lib
)
