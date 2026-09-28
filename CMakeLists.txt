# ============================================================
# BatchMatmulMaxSum Ascend-C 自定义算子主编译配置
# 产物：libbatch_matmul_max_sum.so
# 说明：Host 侧使用 g++ 编译，Device 侧 Kernel 使用 CANN 自带
#       ccec 编译器编译，最终链接为同一个算子动态库。
# ============================================================
cmake_minimum_required(VERSION 3.16)
project(BatchMatmulMaxSum CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------- 1. 定位 CANN Toolkit ----------
# 优先使用 CMake 传入参数，其次环境变量，最后使用默认安装路径
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

# ---------- 2. AI Core 目标架构 ----------
# Atlas A2 训练系列（Ascend 910B）默认为 dav-c220；
# 若评测环境为其他硬件，可通过 -DASCEND_ARCH=xxx 覆盖。
if(NOT DEFINED ASCEND_ARCH)
    set(ASCEND_ARCH "dav-c220" CACHE STRING "AI Core architecture, e.g. dav-c220")
endif()
message(STATUS "ASCEND_ARCH     = ${ASCEND_ARCH}")

# ---------- 3. Host 侧头文件目录 ----------
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

# ---------- 5. Device 侧 Kernel 编译（ccec） ----------
set(KERNEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/batch_matmul_max_sum_kernel.cpp")
set(KERNEL_OBJ "${CMAKE_BINARY_DIR}/batch_matmul_max_sum_kernel.o")

find_program(CCEC_BIN ccec
    PATHS "${ASCEND_HOME_PATH}/compiler/ccec_compiler/bin"
    NO_DEFAULT_PATH
)
if(NOT CCEC_BIN)
    set(CCEC_BIN "ccec")
endif()

# 拼接 Kernel 编译需要的 -I 参数：
#   toolkit 根 include（kernel_operator、matmul 高阶 API）
#   op_host 目录（Host/Device 共用的 TilingData 契约头）
set(KERNEL_INC_FLAGS
    "-I${ASCEND_HOME_PATH}/include"
    "-I${CMAKE_CURRENT_SOURCE_DIR}/op_host"
)

add_custom_command(
    OUTPUT ${KERNEL_OBJ}
    COMMAND ${CCEC_BIN}
        -c -fpic -O2
        -mllvm --aarch64
        -mllvm -cce-aicore-arch=${ASCEND_ARCH}
        ${KERNEL_INC_FLAGS}
        ${KERNEL_SRC}
        -o ${KERNEL_OBJ}
    DEPENDS ${KERNEL_SRC}
    COMMENT "使用 ccec 编译 Ascend-C 融合 Kernel (arch=${ASCEND_ARCH})"
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
# 以下均为 CANN Host 侧框架基础库，随 Toolkit 发布
target_link_libraries(batch_matmul_max_sum PRIVATE
    opsproto
    register
    slog
    ge_common
    error_manager
)

# ---------- 7. 安装规则 ----------
install(TARGETS batch_matmul_max_sum
    LIBRARY DESTINATION lib
)
