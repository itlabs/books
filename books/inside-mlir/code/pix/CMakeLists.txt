# pix 方言 —— 《深入 MLIR》第 7 章起的贯穿案例（out-of-tree MLIR 项目）
#
# 构建（也可以直接用 books/inside-mlir/verify/verify-pix.sh）：
#   cmake -G Ninja -S . -B build \
#     -DMLIR_DIR=<llvm-build>/lib/cmake/mlir \
#     -DLLVM_DIR=<llvm-build>/lib/cmake/llvm \
#     -DCMAKE_BUILD_TYPE=Release
#   ninja -C build          # 得到 build/bin/pix-opt
#   ninja -C build check-pix # 跑 lit 测试（第 12 章）
#
# 这份骨架的结构照着上游 mlir/examples/standalone 裁剪而来（去掉了 Python 绑定、
# C API 与 plugin），第 7 章会把每一块的作用讲清楚。
cmake_minimum_required(VERSION 3.20.0)
project(pix-dialect LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

# 这一句是 out-of-tree 项目的全部秘密：从 MLIR_DIR 找到 MLIRConfig.cmake，
# 它会把 MLIR 的头文件路径、库、以及 add_mlir_dialect 这些 CMake 函数都带进来。
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_LIT_ARGS "-sv" CACHE STRING "Default options for lit")
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)          # mlir_tablegen / add_public_tablegen_target
include(AddLLVM)           # add_llvm_executable
include(AddMLIR)           # add_mlir_dialect / add_mlir_dialect_library
include(HandleLLVMOptions) # 让本项目用与 LLVM 一致的编译选项

set(PIX_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(PIX_BINARY_DIR ${PROJECT_BINARY_DIR})

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PIX_SOURCE_DIR}/include)
include_directories(${PIX_BINARY_DIR}/include)  # TableGen 生成的 .inc 落在这里
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(pix-opt)
add_subdirectory(test)
