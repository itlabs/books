// 第 25 章：gpu 方言、kernel outlining、以及三个出口。
//
// RUN: pix-opt %s --gpu-kernel-outlining | FileCheck %s --check-prefix=OUTLINE
// RUN: pix-opt %s --gpu-kernel-outlining --convert-gpu-to-nvvm | FileCheck %s --check-prefix=NVVM

//===----------------------------------------------------------------------===//
// outlining：把 gpu.launch 的 region 抽成一个独立的 gpu.func
//
// 关键在于**捕获**：region 里引用了外面的 %v 和 %out，而 gpu.func 必须是
// IsolatedFromAbove（它要被编译成独立的 kernel），所以那些值被提成了显式参数。
// 第 13 章讲 IsolatedFromAbove 时说"代价是不能被直接抽成独立函数"，
// 这里是同一枚硬币的正面：**gpu.launch 故意不隔离，好让你写起来自然；
// outlining 负责把它变成隔离的形式。**
//===----------------------------------------------------------------------===//

// 模块被打上 gpu.container_module 标记
// OUTLINE: module attributes {gpu.container_module}
// 原来的 launch 变成一次调用，捕获的值作为 args 传进去
// OUTLINE: gpu.launch_func @launch_kernel::@launch_kernel
// OUTLINE-SAME: args(%cst : f32, %arg0 : memref<8xf32>)
// kernel 自己住在一个 gpu.module 里
// OUTLINE: gpu.module @launch_kernel
// OUTLINE: gpu.func @launch_kernel(%arg0: f32, %arg1: memref<8xf32>) kernel

//===----------------------------------------------------------------------===//
// 出口一：NVVM。注意 memref 参数展成了 descriptor 的五个标量
// （第 24 章那个 1:N），而 thread id 变成了真正的 PTX 特殊寄存器读取。
//===----------------------------------------------------------------------===//

// NVVM: llvm.func @launch_kernel(%arg0: f32, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64, %arg4: i64, %arg5: i64)
// NVVM-SAME: nvvm.kernel
// NVVM: nvvm.read.ptx.sreg.tid.x

func.func @launch(%out: memref<8xf32>) {
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %v = arith.constant 1.5 : f32
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %c1, %gy = %c1, %gz = %c1)
             threads(%tx, %ty, %tz) in (%bsx = %c8, %bsy = %c1, %bsz = %c1) {
    memref.store %v, %out[%tx] : memref<8xf32>
    gpu.terminator
  }
  return
}
