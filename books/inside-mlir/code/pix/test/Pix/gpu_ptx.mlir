// RUN: pix-opt %s --convert-gpu-to-nvvm --gpu-module-to-binary=format=isa | FileCheck %s
//
// 第 25 章：真的出 PTX。断言的是汇编内容本身 —— 这是全书唯一一处断言
// **目标机器汇编**的测试。
//
// 注意：这一步需要构建时带上 NVPTX 后端（LLVM_TARGETS_TO_BUILD 含 NVPTX）。
// 发行版的 mlir-opt 通常带，但自己编的最小构建可能没有，那样会得到
// "Failed to create target machine" 之类的报错。附录 A 第 0 节说明了基线构建的配置。

// CHECK: gpu.binary @kernels
// PTX 的头三样：版本、目标架构、地址宽度
// CHECK-SAME: .version
// CHECK-SAME: .target sm_80
// CHECK-SAME: .address_size 64
// thread id 变成 PTX 的特殊寄存器
// CHECK-SAME: mov.u32{{.*}}%tid.x
// 存到 global 地址空间
// CHECK-SAME: st.global

module attributes {gpu.container_module} {
  gpu.module @kernels [#nvvm.target<chip = "sm_80">] {
    gpu.func @fill(%out: memref<8xf32>, %v: f32) kernel {
      %tx = gpu.thread_id x
      memref.store %v, %out[%tx] : memref<8xf32>
      gpu.return
    }
  }
}
