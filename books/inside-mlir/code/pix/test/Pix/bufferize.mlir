// 第 22 章：从 tensor 的值语义走到 memref 的内存语义。
//
// RUN: pix-opt %s --pix-to-linalg \
// RUN:   --one-shot-bufferize="bufferize-function-boundaries=true" \
// RUN:   --canonicalize | FileCheck %s
//
// 跳过原地分析做对照：每次写都先拷一份，alloc 和 copy 都会明显变多。
// RUN: pix-opt %s --pix-to-linalg \
// RUN:   --one-shot-bufferize="bufferize-function-boundaries=true copy-before-write=true" \
// RUN:   --canonicalize | FileCheck %s --check-prefix=NOANALYSIS

//===----------------------------------------------------------------------===//
// 基本形态：tensor → memref，linalg.generic 不再有结果（写进 outs）
//===----------------------------------------------------------------------===//

// 函数签名也被 bufferize 了（bufferize-function-boundaries=true）。
// 入参带 strided 布局，因为调用者可能传进来一个视图。
// CHECK-LABEL: func.func @two_steps
// CHECK-SAME: memref<8x8xf32, strided<[?, ?], offset: ?>>
// tensor.empty 变成了真正的分配
// CHECK: memref.alloc()
// generic 没有返回值了
// CHECK: linalg.generic
// CHECK-NOT: -> tensor
func.func @two_steps(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>, %f: f32) -> tensor<8x8xf32> {
  %s = pix.add %a, %b : tensor<8x8xf32>
  %m = pix.scale %s, %f : tensor<8x8xf32>
  return %m : tensor<8x8xf32>
}

// 跳过分析时，同一个函数也能 bufferize —— 分析影响的是"多少拷贝"，不是"能不能"。
// NOANALYSIS-LABEL: func.func @two_steps
// NOANALYSIS: memref.alloc()
