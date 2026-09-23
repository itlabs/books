// RUN: pix-opt %s --pix-drr | FileCheck %s
//
// 第 15 章：这些改写一条 C++ 都没写，全部来自 PixPatterns.td 里的 Pat<>。
// 效果必须和第 11 章的 C++ 版、第 14 章的手写 pattern 完全一致——
// 这个文件守的就是"声明式和手写式等价"。

// ① 纯结构改写：DRR 的主场，一行 Pat<> 顶十几行 C++
// CHECK-LABEL: func.func @twice
// CHECK-NEXT: return %arg0 :
// CHECK-NOT: pix.transpose
func.func @twice(%img: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %t1 = pix.transpose %img : tensor<4x8xf32> to tensor<8x4xf32>
  %t2 = pix.transpose %t1 : tensor<8x4xf32> to tensor<4x8xf32>
  return %t2 : tensor<4x8xf32>
}

// 三条链式转置：pattern 在**内层那一对**上触发，塌掉两条，剩一条。
// 奇数条转置必然剩一条，偶数条必然全消——这是"匹配内层"带来的直接结果，
// 也说明 DRR 的 DAG 匹配和 greedy driver 的遍历顺序是两件独立的事。
// CHECK-LABEL: func.func @odd_chain
// CHECK-NEXT: %[[T:.*]] = pix.transpose %arg0
// CHECK-NEXT: return %[[T]] :
func.func @odd_chain(%img: tensor<4x8xf32>) -> tensor<8x4xf32> {
  %t1 = pix.transpose %img : tensor<4x8xf32> to tensor<8x4xf32>
  %t2 = pix.transpose %t1 : tensor<8x4xf32> to tensor<4x8xf32>
  %t3 = pix.transpose %t2 : tensor<4x8xf32> to tensor<8x4xf32>
  return %t3 : tensor<8x4xf32>
}

// ③ 嵌套结果 + NativeCodeCall 建 arith.mulf
// CHECK-LABEL: func.func @nested
// CHECK: %[[M:.*]] = arith.mulf %arg1, %arg2
// CHECK: pix.scale %arg0, %[[M]] :
// CHECK-NOT: pix.scale
func.func @nested(%img: tensor<4x4xf32>, %a: f32, %b: f32) -> tensor<4x4xf32> {
  %s1 = pix.scale %img, %a : tensor<4x4xf32>
  %s2 = pix.scale %s1, %b : tensor<4x4xf32>
  return %s2 : tensor<4x4xf32>
}

// ④ 要凭空造一个常量：DRR 的天花板，只能靠 NativeCodeCall
// CHECK-LABEL: func.func @self
// CHECK: %[[C:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: pix.scale %arg0, %[[C]] :
// CHECK-NOT: pix.add
func.func @self(%img: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %r = pix.add %img, %img : tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}

// 两个操作数不同 → SameValue 约束挡住，不动
// CHECK-LABEL: func.func @two_operands
// CHECK: pix.add %arg0, %arg1
func.func @two_operands(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %r = pix.add %a, %b : tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}
