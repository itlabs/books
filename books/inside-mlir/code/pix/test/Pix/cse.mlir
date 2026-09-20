// RUN: pix-opt %s --cse | FileCheck %s
//
// 我们没有为 pix.add 写一行优化代码，但它声明了 Pure（第 5 章），
// 于是通用的 --cse 就能合并重复计算。这个测试守住那个声明的价值：
// 两次相同的 pix.add 必须被合并成一次。

// CHECK-LABEL: func.func @cse_two_adds
// CHECK: %[[S:.*]] = pix.add %arg0, %arg1
// CHECK: pix.add %[[S]], %[[S]]
// CHECK-NOT: pix.add
func.func @cse_two_adds(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %s1 = pix.add %a, %b : tensor<4x4xf32>
  %s2 = pix.add %a, %b : tensor<4x4xf32>
  %out = pix.add %s1, %s2 : tensor<4x4xf32>
  return %out : tensor<4x4xf32>
}
