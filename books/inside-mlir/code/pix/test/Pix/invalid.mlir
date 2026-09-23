// RUN: pix-opt %s --split-input-file --verify-diagnostics
//
// 给"报错行为"上测试（第 3 章讲过这套写法，第 10 章细讲）。
// 这里守的是 SameOperandsAndResultType 这个 trait 自动生成的检查：
// 我们一行验证代码都没写，但类型不一致必须被拦住。

func.func @shape_mismatch(%a: tensor<4x4xf32>, %b: tensor<8x8xf32>) -> tensor<4x4xf32> {
  // expected-error @+1 {{requires the same type for all operands and results}}
  %sum = "pix.add"(%a, %b) : (tensor<4x4xf32>, tensor<8x8xf32>) -> tensor<4x4xf32>
  return %sum : tensor<4x4xf32>
}

// -----

func.func @wrong_element_type(%a: tensor<4x4xi32>, %b: tensor<4x4xi32>) -> tensor<4x4xi32> {
  // expected-error @+1 {{operand #0 must be tensor of 32-bit float values}}
  %sum = "pix.add"(%a, %b) : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
  return %sum : tensor<4x4xi32>
}
