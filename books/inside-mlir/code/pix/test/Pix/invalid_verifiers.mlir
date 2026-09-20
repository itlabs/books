// RUN: pix-opt %s --split-input-file --verify-diagnostics
//
// 第 10 章：给三个"不变式"上测试。它们都是 ODS 声明拦不住、必须自己守的规则。
// expected-error 精确预言报错文本——哪天 verifier 不报了，这个测试立刻失败。

func.func @transpose_not_reversed(%a: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // expected-error @+1 {{结果形状必须是输入形状的逆序}}
  %t = pix.transpose %a : tensor<4x8xf32> to tensor<4x8xf32>
  return %t : tensor<4x8xf32>
}

// -----

func.func @kernel_shape_mismatch() -> !pix.kernel<5 x 5> {
  // expected-error @+1 {{权重形状与卷积核类型不一致}}
  %k = pix.kernel dense<1.0> : tensor<3x3xf32> -> !pix.kernel<5 x 5>
  return %k : !pix.kernel<5 x 5>
}

// -----

func.func @pipeline_yields_nothing() -> tensor<4x4xf32> {
  // 这条检查不是手写的：实现了 RegionBranchOpInterface 之后由通用机制给出
  // expected-error @+1 {{source has 0 operands, but target successor needs 1}}
  %out = pix.pipeline : tensor<4x4xf32> {
  }
  return %out : tensor<4x4xf32>
}

// -----

// expected-error @+1 {{卷积核的行列必须是奇数}}
func.func @even_kernel(%k: !pix.kernel<2 x 4>) {
  return
}

// -----

// 合法的写法必须仍然通过（否则 verifier 写得太严）
func.func @ok(%a: tensor<4x8xf32>, %img: tensor<8x8xf32>) -> tensor<8x4xf32> {
  %t = pix.transpose %a : tensor<4x8xf32> to tensor<8x4xf32>
  %k = pix.kernel dense<1.0> : tensor<3x3xf32> -> !pix.kernel<3 x 3>
  %c = pix.convolve %img, %k : !pix.kernel<3 x 3>, tensor<8x8xf32>
  %p = pix.pipeline : tensor<8x8xf32> {
    pix.yield %c : tensor<8x8xf32>
  }
  return %t : tensor<8x4xf32>
}
