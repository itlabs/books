// RUN: pix-opt %s --pix-hoist-kernels | FileCheck %s
// 跑两遍结果必须一样（幂等）。第二遍里 pass 一处都不改，会走
// markAllAnalysesPreserved() 那条分支。
// RUN: pix-opt %s --pix-hoist-kernels --pix-hoist-kernels | FileCheck %s

// 两个 pix.kernel 散在中间，应该被提到函数最前面，**而且保持彼此的原有顺序**
// （1.0 在 2.0 前面）。倒序插入就是为了这个。
// CHECK-LABEL: func.func @scattered
// CHECK-NEXT: %[[K1:.*]] = pix.kernel dense<1.000000e+00>
// CHECK-NEXT: %[[K2:.*]] = pix.kernel dense<2.000000e+00>
// CHECK-NEXT: %[[A:.*]] = pix.add
// CHECK-NEXT: %[[C1:.*]] = pix.convolve %[[A]], %[[K1]] :
// CHECK-NEXT: pix.convolve %[[C1]], %[[K2]] :
func.func @scattered(%img: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %a = pix.add %img, %img : tensor<8x8xf32>
  %k1 = pix.kernel dense<1.0> : tensor<3x3xf32> -> !pix.kernel<3 x 3>
  %c1 = pix.convolve %a, %k1 : !pix.kernel<3 x 3>, tensor<8x8xf32>
  %k2 = pix.kernel dense<2.0> : tensor<3x3xf32> -> !pix.kernel<3 x 3>
  %c2 = pix.convolve %c1, %k2 : !pix.kernel<3 x 3>, tensor<8x8xf32>
  return %c2 : tensor<8x8xf32>
}

// 已经在开头的不该被动。这条守的是 pass 的"没改动"判断——
// 如果它把这里也算成改动，第二次运行就会声称改了 IR，
// 上面那条跑两遍的 RUN 行仍然会过，但 markAllAnalysesPreserved 就白写了。
// CHECK-LABEL: func.func @already_first
// CHECK-NEXT: %[[K:.*]] = pix.kernel dense<3.000000e+00>
// CHECK-NEXT: pix.convolve %arg0, %[[K]] :
func.func @already_first(%img: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %k = pix.kernel dense<3.0> : tensor<3x3xf32> -> !pix.kernel<3 x 3>
  %c = pix.convolve %img, %k : !pix.kernel<3 x 3>, tensor<8x8xf32>
  return %c : tensor<8x8xf32>
}

// 一个 kernel 都没有：pass 应当干干净净地什么都不做。
// CHECK-LABEL: func.func @no_kernel
// CHECK-NEXT: pix.add
func.func @no_kernel(%img: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %a = pix.add %img, %img : tensor<8x8xf32>
  return %a : tensor<8x8xf32>
}
