// RUN: pix-opt %s | pix-opt | FileCheck %s
//
// 第 9 章：自定义类型 !pix.kernel<R x C> 与枚举属性 #pix.border。
// 两趟 pix-opt 是为了确认"打印出来的东西还能再解析回去"——自定义类型/属性最容易在这里出问题。

// CHECK-LABEL: func.func @laplace
func.func @laplace(%img: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK: pix.kernel dense<{{.*}}> : tensor<3x3xf32> -> <3 x 3>
  %k = pix.kernel dense<[[0.0, 1.0, 0.0],
                         [1.0, -4.0, 1.0],
                         [0.0, 1.0, 0.0]]> : tensor<3x3xf32> -> !pix.kernel<3 x 3>
  // 非默认的 border 会被打印出来
  // CHECK: pix.convolve %{{.*}}, %{{.*}} border <zero> : <3 x 3>, tensor<8x8xf32>
  %out = pix.convolve %img, %k border<zero> : !pix.kernel<3 x 3>, tensor<8x8xf32>
  // 默认值（clamp）不打印——DefaultValuedAttr + 可选组的效果
  // CHECK: pix.convolve %{{.*}}, %{{.*}} : <3 x 3>, tensor<8x8xf32>
  %dfl = pix.convolve %out, %k : !pix.kernel<3 x 3>, tensor<8x8xf32>
  return %dfl : tensor<8x8xf32>
}

// 在函数签名这种"方言无法从上下文推断"的位置，类型打印完整形式
// CHECK-LABEL: func.func @kernel_in_signature
// CHECK-SAME: !pix.kernel<1 x 5>
func.func @kernel_in_signature(%k: !pix.kernel<1 x 5>) -> !pix.kernel<1 x 5> {
  return %k : !pix.kernel<1 x 5>
}
