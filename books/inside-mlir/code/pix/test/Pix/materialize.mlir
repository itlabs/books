// 第 17 章：物化、签名转换、1:N 展开，以及"不装物化会怎样"。
//
// RUN: pix-opt %s --pix-to-arith=lower-kernels=true | FileCheck %s --check-prefix=SRC
// RUN: not pix-opt %s --pix-to-arith='lower-kernels=true no-materialize=true' 2>&1 | FileCheck %s --check-prefix=NOMAT
// RUN: pix-opt %s --pix-to-arith=convert-signatures=true | FileCheck %s --check-prefix=SIG
// RUN: pix-opt %s --pix-to-arith='convert-signatures=true one-to-n=true' | FileCheck %s --check-prefix=ONETON

//===----------------------------------------------------------------------===//
// source 物化：pix.kernel 变成了 tensor 常量，而 pix.convolve 还等 !pix.kernel，
// driver 插一个 pix.to_kernel 把两边接起来。
//===----------------------------------------------------------------------===//

// SRC-LABEL: func.func @kernel_used
// SRC: %[[T:.*]] = arith.constant dense<1.000000e+00> : tensor<3x3xf32>
// SRC-NEXT: %[[K:.*]] = pix.to_kernel %[[T]] :
// SRC-NEXT: pix.convolve %arg0, %[[K]] :
// 注意写成 "pix.kernel dense"：光写 pix.kernel 会误匹配下面那个函数签名里的
// !pix.kernel **类型**。CHECK-NOT 的范围一直延到下一个 LABEL，很容易踩到。
// SRC-NOT: pix.kernel dense

// 不装物化钩子 → driver 插了 unrealized_conversion_cast 却没人解，直接失败。
// 这条断言守的是"报错信息本身"——它是第 17 章教你读的那种。
// NOMAT: error: failed to legalize unresolved materialization from ('tensor<3x3xf32>') to ('!pix.kernel<3 x 3>')
// NOMAT: builtin.unrealized_conversion_cast
func.func @kernel_used(%img: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %k = pix.kernel dense<1.0> : tensor<3x3xf32> -> !pix.kernel<3 x 3>
  %c = pix.convolve %img, %k : !pix.kernel<3 x 3>, tensor<8x8xf32>
  return %c : tensor<8x8xf32>
}

//===----------------------------------------------------------------------===//
// 签名转换：函数参数里的 !pix.kernel 换成 tensor，
// 然后 source 物化自动把块参数接回 !pix.kernel 给 pix.convolve 用。
//===----------------------------------------------------------------------===//

// SIG-LABEL: func.func @takes_kernel
// SIG-SAME: %arg0: tensor<3x3xf32>
// SIG-SAME: %arg1: tensor<8x8xf32>
// SIG: %[[K:.*]] = pix.to_kernel %arg0 :
// SIG-NEXT: pix.convolve %arg1, %[[K]] :

// 1:N：同一个 !pix.kernel 参数展成**两个**（权重张量 + 半径 index），
// 物化再把它们装回一个 kernel。注意 %arg2 —— 原来的 img 被挤到了第三个位置。
// ONETON-LABEL: func.func @takes_kernel
// ONETON-SAME: %arg0: tensor<3x3xf32>
// ONETON-SAME: %arg1: index
// ONETON-SAME: %arg2: tensor<8x8xf32>
// ONETON: %[[K:.*]] = pix.to_kernel %arg0 :
// ONETON-NEXT: pix.convolve %arg2, %[[K]] :
func.func @takes_kernel(%k: !pix.kernel<3 x 3>, %img: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c = pix.convolve %img, %k : !pix.kernel<3 x 3>, tensor<8x8xf32>
  return %c : tensor<8x8xf32>
}
