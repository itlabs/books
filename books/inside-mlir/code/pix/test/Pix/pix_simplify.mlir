// RUN: pix-opt %s --pix-simplify | FileCheck %s
//
// 第 12 章：--pix-simplify 是我们注册的一条**命名管线**
// （canonicalize + cse + symbol-dce），不是新写的 pass。
//
// 这个测试守的是"这个名字仍然做该做的事"。以后管线里加了 pix 自己的 pass
// （第 13 章），只要效果不退步，这个文件不用改——命名管线的意义就在这儿：
// 它是给使用者的接口，内容可以换。

// 三件事一次做完：
//   %a  scale 系数 1  → fold 掉（canonicalize）
//   %b %c 两个相同的 pix.add → 合并成一个（cse，靠 Pure）
//   %b %c %d 都没人用 → 删掉（DCE）
// 最后整个函数只剩一句 return。

// CHECK-LABEL: func.func @everything_collapses
// CHECK-NEXT: return %arg0
// CHECK-NOT: pix.
// CHECK-NOT: arith.
func.func @everything_collapses(%img: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %one = arith.constant 1.0 : f32
  %a = pix.scale %img, %one : tensor<4x4xf32>
  %b = pix.add %img, %img : tensor<4x4xf32>
  %c = pix.add %img, %img : tensor<4x4xf32>
  %d = pix.add %b, %c : tensor<4x4xf32>
  return %a : tensor<4x4xf32>
}

// 有人用的结果不能被删。这条是上一条的对照：证明上面那些消失是因为
// "没人用"，不是因为 --pix-simplify 手黑。
// CHECK-LABEL: func.func @used_result_survives
// CHECK: %[[C:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[S:.*]] = pix.scale %arg0, %[[C]] :
// CHECK: return %[[S]] :
func.func @used_result_survives(%img: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %two = arith.constant 2.0 : f32
  %s = pix.scale %img, %two : tensor<4x4xf32>
  return %s : tensor<4x4xf32>
}

// CSE 单独看一眼：两个相同的 pix.add，结果都被用到，必须合并成一个。
// CHECK-LABEL: func.func @cse_merges
// CHECK: %[[S:.*]] = pix.add %arg0, %arg1
// CHECK-NEXT: pix.add %[[S]], %[[S]] :
// CHECK-NOT: pix.add
func.func @cse_merges(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %x = pix.add %a, %b : tensor<4x4xf32>
  %y = pix.add %a, %b : tensor<4x4xf32>
  %z = pix.add %x, %y : tensor<4x4xf32>
  return %z : tensor<4x4xf32>
}
