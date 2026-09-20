// RUN: pix-opt %s --canonicalize | FileCheck %s
//
// 第 12 章：pix.reduce 的归约方式变成枚举属性（sum/max/min），
// 兑现第 8 章留下的那句"以后用枚举扩展"。
//
// 这个文件同时守住两件事：
//   1. 三种 kind 的常量折叠算得对
//   2. **老写法仍然能解析**——kind 有默认值，所以不带属性的 pix.reduce
//      和加属性之前一模一样。给已有 op 加属性时这是必须守住的兼容性。

// 同一张图 [[1, 5], [3, 2]]：sum = 11，max = 5，min = 1。
// 三个答案互不相同，所以这组测试真的能区分三种算法——
// 如果 fold 里 switch 写漏一个分支，这里立刻红。

//===----------------------------------------------------------------------===//
// 老写法：不写 kind，默认 sum
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @default_is_sum
// CHECK: %[[C:.*]] = arith.constant 1.100000e+01 : f32
// CHECK-NEXT: return %[[C]] :
// CHECK-NOT: pix.reduce
func.func @default_is_sum() -> f32 {
  %img = arith.constant dense<[[1.0, 5.0], [3.0, 2.0]]> : tensor<2x2xf32>
  %s = pix.reduce %img : tensor<2x2xf32> -> f32
  return %s : f32
}

// 显式写 sum，结果必须和默认一致。
// CHECK-LABEL: func.func @explicit_sum
// CHECK: arith.constant 1.100000e+01 : f32
func.func @explicit_sum() -> f32 {
  %img = arith.constant dense<[[1.0, 5.0], [3.0, 2.0]]> : tensor<2x2xf32>
  %s = pix.reduce %img {kind = #pix.reduce_kind<sum>} : tensor<2x2xf32> -> f32
  return %s : f32
}

//===----------------------------------------------------------------------===//
// max / min
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @as_max
// CHECK: arith.constant 5.000000e+00 : f32
// CHECK-NOT: pix.reduce
func.func @as_max() -> f32 {
  %img = arith.constant dense<[[1.0, 5.0], [3.0, 2.0]]> : tensor<2x2xf32>
  %s = pix.reduce %img {kind = #pix.reduce_kind<max>} : tensor<2x2xf32> -> f32
  return %s : f32
}

// CHECK-LABEL: func.func @as_min
// CHECK: arith.constant 1.000000e+00 : f32
// CHECK-NOT: pix.reduce
func.func @as_min() -> f32 {
  %img = arith.constant dense<[[1.0, 5.0], [3.0, 2.0]]> : tensor<2x2xf32>
  %s = pix.reduce %img {kind = #pix.reduce_kind<min>} : tensor<2x2xf32> -> f32
  return %s : f32
}

//===----------------------------------------------------------------------===//
// splat 的捷径对三种 kind 都成立，但理由不同：
// sum 是"值 × 元素个数"，max/min 是"就是那个值"。
//===----------------------------------------------------------------------===//

// 2.5 × 16 = 40
// CHECK-LABEL: func.func @splat_sum
// CHECK: arith.constant 4.000000e+01 : f32
func.func @splat_sum() -> f32 {
  %img = arith.constant dense<2.5> : tensor<4x4xf32>
  %s = pix.reduce %img : tensor<4x4xf32> -> f32
  return %s : f32
}

// max 不乘元素个数
// CHECK-LABEL: func.func @splat_max
// CHECK: arith.constant 2.500000e+00 : f32
func.func @splat_max() -> f32 {
  %img = arith.constant dense<2.5> : tensor<4x4xf32>
  %s = pix.reduce %img {kind = #pix.reduce_kind<max>} : tensor<4x4xf32> -> f32
  return %s : f32
}

//===----------------------------------------------------------------------===//
// 折不动的情形
//===----------------------------------------------------------------------===//

// 图不是常量。
// CHECK-LABEL: func.func @dynamic_stays
// CHECK: pix.reduce
func.func @dynamic_stays(%img: tensor<4x4xf32>) -> f32 {
  %s = pix.reduce %img {kind = #pix.reduce_kind<max>} : tensor<4x4xf32> -> f32
  return %s : f32
}

// 空图：max/min 没有定义（该返回什么？），所以 fold 直接放弃。
// 这条守的是"别为了少写一个 if 就给出一个编出来的答案"。
// CHECK-LABEL: func.func @empty_stays
// CHECK: pix.reduce
func.func @empty_stays() -> f32 {
  %img = arith.constant dense<[]> : tensor<0xf32>
  %s = pix.reduce %img {kind = #pix.reduce_kind<min>} : tensor<0xf32> -> f32
  return %s : f32
}
