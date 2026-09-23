// RUN: pix-opt %s --canonicalize | FileCheck %s
//
// 第 11 章：fold 与 canonicalize 各管一段。
//   fold          —— 就地给出答案（返回已有的值或常量），不许造 op
//   canonicalizer —— 需要造新 op 的改写
// --canonicalize 这一个 pass 会把两者、以及别的方言的 folder 一起跑到不动点。

//===----------------------------------------------------------------------===//
// fold 的第一种返回：一个**已经存在的值**
//===----------------------------------------------------------------------===//

// 加一张全零图等于什么都没加。
// CHECK-LABEL: func.func @add_zero
// CHECK-NEXT: return %arg0
// CHECK-NOT: pix.add
func.func @add_zero(%img: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %zero = arith.constant dense<0.0> : tensor<4x4xf32>
  %r = pix.add %img, %zero : tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}

// 零图在左边也要认（Commutative 不负责把常量挪到右边）。
// CHECK-LABEL: func.func @zero_add
// CHECK-NEXT: return %arg0
// CHECK-NOT: pix.add
func.func @zero_add(%img: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %zero = arith.constant dense<0.0> : tensor<4x4xf32>
  %r = pix.add %zero, %img : tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}

// 不是全零就不该动。
// CHECK-LABEL: func.func @add_nonzero_stays
// CHECK: pix.add
func.func @add_nonzero_stays(%img: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %c = arith.constant dense<1.0> : tensor<4x4xf32>
  %r = pix.add %img, %c : tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}

//===----------------------------------------------------------------------===//
// fold 的第二种返回：一个**算出来的 Attribute**（真正的常量折叠）
//   能落地成 arith.constant，靠的是方言实现了 materializeConstant；
//   不实现那个钩子，这里折出来的常量没人放得回 IR，测试会失败。
//===----------------------------------------------------------------------===//

// splat：2.5 × 16 个元素 = 40
// CHECK-LABEL: func.func @reduce_splat
// CHECK: %[[C:.*]] = arith.constant 4.000000e+01 : f32
// CHECK-NEXT: return %[[C]] :
// CHECK-NOT: pix.reduce
func.func @reduce_splat() -> f32 {
  %img = arith.constant dense<2.5> : tensor<4x4xf32>
  %s = pix.reduce %img : tensor<4x4xf32> -> f32
  return %s : f32
}

// 逐元素：1 + 2 + 3 + 4 = 10
// CHECK-LABEL: func.func @reduce_dense
// CHECK: %[[C:.*]] = arith.constant 1.000000e+01 : f32
// CHECK-NEXT: return %[[C]] :
// CHECK-NOT: pix.reduce
func.func @reduce_dense() -> f32 {
  %img = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %s = pix.reduce %img : tensor<2x2xf32> -> f32
  return %s : f32
}

// 图不是常量，折不动。
// CHECK-LABEL: func.func @reduce_dynamic_stays
// CHECK: pix.reduce
func.func @reduce_dynamic_stays(%img: tensor<4x4xf32>) -> f32 {
  %s = pix.reduce %img : tensor<4x4xf32> -> f32
  return %s : f32
}

//===----------------------------------------------------------------------===//
// fold：系数为 1 时，整个 op 等于它的输入
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @scale_by_one
// CHECK-NEXT: return %arg0
// CHECK-NOT: pix.scale
func.func @scale_by_one(%img: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %one = arith.constant 1.0 : f32
  %r = pix.scale %img, %one : tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}

// 系数不是 1 就不该动。这条守的是"别把化简写得太贪"。
// CHECK-LABEL: func.func @scale_by_two_stays
// CHECK: pix.scale
func.func @scale_by_two_stays(%img: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %two = arith.constant 2.0 : f32
  %r = pix.scale %img, %two : tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}

// 系数是个运行时值，folder 拿不到常量，同样不该动。
// CHECK-LABEL: func.func @scale_by_unknown_stays
// CHECK: pix.scale
func.func @scale_by_unknown_stays(%img: tensor<4x4xf32>, %f: f32) -> tensor<4x4xf32> {
  %r = pix.scale %img, %f : tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}

//===----------------------------------------------------------------------===//
// fold：双重转置抵消
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @double_transpose
// CHECK-NEXT: return %arg0
// CHECK-NOT: pix.transpose
func.func @double_transpose(%img: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %t1 = pix.transpose %img : tensor<4x8xf32> to tensor<8x4xf32>
  %t2 = pix.transpose %t1 : tensor<8x4xf32> to tensor<4x8xf32>
  return %t2 : tensor<4x8xf32>
}

// 单层转置当然留着。
// CHECK-LABEL: func.func @single_transpose_stays
// CHECK: pix.transpose
func.func @single_transpose_stays(%img: tensor<4x8xf32>) -> tensor<8x4xf32> {
  %t = pix.transpose %img : tensor<4x8xf32> to tensor<8x4xf32>
  return %t : tensor<8x4xf32>
}

//===----------------------------------------------------------------------===//
// canonicalize：嵌套 scale 合成一层
//   注意 CHECK 里等的是 6.0，不是 arith.mulf —— 我们的 pattern 造的是 mulf，
//   但 arith 自己的 folder 在同一轮里把 2.0 * 3.0 折成了常量。
//   两个方言的化简自动接力，谁都不认识谁。
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @nested_scale
// CHECK: %[[C:.*]] = arith.constant 6.000000e+00 : f32
// CHECK: pix.scale %arg0, %[[C]] :
// CHECK-NOT: pix.scale
// CHECK-NOT: arith.mulf
func.func @nested_scale(%img: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %a = arith.constant 2.0 : f32
  %b = arith.constant 3.0 : f32
  %s1 = pix.scale %img, %a : tensor<4x4xf32>
  %s2 = pix.scale %s1, %b : tensor<4x4xf32>
  return %s2 : tensor<4x4xf32>
}

// 系数不是常量时，合并照样发生，只是留下一个真的 arith.mulf。
// 这说明 pattern 本身不依赖常量——能折是 arith 的功劳。
// CHECK-LABEL: func.func @nested_scale_dynamic
// CHECK: %[[M:.*]] = arith.mulf %arg1, %arg2
// CHECK: pix.scale %arg0, %[[M]] :
// CHECK-NOT: pix.scale
func.func @nested_scale_dynamic(%img: tensor<4x4xf32>, %a: f32, %b: f32) -> tensor<4x4xf32> {
  %s1 = pix.scale %img, %a : tensor<4x4xf32>
  %s2 = pix.scale %s1, %b : tensor<4x4xf32>
  return %s2 : tensor<4x4xf32>
}

//===----------------------------------------------------------------------===//
// canonicalize：add %x, %x → scale %x, 2.0
//   输入里一个 arith op 都没有。能跑得通，靠的是方言声明了
//   dependentDialects = ["::mlir::arith::ArithDialect"]——漏掉它这里会直接 abort。
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @add_self
// CHECK: %[[C:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: pix.scale %arg0, %[[C]] :
// CHECK-NOT: pix.add
func.func @add_self(%img: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %r = pix.add %img, %img : tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}

// 两个操作数不同就不是这条规则的事。
// CHECK-LABEL: func.func @add_two_stays
// CHECK: pix.add
func.func @add_two_stays(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %r = pix.add %a, %b : tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}

//===----------------------------------------------------------------------===//
// 接力：几条规则叠在一起，一遍 --canonicalize 全部收敛
//   add %x, %x            → scale %x, 2.0
//   再套一层 scale %_, 3.0 → scale %x, 6.0
//   最外面套一层 scale 1.0 → 被 fold 掉
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @relay
// CHECK: %[[C:.*]] = arith.constant 6.000000e+00 : f32
// CHECK: pix.scale %arg0, %[[C]] :
// CHECK-NOT: pix.add
func.func @relay(%img: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %three = arith.constant 3.0 : f32
  %one = arith.constant 1.0 : f32
  %sum = pix.add %img, %img : tensor<4x4xf32>
  %s1 = pix.scale %sum, %three : tensor<4x4xf32>
  %s2 = pix.scale %s1, %one : tensor<4x4xf32>
  return %s2 : tensor<4x4xf32>
}
