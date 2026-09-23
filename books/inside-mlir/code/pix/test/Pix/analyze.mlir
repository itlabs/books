// RUN: pix-opt %s --pix-analyze | FileCheck %s
// 副作用的实际后果，用通用 pass 对照：
// RUN: pix-opt %s --canonicalize --cse | FileCheck %s --check-prefix=EFFECT
//
// 第 19 章：MLIR 自带的三样分析。结果写成属性，好断言。

//===----------------------------------------------------------------------===//
// 数据流分析的价值：它推出了**单看一个 op 看不出来**的结论。
// %m = pix.mul %img, %t 里 %img 完全未知，但 %t 已知全零，
// 0 × 任何 = 0，所以 %m 也是零。known_zero 数到 3 个（常量、transpose、mul）。
//===----------------------------------------------------------------------===//

// 两个坑，都在这里踩过：
// ① CHECK-SAME 必须按**字典序**排——属性字典是排序后打印的，而 CHECK-SAME
//    要求同一行内按给出的顺序依次匹配。顺序写错会报"expected string not
//    found in input"，而实际上那个属性就在那儿。
// ② 散文注释里**绝对不要写出带冒号的指令字面量**。FileCheck 在整个文件里扫
//    前缀，不管它在不在散文里，于是会凭空多出一条指令。写这个注释时我连着
//    踩了两次：先是写了带冒号的 CHECK-SAME，改完又在引用报错原文时带出了一个
//    带冒号的 CHECK。要提到它们就去掉冒号，或者干脆别写全。
// CHECK-LABEL: func.func @zeros
// transpose 支配 mul，反之不然 → 只有 1 个支配全体
// CHECK-SAME: pix.dominators = 1 : i64
// CHECK-SAME: pix.known_zero = 3 : i64
// 两个 pix op（transpose、mul）都是 Pure
// CHECK-SAME: pix.pure = 2 : i64
func.func @zeros(%img: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %z = arith.constant dense<0.0> : tensor<4x4xf32>
  %t = pix.transpose %z : tensor<4x4xf32> to tensor<4x4xf32>
  %m = pix.mul %img, %t : tensor<4x4xf32>
  return %m : tensor<4x4xf32>
}

//===----------------------------------------------------------------------===//
// 副作用：pix.trace 声明了 MemWrite，于是三种查询它一个都不满足。
// 函数里有 2 个 pix op（add、trace），但 pure / effect_free / trivially_dead
// 都只数到 1 —— 那个 1 就是 add。
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @effects
// CHECK-SAME: pix.effect_free = 1 : i64
// CHECK-SAME: pix.pure = 1 : i64
// CHECK-SAME: pix.trivially_dead = 1 : i64
func.func @effects(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %x = pix.add %a, %a : tensor<4x4xf32>
  pix.trace %x : tensor<4x4xf32>
  return %x : tensor<4x4xf32>
}

//===----------------------------------------------------------------------===//
// 同一份输入换通用 pass 跑：Pure 的被合并/删除，有副作用的一个不动。
// 这组断言证明"副作用建模"不是文档里的说法，而是真的在管事。
//===----------------------------------------------------------------------===//

// 两个一样的 add 合并成一个，没人用的 mul 被删
// EFFECT-LABEL: func.func @pure_ops
// EFFECT: pix.add %arg0, %arg1
// EFFECT-NEXT: return
// EFFECT-NOT: pix.mul
func.func @pure_ops(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %x = pix.add %a, %b : tensor<4x4xf32>
  %y = pix.add %a, %b : tensor<4x4xf32>
  %dead = pix.mul %a, %b : tensor<4x4xf32>
  return %x : tensor<4x4xf32>
}

// 两个一样的 trace 一个都不能合并——次数和顺序都是语义
// EFFECT-LABEL: func.func @two_traces
// EFFECT: pix.trace
// EFFECT-NEXT: pix.trace
func.func @two_traces(%a: tensor<4x4xf32>) {
  pix.trace %a : tensor<4x4xf32>
  pix.trace %a : tensor<4x4xf32>
  return
}
