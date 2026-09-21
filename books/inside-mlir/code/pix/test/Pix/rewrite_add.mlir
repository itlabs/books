// 第 14 章：benefit 决定哪条 pattern 赢；Listener 把 driver 的通知计数出来。
//
// RUN: pix-opt %s --pix-rewrite-add            | FileCheck %s --check-prefix=SCALE
// RUN: pix-opt %s --pix-rewrite-add=swap=true  | FileCheck %s --check-prefix=MUL
//
// 注意计数属性挂在 func.func **那一行**上，所以用 CHECK-SAME 而不是 CHECK——
// 一旦用了 CHECK，游标已经走到函数体里，再往后就找不到它们了。第 12 章讲过
// CHECK-SAME 的用途，这里是它的典型场合。
//
// 另外靠 CHECK-LABEL 把三个函数切成互不干扰的段，下面那些 CHECK-NOT 才不会
// 跑到别的函数里去乱匹配（@untouched 里就有一个合法的 pix.mul）。

//===----------------------------------------------------------------------===//
// 两条 pattern 匹配的输入完全相同，只是结果不同。benefit 高的赢。
//
// 而且 tried = 1：高 benefit 那条一次就成了，**低 benefit 那条根本没被试**。
// 这条断言守住的正是"成功即短路"这个语义。
//===----------------------------------------------------------------------===//

// SCALE-LABEL: func.func @self_add
// SCALE-SAME: pix.patterns_tried = 1 : i64
// SCALE-SAME: pix.patterns_won = 1 : i64
// SCALE: pix.scale %arg0
// SCALE-NOT: pix.mul
// SCALE-NOT: pix.add

// swap=true：两条对调 → 变成 pix.mul，计数不变
// MUL-LABEL: func.func @self_add
// MUL-SAME: pix.patterns_tried = 1 : i64
// MUL-SAME: pix.patterns_won = 1 : i64
// MUL: pix.mul %arg0
// MUL-NOT: pix.scale
func.func @self_add(%img: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %r = pix.add %img, %img : tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}

//===----------------------------------------------------------------------===//
// 两个操作数不同 → 两条都试了、都失败 → tried 2、won 0。
// 和上面那条对照，才说明 tried=1 不是因为"只注册了一条 pattern"。
//===----------------------------------------------------------------------===//

// swap 不影响这个函数：两条都匹配不上
// MUL-LABEL: func.func @two_operands
// MUL: pix.add %arg0, %arg1
//
// SCALE-LABEL: func.func @two_operands
// SCALE-SAME: pix.patterns_tried = 2 : i64
// SCALE-SAME: pix.patterns_won = 0 : i64
// SCALE: pix.add %arg0, %arg1
func.func @two_operands(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %r = pix.add %a, %b : tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}

//===----------------------------------------------------------------------===//
// 一个 pix.add 都没有 → **一条 pattern 都不会被试**（tried = 0）。
// pattern 是按"根 op 名"索引的，driver 只会拿 pix.add 的 pattern 去试 pix.add；
// 别的 op 连问都不问，所以注册一大堆 pattern 不会拖慢无关的 op。
//===----------------------------------------------------------------------===//

// SCALE-LABEL: func.func @untouched
// SCALE-SAME: pix.patterns_tried = 0 : i64
// SCALE-SAME: pix.patterns_won = 0 : i64
// SCALE: pix.mul %arg0, %arg0
// SCALE-NOT: pix.scale
func.func @untouched(%img: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %r = pix.mul %img, %img : tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}
