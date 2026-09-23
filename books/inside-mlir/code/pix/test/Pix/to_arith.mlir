// 第 16 章：Dialect Conversion 的四个部件。
//
// RUN: pix-opt %s --pix-to-arith | FileCheck %s --check-prefix=PARTIAL
// 完全转换会失败，因为 pix.reduce 没有下降路径。用 not 断言"它必须失败"，
// 并检查报错里点名了那个 op —— 报错信息本身也是行为的一部分。
// RUN: not pix-opt %s --pix-to-arith=full=true 2>&1 | FileCheck %s --check-prefix=FULLFAIL

//===----------------------------------------------------------------------===//
// 三个有下降路径的 op 全部换掉
//===----------------------------------------------------------------------===//

// PARTIAL-LABEL: func.func @lowerable
// PARTIAL: arith.addf %arg0, %arg1
// PARTIAL: arith.mulf
// PARTIAL: tensor.splat %arg2
// PARTIAL: arith.mulf
// PARTIAL-NOT: pix.
func.func @lowerable(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %s: f32) -> tensor<4x4xf32> {
  %0 = pix.add %a, %b : tensor<4x4xf32>
  %1 = pix.mul %0, %b : tensor<4x4xf32>
  %2 = pix.scale %1, %s : tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}

//===----------------------------------------------------------------------===//
// 部分转换的要点：没有下降路径的 op **原样留下**，而且这不算失败。
// pix.reduce 处于 Unknown 状态（我们没标它合法、也没标非法），
// partial 模式把 Unknown 当合法，于是 driver 不管它。
//===----------------------------------------------------------------------===//

// PARTIAL-LABEL: func.func @has_reduce
// PARTIAL: arith.addf %arg0, %arg0
// PARTIAL: pix.reduce
func.func @has_reduce(%a: tensor<4x4xf32>) -> f32 {
  %0 = pix.add %a, %a : tensor<4x4xf32>
  %1 = pix.reduce %0 : tensor<4x4xf32> -> f32
  return %1 : f32
}

//===----------------------------------------------------------------------===//
// 完全转换：任何漏网的 op 都算失败。报错挂在**最外层**的 builtin.module 上，
// 而不是那个真正有问题的 op —— 这个"报错离现场很远"的毛病是第 17 章的主题。
//===----------------------------------------------------------------------===//

// FULLFAIL: error: failed to legalize operation 'builtin.module'
