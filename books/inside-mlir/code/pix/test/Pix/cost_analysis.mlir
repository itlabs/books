// 第 13 章：分析的缓存与失效。三条 RUN 行是三组对照实验，靠 --check-prefix
// 让它们在同一个文件里共存——这是 lit 里很常用的一招。
//
// RUN: pix-opt %s --pix-cost-report                        | FileCheck %s --check-prefix=REPORT
// RUN: pix-opt %s --pix-cost-recheck                       | FileCheck %s --check-prefix=NEVER
// RUN: pix-opt %s --pix-cost-report --pix-cost-recheck      | FileCheck %s --check-prefix=CACHED
// RUN: pix-opt %s --pix-cost-report --canonicalize --pix-cost-recheck | FileCheck %s --check-prefix=INVALID

//===----------------------------------------------------------------------===//
// 分析算出来的数对不对：2 个 pix op，结果张量各 4x4 → 32 个元素
//===----------------------------------------------------------------------===//

// REPORT-LABEL: func.func @two_ops
// REPORT-SAME: pix.elements = 32 : i64
// REPORT-SAME: pix.num_ops = 2 : i64
func.func @two_ops(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %s = pix.add %a, %b : tensor<4x4xf32>
  %t = pix.mul %s, %b : tensor<4x4xf32>
  return %t : tensor<4x4xf32>
}

// 一个 pix op 都没有的函数也要算，得到 0 —— 分析的作用域是**本函数**，
// 不会把上面那个函数的 2 个 op 算进来。
// REPORT-LABEL: func.func @no_pix
// REPORT-SAME: pix.elements = 0 : i64
// REPORT-SAME: pix.num_ops = 0 : i64
func.func @no_pix(%a: tensor<8x8xf32>) -> tensor<8x8xf32> {
  return %a : tensor<8x8xf32>
}

//===----------------------------------------------------------------------===//
// ① 只跑 recheck：分析从没算过 → 缓存里当然没有
//===----------------------------------------------------------------------===//
// NEVER: pix.cost_cached = false
// NEVER: pix.cost_cached = false

//===----------------------------------------------------------------------===//
// ② report 之后紧跟 recheck：report 里调了
//    markAnalysesPreserved<PixCostAnalysis>()，所以缓存还在
//===----------------------------------------------------------------------===//
// CACHED: pix.cost_cached = true
// CACHED: pix.cost_cached = true

//===----------------------------------------------------------------------===//
// ③ 中间插一个 canonicalize：它改了 IR、也没声明保留我们这份分析，
//    于是缓存被作废，recheck 看到的是 false
//===----------------------------------------------------------------------===//
// INVALID: pix.cost_cached = false
// INVALID: pix.cost_cached = false
