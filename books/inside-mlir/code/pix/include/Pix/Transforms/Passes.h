//===- Passes.h - pix 的 pass 入口 -----------------------------*- C++ -*-===//
//
// 《深入 MLIR》第 13 章。两个 GEN_PASS_* 宏各切出生成代码的一段
// （和第 7 章 GET_OP_CLASSES 是同一套把戏）：
//   GEN_PASS_DECL         —— createXxxPass() 的声明与 pass 基类
//   GEN_PASS_REGISTRATION —— registerPixPasses()，pix-opt 里调一次即可
//
//===----------------------------------------------------------------------===//

#ifndef PIX_TRANSFORMS_PASSES_H
#define PIX_TRANSFORMS_PASSES_H

// 第 16 章：Passes.td 里写了 dependentDialects，生成代码会引用这些方言类，
// 所以要在 include 生成代码之前把它们的头带进来（和第 11 章方言那次同理）。
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace pix {

#define GEN_PASS_DECL
#include "Pix/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "Pix/Transforms/Passes.h.inc"

} // namespace pix
} // namespace mlir

#endif // PIX_TRANSFORMS_PASSES_H
