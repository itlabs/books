//===- DRRPass.cpp - 第 15 章：跑 DRR 生成的 pattern -----------*- C++ -*-===//
//
// 这个文件是"声明式"的证据：一条 pattern 都没在这里写，全在 PixPatterns.td。
//
//===----------------------------------------------------------------------===//

#include "Pix/Transforms/Passes.h"

#include "Pix/PixDialect.h"
#include "Pix/PixOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace pix {

#define GEN_PASS_DEF_PIXDRR
#include "Pix/Transforms/Passes.h.inc"

namespace {

// 生成的 pattern 类都藏在这个 .inc 里，还顺带定义了 populateWithGenerated()。
// 放在匿名命名空间里，免得它们泄漏成公开符号。
#include "PixPatterns.inc"

struct PixDRRPass : impl::PixDRRBase<PixDRRPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateWithGenerated(patterns);   // ← 一句话装完所有 Pat<>
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace pix
} // namespace mlir
