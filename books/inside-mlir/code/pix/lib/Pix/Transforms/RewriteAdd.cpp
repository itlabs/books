//===- RewriteAdd.cpp - 第 14 章：benefit 与通知机制 ------------*- C++ -*-===//
//
// 两条 pattern 抢同一个 pix.add，靠 benefit 决胜负；外加一个 Listener，
// 把 driver 平时看不见的通知计数后写成属性。
//
//===----------------------------------------------------------------------===//

#include "Pix/Transforms/Passes.h"

#include "Pix/PixDialect.h"
#include "Pix/PixOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace pix {

#define GEN_PASS_DEF_PIXREWRITEADD
#include "Pix/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// 两条抢同一个 op 的 pattern
//===----------------------------------------------------------------------===//

// add(%x, %x) → scale(%x, 2.0)
struct AddSelfToScale : OpRewritePattern<AddOp> {
  // benefit 从构造函数传进来。OpRewritePattern 的第二个参数就是它，
  // 默认值是 1——这也是为什么"不写 benefit"的 pattern 之间顺序不确定。
  AddSelfToScale(MLIRContext *ctx, PatternBenefit benefit)
      : OpRewritePattern<AddOp>(ctx, benefit) {}

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getLhs() != op.getRhs())
      return rewriter.notifyMatchFailure(op, "两个操作数不是同一个值");

    Value two = arith::ConstantOp::create(rewriter, op.getLoc(),
                                          rewriter.getF32FloatAttr(2.0));
    rewriter.replaceOpWithNewOp<ScaleOp>(op, op.getLhs(), two);
    return success();
  }
};

// add(%x, %x) → mul(%x, 全 2 的图)
// 和上面那条**匹配完全相同的输入**，只是结果不同。谁赢由 benefit 决定。
struct AddSelfToMul : OpRewritePattern<AddOp> {
  AddSelfToMul(MLIRContext *ctx, PatternBenefit benefit)
      : OpRewritePattern<AddOp>(ctx, benefit) {}

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getLhs() != op.getRhs())
      return rewriter.notifyMatchFailure(op, "两个操作数不是同一个值");

    auto ty = cast<ShapedType>(op.getType());
    auto splat = DenseFPElementsAttr::get(cast<ShapedType>(ty), {2.0f});
    Value twos = arith::ConstantOp::create(rewriter, op.getLoc(), splat);
    rewriter.replaceOpWithNewOp<MulOp>(op, op.getLhs(), twos);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// 一个把通知计数的 Listener
//
// RewriterBase::Listener 的回调覆盖了 driver 改 IR 的每一种动作。平时没人
// 看它们，但 driver 正是靠这些通知维护 worklist 的——第 11 章那个"改完就
// 重新入队"的行为，底层就是 notifyOperationInserted / notifyOperationModified。
//===----------------------------------------------------------------------===//

struct CountingListener : public RewriterBase::Listener {
  int64_t inserted = 0, replaced = 0, erased = 0, modified = 0;
  int64_t patternsTried = 0, patternsWon = 0;

  void notifyOperationInserted(Operation *, OpBuilder::InsertPoint) override {
    ++inserted;
  }
  void notifyOperationReplaced(Operation *, ValueRange) override { ++replaced; }
  void notifyOperationErased(Operation *) override { ++erased; }
  void notifyOperationModified(Operation *) override { ++modified; }
  void notifyPatternBegin(const Pattern &, Operation *) override {
    ++patternsTried;
  }
  void notifyPatternEnd(const Pattern &, LogicalResult status) override {
    if (succeeded(status))
      ++patternsWon;
  }
};

//===----------------------------------------------------------------------===//
// pix-rewrite-add
//===----------------------------------------------------------------------===//

struct PixRewriteAddPass : impl::PixRewriteAddBase<PixRewriteAddPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp fn = getOperation();
    MLIRContext *ctx = &getContext();

    // swap=false：scale 那条 benefit 2，赢；swap=true：反过来。
    PatternBenefit scaleBenefit = swap ? PatternBenefit(1) : PatternBenefit(2);
    PatternBenefit mulBenefit = swap ? PatternBenefit(2) : PatternBenefit(1);

    RewritePatternSet patterns(ctx);
    patterns.add<AddSelfToScale>(ctx, scaleBenefit);
    patterns.add<AddSelfToMul>(ctx, mulBenefit);

    CountingListener listener;
    GreedyRewriteConfig config;
    config.setListener(&listener);

    // 显式驱动，而不是靠 --canonicalize。这才是"装自己的一套 pattern"
    // 的标准做法：第 11 章那些是挂在 op 上的 canonicalizer，全局生效；
    // 这里这套只在这个 pass 里生效。
    if (failed(applyPatternsGreedily(fn, std::move(patterns), config)))
      return signalPassFailure();

    OpBuilder b(fn);
    fn->setAttr("pix.inserted", b.getI64IntegerAttr(listener.inserted));
    fn->setAttr("pix.replaced", b.getI64IntegerAttr(listener.replaced));
    fn->setAttr("pix.erased", b.getI64IntegerAttr(listener.erased));
    fn->setAttr("pix.modified", b.getI64IntegerAttr(listener.modified));
    fn->setAttr("pix.patterns_tried",
                b.getI64IntegerAttr(listener.patternsTried));
    fn->setAttr("pix.patterns_won", b.getI64IntegerAttr(listener.patternsWon));
  }
};

} // namespace
} // namespace pix
} // namespace mlir
