//===- CostPasses.cpp - 第 13 章的三个 pass ---------------------*- C++ -*-===//
//
// 这三个 pass 各演示 Pass 框架的一个机制。共同的骨架来自 TableGen：
// `GEN_PASS_DEF_<大写 pass 名>` 会切出一个 `impl::<PassName>Base<DerivedT>`
// 模板，帮你实现 getArgument()/getDescription()/getName()/clonePass() 这些
// 样板，还顺带以 friend 的形式定义好 createXxx()——所以**不要自己写
// create 函数**，也别把 pass 结构放到别的命名空间里去，那个 friend 找不到它。
//
//===----------------------------------------------------------------------===//

#include "Pix/Transforms/Passes.h"

#include "Pix/PixDialect.h"
#include "Pix/PixOps.h"
#include "Pix/Transforms/CostAnalysis.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace pix {

// 三个 GEN_PASS_DEF_ 宏各切出一段生成代码。宏名是 pass 名的全大写形式。
#define GEN_PASS_DEF_PIXCOSTREPORT
#define GEN_PASS_DEF_PIXCOSTRECHECK
#define GEN_PASS_DEF_PIXHOISTKERNELS
#include "Pix/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// pix-cost-report —— 函数级 pass + 用分析 + 精确声明保留
//===----------------------------------------------------------------------===//

struct PixCostReportPass : impl::PixCostReportBase<PixCostReportPass> {
  void runOnOperation() override {
    func::FuncOp fn = getOperation(); // 类型已经收窄，不用 cast

    // 第一次要这份分析 → AnalysisManager 构造它并缓存；之后同一作用域内
    // 再要就是同一份。注意作用域：函数级 pass 拿到的分析只覆盖本函数。
    const PixCostAnalysis &cost = getAnalysis<PixCostAnalysis>();

    OpBuilder b(fn);
    fn->setAttr("pix.num_ops", b.getI64IntegerAttr(cost.getNumOps()));
    fn->setAttr("pix.elements", b.getI64IntegerAttr(cost.getElements()));

    // 关键一行。我们只挂了两个可丢弃属性，op 的构成一点没变，所以那份
    // 统计仍然成立——这件事只有我们知道。不说的话，框架会保守地把**所有**
    // 分析作废，下一个 pass 得重新遍历一遍。
    markAnalysesPreserved<PixCostAnalysis>();
  }
};

//===----------------------------------------------------------------------===//
// pix-cost-recheck —— 让"缓存命中"这件看不见的事变得可观察
//===----------------------------------------------------------------------===//

struct PixCostRecheckPass : impl::PixCostRecheckBase<PixCostRecheckPass> {
  void runOnOperation() override {
    func::FuncOp fn = getOperation();

    // getCachedAnalysis 和 getAnalysis 的区别就在这儿：它**不会**在缓存
    // 未命中时去构造，而是返回一个空的 optional。所以它能用来观察缓存状态。
    auto cached = getCachedAnalysis<PixCostAnalysis>();

    OpBuilder b(fn);
    fn->setAttr("pix.cost_cached", b.getBoolAttr(cached.has_value()));
    markAnalysesPreserved<PixCostAnalysis>();
  }
};

//===----------------------------------------------------------------------===//
// pix-hoist-kernels —— 这一章唯一真的改 IR 的 pass
//===----------------------------------------------------------------------===//

struct PixHoistKernelsPass : impl::PixHoistKernelsBase<PixHoistKernelsPass> {
  void runOnOperation() override {
    func::FuncOp fn = getOperation();
    if (fn.getBody().empty())
      return;
    Block &entry = fn.getBody().front();

    // 先收集再移动：边遍历边改 IR 是常见的崩溃来源。
    SmallVector<KernelOp> kernels;
    fn.walk([&](KernelOp k) { kernels.push_back(k); });

    // 已经在开头的不算改动。倒序插到最前面，好保持它们之间的原有顺序。
    bool changed = false;
    for (KernelOp k : llvm::reverse(kernels)) {
      if (k->getBlock() == &entry && k->getPrevNode() == nullptr)
        continue;
      k->moveBefore(&entry, entry.begin());
      changed = true;
    }

    // 什么都没做就要说一声。漏了这一行，一个空跑的 pass 也会把后面所有
    // 分析的缓存冲掉——这是很容易忽略的性能问题，不会有任何报错提示你。
    if (!changed)
      markAllAnalysesPreserved();
  }
};

} // namespace
} // namespace pix
} // namespace mlir
