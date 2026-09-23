//===- AnalyzePass.cpp - 第 19 章：三种自带分析 ----------------*- C++ -*-===//
//
// 第 13 章那个 PixCostAnalysis 是我们自己遍历数出来的。这一章用 MLIR 自带的
// 三样东西，每样都把结果写成属性，好用 FileCheck 断言：
//   ① DominanceInfo      —— 支配关系
//   ② 副作用查询          —— isPure / isMemoryEffectFree / wouldOpBeTriviallyDead
//   ③ DataFlowSolver     —— 一个真的稀疏前向数据流分析（"这个值是不是全零图"）
//
//===----------------------------------------------------------------------===//

#include "Pix/Transforms/Passes.h"

#include "Pix/PixDialect.h"
#include "Pix/PixOps.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace pix {

#define GEN_PASS_DEF_PIXANALYZE
#include "Pix/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// ③ 一个稀疏前向数据流分析：这个值是不是"已知全零的图"
//
// 格（lattice）只有三层：未初始化 → 已知全零 / 已知非全零 → 未知。
// 值类型要满足的契约就是下面这几个成员（照 ConstantPropagationAnalysis 里的
// ConstantValue 抄）：== / print / getUninitialized / isUninitialized / join。
//===----------------------------------------------------------------------===//

class ZeroState {
public:
  ZeroState() = default; // 未初始化
  static ZeroState getUninitialized() { return ZeroState{}; }
  bool isUninitialized() const { return !known.has_value(); }

  static ZeroState getZero() { return ZeroState(true); }
  static ZeroState getNotZero() { return ZeroState(false); }
  /// "未知"用 known=false 表达不了——那是"确定不是零"。所以单独一个标记。
  static ZeroState getUnknown() {
    ZeroState s(false);
    s.unknown = true;
    return s;
  }

  bool isZero() const { return known.value_or(false) && !unknown; }
  bool isUnknown() const { return unknown; }

  bool operator==(const ZeroState &o) const {
    return known == o.known && unknown == o.unknown;
  }
  void print(raw_ostream &os) const {
    os << (isUninitialized() ? "uninit" : isUnknown() ? "unknown"
                               : isZero()             ? "zero"
                                                      : "nonzero");
  }

  /// join 是格的核心：两条路径汇合时取"更保守"的那个。
  /// 这里的偏序是 uninit < zero/nonzero < unknown。
  static ZeroState join(const ZeroState &a, const ZeroState &b) {
    if (a.isUninitialized())
      return b;
    if (b.isUninitialized())
      return a;
    if (a == b)
      return a;
    return getUnknown(); // 结论不一致 → 只能说不知道
  }

private:
  explicit ZeroState(bool isZero) : known(isZero) {}
  std::optional<bool> known;
  bool unknown = false;
};

class ZeroLattice : public dataflow::Lattice<ZeroState> {
public:
  using Lattice::Lattice;
  // 匿名命名空间里的类要用 TypeID（DataFlowSolver 拿它当键），必须显式定义，
  // 否则运行时 abort：
  //   "Using TypeID on a class with an anonymous namespace requires an
  //    explicit TypeID definition"
  // 报错本身把该用的宏名都告诉你了。上游同类代码也是这么写的
  // （mlir/lib/Dialect/XeGPU/Transforms/XeGPUPropagateLayout.cpp:158）。
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZeroLattice)
};

class ZeroAnalysis : public dataflow::SparseForwardDataFlowAnalysis<ZeroLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZeroAnalysis)

  /// 入口状态：函数参数之类我们一无所知的值，一律"未知"。
  /// 这一步很关键——保守的起点保证分析结果是可信的。
  void setToEntryState(ZeroLattice *lattice) override {
    propagateIfChanged(lattice,
                       lattice->join(ZeroState::getUnknown()));
  }

  /// 传递函数：给定操作数的格值，算出结果的格值。这是分析的全部内容。
  LogicalResult visitOperation(Operation *op,
                              ArrayRef<const ZeroLattice *> operands,
                              ArrayRef<ZeroLattice *> results) override {
    auto setAll = [&](ZeroState s) {
      for (ZeroLattice *r : results)
        propagateIfChanged(r, r->join(s));
    };

    // arith.constant dense<0.0> → 零；别的常量 → 非零
    if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
      auto dense = dyn_cast<DenseFPElementsAttr>(cst.getValue());
      if (dense && dense.isSplat() && dense.getSplatValue<APFloat>().isZero())
        setAll(ZeroState::getZero());
      else
        setAll(ZeroState::getNotZero());
      return success();
    }

    // pix.add：两边都是零才是零
    if (isa<AddOp>(op)) {
      bool allZero = llvm::all_of(operands, [](const ZeroLattice *l) {
        return !l->getValue().isUninitialized() && l->getValue().isZero();
      });
      setAll(allZero ? ZeroState::getZero() : ZeroState::getUnknown());
      return success();
    }

    // pix.mul / pix.scale：**任意一边**是零，结果就是零（0 × 任何 = 0）
    // 这条是这个分析最有价值的地方：它推出了单看一个 op 看不出来的结论。
    if (isa<MulOp, ScaleOp>(op)) {
      bool anyZero = llvm::any_of(operands, [](const ZeroLattice *l) {
        return !l->getValue().isUninitialized() && l->getValue().isZero();
      });
      setAll(anyZero ? ZeroState::getZero() : ZeroState::getUnknown());
      return success();
    }

    // pix.transpose：零的转置还是零
    if (isa<TransposeOp>(op)) {
      bool zero = !operands.empty() &&
                  !operands[0]->getValue().isUninitialized() &&
                  operands[0]->getValue().isZero();
      setAll(zero ? ZeroState::getZero() : ZeroState::getUnknown());
      return success();
    }

    setAll(ZeroState::getUnknown());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pix-analyze
//===----------------------------------------------------------------------===//

struct PixAnalyzePass : impl::PixAnalyzeBase<PixAnalyzePass> {
  void runOnOperation() override {
    func::FuncOp fn = getOperation();
    OpBuilder b(fn);

    //=== ① 支配关系 ===//
    // DominanceInfo 是按"最近的 IsolatedFromAbove 祖先"建的（第 13 章那个
    // trait 又一次出现）。这里数一件事：有多少个 pix op 支配着函数里**所有**
    // 别的 pix op —— 也就是"放在它后面一定安全"的那些。
    DominanceInfo dom(fn);
    SmallVector<Operation *> pixOps;
    fn.walk([&](Operation *op) {
      if (op->getDialect() == fn->getContext()->getLoadedDialect("pix"))
        pixOps.push_back(op);
    });
    int64_t dominators = 0;
    for (Operation *a : pixOps)
      if (llvm::all_of(pixOps, [&](Operation *c) {
            return a == c || dom.properlyDominates(a, c);
          }))
        ++dominators;

    //=== ② 副作用查询 ===//
    // 三个函数问的是三件**不同**的事，别混（见第 19 章正文那张表）。
    int64_t pure = 0, effectFree = 0, triviallyDead = 0;
    for (Operation *op : pixOps) {
      if (isPure(op))
        ++pure;
      if (isMemoryEffectFree(op))
        ++effectFree;
      if (wouldOpBeTriviallyDead(op))
        ++triviallyDead;
    }

    //=== ③ 数据流分析 ===//
    // DeadCodeAnalysis 是几乎所有稀疏分析的前提：它负责算"哪些块可达"，
    // 少了它，不可达代码里的格值会污染结果。
    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<ZeroAnalysis>();
    if (failed(solver.initializeAndRun(fn)))
      return signalPassFailure();

    int64_t knownZero = 0;
    fn.walk([&](Operation *op) {
      for (Value res : op->getResults())
        if (auto *l = solver.lookupState<ZeroLattice>(res))
          if (!l->getValue().isUninitialized() && l->getValue().isZero())
            ++knownZero;
    });

    fn->setAttr("pix.dominators", b.getI64IntegerAttr(dominators));
    fn->setAttr("pix.pure", b.getI64IntegerAttr(pure));
    fn->setAttr("pix.effect_free", b.getI64IntegerAttr(effectFree));
    fn->setAttr("pix.trivially_dead", b.getI64IntegerAttr(triviallyDead));
    fn->setAttr("pix.known_zero", b.getI64IntegerAttr(knownZero));
    markAllAnalysesPreserved();
  }
};

} // namespace
} // namespace pix
} // namespace mlir
