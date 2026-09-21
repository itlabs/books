//===- CostAnalysis.h - pix 的第一个分析 -----------------------*- C++ -*-===//
//
// 《深入 MLIR》第 13 章。MLIR 的"分析"不需要继承任何基类、不需要注册，
// 只要满足一个约定：**有一个接受 Operation* 的构造函数**。
// AnalysisManager 用类型本身（TypeID）当键，第一次 getAnalysis<T>() 时
// 构造一个，之后命中缓存直接给同一份。
//
//===----------------------------------------------------------------------===//

#ifndef PIX_TRANSFORMS_COSTANALYSIS_H
#define PIX_TRANSFORMS_COSTANALYSIS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace pix {

/// 统计一段 IR 里 pix 的计算规模：多少个 pix op、结果张量一共多少元素。
/// 真实的 cost model 当然复杂得多，这里只要"有一份需要遍历才能得到、
/// 值得被缓存的信息"就够讲清机制了。
class PixCostAnalysis {
public:
  /// 这个构造函数就是全部约定。op 是分析的**作用范围**——
  /// 函数级 pass 传进来的是那个 func.func，所以统计天然只覆盖本函数。
  explicit PixCostAnalysis(Operation *op) {
    op->walk([&](Operation *nested) {
      if (nested->getDialect() != op->getContext()->getLoadedDialect("pix"))
        return;
      ++numOps;
      for (Type ty : nested->getResultTypes())
        if (auto shaped = dyn_cast<ShapedType>(ty))
          if (shaped.hasStaticShape())
            elements += shaped.getNumElements();
    });
  }

  int64_t getNumOps() const { return numOps; }
  int64_t getElements() const { return elements; }

private:
  int64_t numOps = 0;
  int64_t elements = 0;
};

} // namespace pix
} // namespace mlir

#endif // PIX_TRANSFORMS_COSTANALYSIS_H
