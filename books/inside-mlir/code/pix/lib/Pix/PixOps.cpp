//===- PixOps.cpp - pix 方言的 op ------------------------------*- C++ -*-===//
//
// op 的骨架（builder、parser、printer、verifyInvariants）全部由 TableGen 生成；
// 这里只放"必须手写"的部分：第 10 章的 verifier 与接口实现，第 11 章的 folder。
//
//===----------------------------------------------------------------------===//
#include "Pix/PixOps.h"
#include "Pix/PixDialect.h"

// 第 11 章：canonicalize pattern 要**造** arith.constant / arith.mulf，
// 于是这里必须依赖 arith 方言（CMakeLists 里也要加 MLIRArithDialect）。
// 只解析别人的 op 不需要依赖，要创建才需要——这条界线值得记住。
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::pix;

#define GET_OP_CLASSES
#include "Pix/PixOps.cpp.inc"

//===----------------------------------------------------------------------===//
// pix.transpose
//===----------------------------------------------------------------------===//

LogicalResult TransposeOp::verify() {
  // 注意：走到这里时，ODS 生成的检查（两边都是 F32Tensor）已经过了。
  // 我们只需要补"形状互为转置"这一条。
  auto inTy = cast<RankedTensorType>(getImage().getType());
  auto outTy = cast<RankedTensorType>(getResult().getType());

  if (inTy.getRank() != 2 || outTy.getRank() != 2)
    return emitOpError("只支持二维图像，收到 ") << inTy << " 和 " << outTy;

  ArrayRef<int64_t> in = inTy.getShape(), out = outTy.getShape();
  // 动态维（?）无从比较，放过；静态维必须逐对交换。
  for (auto [i, o] : {std::pair{in[0], out[1]}, std::pair{in[1], out[0]}}) {
    if (!ShapedType::isDynamic(i) && !ShapedType::isDynamic(o) && i != o)
      return emitOpError("结果形状必须是输入形状的逆序：输入 ")
             << inTy << "，结果应为 " << out[1] << "x" << out[0]
             << "，实际是 " << outTy;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// pix.kernel
//===----------------------------------------------------------------------===//

LogicalResult KernelOp::verify() {
  auto kernelTy = getResult().getType();          // !pix.kernel<R x C>
  auto weightsTy = dyn_cast<ShapedType>(getWeights().getType());
  if (!weightsTy || weightsTy.getRank() != 2)
    return emitOpError("权重必须是二维的 elements 属性");

  if (weightsTy.getDimSize(0) != kernelTy.getRows() ||
      weightsTy.getDimSize(1) != kernelTy.getCols())
    return emitOpError("权重形状与卷积核类型不一致：权重是 ")
           << weightsTy.getDimSize(0) << "x" << weightsTy.getDimSize(1)
           << "，而类型说的是 " << kernelTy.getRows() << "x"
           << kernelTy.getCols();
  return success();
}

//===----------------------------------------------------------------------===//
// pix.pipeline
//===----------------------------------------------------------------------===//

// RegionBranchOpInterface 要两个方法：一个说控制流怎么在"本 op"与"它的 region"
// 之间流动，另一个说每条边的**落点**接收哪些值。实现它们之后，通用机制会顺带替我们
// 检查每条边上的值个数与类型（也就是 yield 与 op 结果的对齐）——这就是"实现接口
// 换来免费检查"的最小例子。
void PipelineOp::getSuccessorRegions(RegionBranchPoint point,
                                     SmallVectorImpl<RegionSuccessor> &regions) {
  // 从 pipeline 自己出发 → 进入 body
  if (point.isParent()) {
    regions.push_back(RegionSuccessor(&getBody()));
    return;
  }
  // 从 body 出来 → 回到 pipeline
  regions.push_back(RegionSuccessor(getOperation()));
}

// 落点接收哪些值。RegionSuccessor 本身只记"落在哪"（region 还是 op），不带值，
// 所以得单独告诉框架：落在本 op 上时，接收方是本 op 的结果；落在 body 上时，
// body 没有块参数，接收方为空。前者正是 yield 要对齐的目标。
ValueRange PipelineOp::getSuccessorInputs(RegionSuccessor successor) {
  if (successor.isOperation())
    return getOperation()->getResults();
  return {};
}

//===----------------------------------------------------------------------===//
// 第 11 章：fold —— 只许"就地算出答案"，不许造 op
//===----------------------------------------------------------------------===//

// fold 的返回值是 OpFoldResult：要么一个 Attribute（算出了常量），
// 要么一个已经存在的 Value（这个 op 其实等于别人），要么空（折不动）。
// 参数 adaptor 提供的是"每个操作数**如果是常量**，它的 Attribute；否则为 null"。

// ① 返回**已有的值**：加一张全零图等于什么都没加。
//    Commutative 不会帮你把常量挪到右边（那是 arith 自己 pattern 做的事），
//    所以两边都得看。
OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  auto isZeroImage = [](Attribute attr) {
    auto dense = dyn_cast_or_null<DenseFPElementsAttr>(attr);
    return dense && dense.isSplat() &&
           dense.getSplatValue<APFloat>().isZero();
  };
  if (isZeroImage(adaptor.getRhs()))
    return getLhs();
  if (isZeroImage(adaptor.getLhs()))
    return getRhs();
  return {};
}

// ② 返回**算出来的常量**（一个 Attribute）：整张常量图在编译期就归约完。
//    这是 OpFoldResult 的另一半，也是"常量折叠"最本来的含义。
//
// 第 12 章：kind 变成枚举之后，这里按三种方式分别算。注意 splat 那条捷径
// 对三者都成立，但理由不同：sum 是"乘以元素个数"，max/min 是"就是那个值"。
OpFoldResult ReduceOp::fold(FoldAdaptor adaptor) {
  auto dense = dyn_cast_or_null<DenseFPElementsAttr>(adaptor.getImage());
  if (!dense)
    return {};
  // 空图的 max/min 没有定义（min 该返回什么？），直接不折。
  if (dense.getNumElements() == 0)
    return {};

  const ReduceKind kind = getKind();

  if (dense.isSplat()) {
    APFloat v = dense.getSplatValue<APFloat>();
    APFloat r = kind == ReduceKind::Sum
                    ? v * APFloat(static_cast<float>(dense.getNumElements()))
                    : v; // max/min 全都一样，就是它自己
    return FloatAttr::get(getResult().getType(), r);
  }

  auto values = dense.getValues<APFloat>();
  APFloat acc = *values.begin();
  bool first = true;
  for (APFloat v : values) {
    if (first) {
      first = false;
      if (kind == ReduceKind::Sum)
        acc = v;
      continue;
    }
    switch (kind) {
    case ReduceKind::Sum:
      acc = acc + v;
      break;
    case ReduceKind::Max:
      acc = maxnum(acc, v);
      break;
    case ReduceKind::Min:
      acc = minnum(acc, v);
      break;
    }
  }
  return FloatAttr::get(getResult().getType(), acc);
}

OpFoldResult ScaleOp::fold(FoldAdaptor adaptor) {
  // 系数是常量 1.0 → 整个 op 等于它的输入图
  if (auto f = dyn_cast_or_null<FloatAttr>(adaptor.getFactor()))
    if (f.getValue().isExactlyValue(1.0))
      return getImage();
  return {};
}

// 双重转置互相抵消。这里要看**操作数是谁产生的**，adaptor 帮不上忙
// （它只管常量），所以直接顺着 def-use 链问一句。
OpFoldResult TransposeOp::fold(FoldAdaptor adaptor) {
  auto inner = getImage().getDefiningOp<TransposeOp>();
  if (!inner)
    return {};
  // 防身：形状真的绕回来了才抵消。verifier 已保证每一层都是逆序，
  // 但 fold 不该依赖"别人一定验过"这种假设。
  if (inner.getImage().getType() != getResult().getType())
    return {};
  return inner.getImage();
}

//===----------------------------------------------------------------------===//
// 第 11 章：canonicalize —— 需要造新 op 时走这条路
//===----------------------------------------------------------------------===//

namespace {

// scale(scale(%x, a), b) → scale(%x, a * b)
//
// 为什么不能写成 fold：结果里有一个**新的** arith.mulf。
// 注意这里不要求 a、b 是常量——直接造 arith.mulf 就行；如果它们恰好都是常量，
// arith 自己的 folder 会在同一轮 canonicalize 里把 mulf 折掉。
// 两个方言的化简就这样自动接力，谁都不用知道对方存在。
struct MergeNestedScale : OpRewritePattern<ScaleOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ScaleOp op,
                                PatternRewriter &rewriter) const override {
    auto inner = op.getImage().getDefiningOp<ScaleOp>();
    if (!inner)
      return failure();

    Value merged = arith::MulFOp::create(rewriter, op.getLoc(),
                                        inner.getFactor(), op.getFactor());
    rewriter.replaceOpWithNewOp<ScaleOp>(op, inner.getImage(), merged);
    return success();
  }
};

// add(%x, %x) → scale(%x, 2.0)
//
// 同样是"造新 op"，所以也只能是 pattern。
// 方向很关键：往 scale 走是**收敛**的（一个 op 换一个 op，且不会再触发自己）；
// 反过来写 scale→add 就和这条互为逆操作，两条同时注册会让 greedy driver
// 原地打转——第 11 章正文专门拿这个当反例。
struct AddSelfToScale : OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getLhs() != op.getRhs())
      return failure();

    Value two = arith::ConstantOp::create(rewriter, op.getLoc(),
                                          rewriter.getF32FloatAttr(2.0));
    rewriter.replaceOpWithNewOp<ScaleOp>(op, op.getLhs(), two);
    return success();
  }
};

} // namespace

void ScaleOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                          MLIRContext *context) {
  patterns.add<MergeNestedScale>(context);
}

void AddOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
  patterns.add<AddSelfToScale>(context);
}
