//===- PixToLinalg.cpp - 第 20 章：pix → linalg -----------------*- C++ -*-===//
//
// 第 16 章把 pix 降到了 arith（逐点运算在 tensor 上直接就是逐点的）。
// 那条路最短，但**丢掉了迭代空间**：arith.addf 在 tensor 上是一个不透明的
// "整张图加整张图"，没人知道它由哪些循环组成，于是 tile 不了、融合不了。
//
// linalg 的做法相反：把"迭代空间"和"每个点上算什么"分开写。前者是
// indexing_maps + iterator_types，后者是 region 里那几行标量代码。
// 正因为迭代空间是**显式的、可分析的**，tiling 和融合才能通用地实现。
//
//===----------------------------------------------------------------------===//

#include "Pix/Transforms/Passes.h"

#include "Pix/PixDialect.h"
#include "Pix/PixOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace pix {

#define GEN_PASS_DEF_PIXTOLINALG
#include "Pix/Transforms/Passes.h.inc"

namespace {

/// 造一个"恒等"的 indexing map：(d0, d1) -> (d0, d1)。
/// 逐点运算的三个操作数（两个输入一个输出）都用它——含义是
/// "第 (i, j) 个迭代点读写第 (i, j) 个元素"。
static AffineMap identityMap(unsigned rank, MLIRContext *ctx) {
  return AffineMap::getMultiDimIdentityMap(rank, ctx);
}

/// 逐点二元运算的通用下降：pix.add / pix.mul → linalg.generic。
/// 两者的区别只在 region 里那一个标量 op，所以用模板参数传进来。
template <typename PixOp, typename ScalarOp>
struct PointwiseLowering : OpConversionPattern<PixOp> {
  using OpConversionPattern<PixOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<PixOp>::OpAdaptor;

  LogicalResult matchAndRewrite(PixOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto ty = dyn_cast<RankedTensorType>(op.getType());
    if (!ty || !ty.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "需要静态形状的有阶张量");

    Location loc = op.getLoc();
    // outs 要一个"目标张量"——linalg 是 destination-passing style（第 20 章正文）。
    // 这里只是要一块形状对的地方放结果，所以用 tensor.empty。
    Value init = tensor::EmptyOp::create(rewriter, loc, ty.getShape(),
                                         ty.getElementType());

    AffineMap id = identityMap(ty.getRank(), rewriter.getContext());
    SmallVector<AffineMap> maps{id, id, id};           // lhs, rhs, out
    SmallVector<utils::IteratorType> iters(
        ty.getRank(), utils::IteratorType::parallel);  // 逐点 → 全部 parallel

    auto generic = linalg::GenericOp::create(
        rewriter, loc, TypeRange{ty},
        ValueRange{adaptor.getLhs(), adaptor.getRhs()}, ValueRange{init}, maps,
        iters,
        [&](OpBuilder &b, Location nested, ValueRange args) {
          // args = (lhs 的一个元素, rhs 的一个元素, out 的一个元素)
          Value s = ScalarOp::create(b, nested, args[0], args[1]);
          linalg::YieldOp::create(b, nested, s);
        });

    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

/// pix.scale %img, %f → linalg.generic，region 里乘那个标量。
/// 看点：标量 %f **不进 ins**，而是直接在 region 里引用外面的值。
/// linalg.generic 的 region 不是 IsolatedFromAbove（第 13 章那个 trait），
/// 所以这么写是合法的，而且比先 splat 成一整张图再逐点乘省一次实体化。
struct ScaleToLinalg : OpConversionPattern<ScaleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ScaleOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto ty = dyn_cast<RankedTensorType>(op.getType());
    if (!ty || !ty.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "需要静态形状的有阶张量");

    Location loc = op.getLoc();
    Value init = tensor::EmptyOp::create(rewriter, loc, ty.getShape(),
                                         ty.getElementType());
    AffineMap id = identityMap(ty.getRank(), rewriter.getContext());
    SmallVector<AffineMap> maps{id, id};               // image, out
    SmallVector<utils::IteratorType> iters(ty.getRank(),
                                          utils::IteratorType::parallel);

    Value factor = adaptor.getFactor();
    auto generic = linalg::GenericOp::create(
        rewriter, loc, TypeRange{ty}, ValueRange{adaptor.getImage()},
        ValueRange{init}, maps, iters,
        [&](OpBuilder &b, Location nested, ValueRange args) {
          Value s = arith::MulFOp::create(b, nested, args[0], factor);
          linalg::YieldOp::create(b, nested, s);
        });

    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

/// pix.reduce → linalg.generic，但迭代器类型不同：
/// 输入的 map 是 (d0, d1) -> (d0, d1)，输出的 map 是 (d0, d1) -> ()，
/// 也就是"两个维度都归约掉"。iterator_types 于是全是 reduction。
///
/// **这一条最能说明 indexing map 的表达力**：同一个 linalg.generic，
/// 只是换了 map 和 iterator 类型，就从逐点变成了归约。
struct ReduceToLinalg : OpConversionPattern<ReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ReduceOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto ty = dyn_cast<RankedTensorType>(adaptor.getImage().getType());
    if (!ty || !ty.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "需要静态形状的有阶张量");
    if (op.getKind() != ReduceKind::Sum)
      return rewriter.notifyMatchFailure(op, "这一章只降 sum");

    Location loc = op.getLoc();
    unsigned rank = ty.getRank();
    MLIRContext *ctx = rewriter.getContext();

    // 输出是 0 阶张量（一个标量），初值必须是 0——归约的单位元。
    Value zero = arith::ConstantOp::create(rewriter, loc,
                                           rewriter.getF32FloatAttr(0.0));
    auto scalarTy = RankedTensorType::get({}, ty.getElementType());
    Value empty = tensor::EmptyOp::create(rewriter, loc, ArrayRef<int64_t>{},
                                          ty.getElementType());
    Value init = linalg::FillOp::create(rewriter, loc, ValueRange{zero},
                                        ValueRange{empty}).getResult(0);

    SmallVector<AffineMap> maps{
        identityMap(rank, ctx),                            // 输入：逐点读
        AffineMap::get(rank, /*symbolCount=*/0, {}, ctx)};  // 输出：() ← 全归约
    SmallVector<utils::IteratorType> iters(rank,
                                          utils::IteratorType::reduction);

    auto generic = linalg::GenericOp::create(
        rewriter, loc, TypeRange{scalarTy}, ValueRange{adaptor.getImage()},
        ValueRange{init}, maps, iters,
        [&](OpBuilder &b, Location nested, ValueRange args) {
          // args[1] 是"累加器当前值"——destination-passing style 的体现
          Value s = arith::AddFOp::create(b, nested, args[0], args[1]);
          linalg::YieldOp::create(b, nested, s);
        });

    // pix.reduce 的结果是 f32 标量，linalg 给的是 tensor<f32>，取出来。
    Value scalar = tensor::ExtractOp::create(rewriter, loc,
                                             generic.getResult(0), ValueRange{});
    rewriter.replaceOp(op, scalar);
    return success();
  }
};

/// pix.convolve → tensor.pad + linalg.generic（4 维迭代空间的 stencil）
///
/// 这一条是第 21 章 halo 那一节的素材。要点在 indexing map：
///   输入图：(d0,d1,d2,d3) -> (d0+d2, d1+d3)   ← **下标带偏移**
///   权重：  (d0,d1,d2,d3) -> (d2, d3)
///   输出：  (d0,d1,d2,d3) -> (d0, d1)
/// iterator_types = [parallel, parallel, reduction, reduction]
///
/// 输入下标是 d0+d2，所以一个输出点要读一个 R×C 的邻域。这意味着
/// **输出 tile 是 4×4 时，输入 tile 必须是 6×6**——那两圈多出来的就是 halo。
/// pix.convolve 是"same"语义（输入输出同形），所以先 pad 一圈半径。
struct ConvolveToLinalg : OpConversionPattern<ConvolveOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ConvolveOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto imgTy = dyn_cast<RankedTensorType>(adaptor.getImage().getType());
    if (!imgTy || !imgTy.hasStaticShape() || imgTy.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "需要静态形状的二维图");

    // 卷积核的尺寸从类型里读——这正是第 9 章把尺寸放进类型的回报。
    auto kty = dyn_cast<KernelType>(op.getKernel().getType());
    if (!kty)
      return rewriter.notifyMatchFailure(op, "kernel 操作数不是 !pix.kernel");
    int64_t kr = kty.getRows(), kc = kty.getCols();
    if (kr % 2 == 0 || kc % 2 == 0)
      return rewriter.notifyMatchFailure(op, "只处理奇数边长（有明确中心）");

    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    Type elt = imgTy.getElementType();

    // 权重张量：pix.kernel 是 ConstantLike，直接把它的属性拿出来造常量。
    auto kernelDef = op.getKernel().getDefiningOp<KernelOp>();
    if (!kernelDef)
      return rewriter.notifyMatchFailure(op, "kernel 不是直接由 pix.kernel 产出");
    auto weights = dyn_cast<DenseFPElementsAttr>(kernelDef.getWeights());
    if (!weights)
      return rewriter.notifyMatchFailure(op, "权重不是 DenseFPElementsAttr");
    Value w = arith::ConstantOp::create(
        rewriter, loc, RankedTensorType::get({kr, kc}, elt), weights);

    Value zero = arith::ConstantOp::create(rewriter, loc,
                                           rewriter.getZeroAttr(elt));

    // same 语义 → 先 pad 一圈半径。边界值这里固定用 0（对应 #pix.border<zero>）；
    // clamp / wrap 需要在 pad 的 region 里算下标，留作练习。
    int64_t rr = kr / 2, rc = kc / 2;
    SmallVector<int64_t> lo{rr, rc}, hi{rr, rc};
    auto padTy = RankedTensorType::get(
        {imgTy.getDimSize(0) + 2 * rr, imgTy.getDimSize(1) + 2 * rc}, elt);
    auto pad = tensor::PadOp::create(rewriter, loc, padTy, adaptor.getImage(),
                                     lo, hi, ValueRange{}, ValueRange{});
    {
      OpBuilder::InsertionGuard g(rewriter);
      Block *body = rewriter.createBlock(
          &pad.getRegion(), pad.getRegion().end(),
          {rewriter.getIndexType(), rewriter.getIndexType()}, {loc, loc});
      rewriter.setInsertionPointToStart(body);
      tensor::YieldOp::create(rewriter, loc, zero);
    }

    // 累加器初值 0
    Value empty = tensor::EmptyOp::create(rewriter, loc, imgTy.getShape(), elt);
    Value init = linalg::FillOp::create(rewriter, loc, ValueRange{zero},
                                        ValueRange{empty}).getResult(0);

    // 四个迭代维：d0/d1 走输出的行列，d2/d3 走卷积核窗口
    auto d0 = getAffineDimExpr(0, ctx), d1 = getAffineDimExpr(1, ctx);
    auto d2 = getAffineDimExpr(2, ctx), d3 = getAffineDimExpr(3, ctx);
    SmallVector<AffineMap> maps{
        AffineMap::get(4, 0, {d0 + d2, d1 + d3}, ctx), // 输入：带偏移 → halo 的来源
        AffineMap::get(4, 0, {d2, d3}, ctx),           // 权重
        AffineMap::get(4, 0, {d0, d1}, ctx)};          // 输出
    SmallVector<utils::IteratorType> iters{
        utils::IteratorType::parallel, utils::IteratorType::parallel,
        utils::IteratorType::reduction, utils::IteratorType::reduction};

    auto generic = linalg::GenericOp::create(
        rewriter, loc, TypeRange{imgTy},
        ValueRange{pad.getResult(), w}, ValueRange{init}, maps, iters,
        [&](OpBuilder &b, Location nested, ValueRange args) {
          Value m = arith::MulFOp::create(b, nested, args[0], args[1]);
          Value s = arith::AddFOp::create(b, nested, args[2], m);
          linalg::YieldOp::create(b, nested, s);
        });

    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

struct PixToLinalgPass : impl::PixToLinalgBase<PixToLinalgPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ConversionTarget target(*ctx);
    target.addLegalDialect<arith::ArithDialect, linalg::LinalgDialect,
                           tensor::TensorDialect, func::FuncDialect>();
    target.addIllegalOp<AddOp, MulOp, ScaleOp, ReduceOp, ConvolveOp>();

    TypeConverter converter;
    converter.addConversion([](Type ty) { return ty; });

    RewritePatternSet patterns(ctx);
    patterns.add<PointwiseLowering<AddOp, arith::AddFOp>,
                 PointwiseLowering<MulOp, arith::MulFOp>, ScaleToLinalg,
                 ReduceToLinalg, ConvolveToLinalg>(converter, ctx);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace pix
} // namespace mlir
