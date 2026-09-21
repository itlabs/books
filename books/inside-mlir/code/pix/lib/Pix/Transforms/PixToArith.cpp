//===- PixToArith.cpp - 第 16 章：Dialect Conversion 的四个部件 -*- C++ -*-===//
//
// 把能下降的 pix op 换成 arith/tensor。这个文件刻意把四个部件分成四段，
// 顺序就是第 16 章正文的顺序。
//
//===----------------------------------------------------------------------===//

#include "Pix/Transforms/Passes.h"

#include "Pix/PixDialect.h"
#include "Pix/PixOps.h"
#include "Pix/PixTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace pix {

#define GEN_PASS_DEF_PIXTOARITH
#include "Pix/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// 部件二：TypeConverter —— 类型怎么换
//
// 最容易踩的坑在第一条：**必须显式写"其他类型原样保留"这条身份规则**。
// TypeConverter 默认什么都不认，漏了它，tensor<4x4xf32> 这种根本没打算换的
// 类型也会被判成"无法转换"，于是连函数签名都过不去。
//===----------------------------------------------------------------------===//

class PixTypeConverter : public TypeConverter {
public:
  /// withMaterializations=false 时故意不装物化钩子，好在第 16 章末尾制造那次
  /// "unresolved materialization" 失败；第 17 章把它们装上。
  /// oneToN=true 时把 !pix.kernel 展成**两个**值（权重张量 + 半径），
  /// 演示 1:N 类型转换。真实动机：下降成循环之后，边界处理需要半径当循环边界。
  explicit PixTypeConverter(bool withMaterializations = true,
                            bool oneToN = false) {
    // ① 身份规则：默认原样保留。必须有。
    addConversion([](Type ty) { return ty; });

    // ② !pix.kernel<R x C> → tensor<RxCxf32>
    //    返回 std::nullopt 表示"我不管这个类型"，返回空 Type 表示"转换失败"。
    //    两者含义不同，别混。
    addConversion([](KernelType kty) -> std::optional<Type> {
      return RankedTensorType::get({kty.getRows(), kty.getCols()},
                                   Float32Type::get(kty.getContext()));
    });

    // 1:N 形式的 addConversion：回调第二个参数是个 SmallVectorImpl<Type>&，
    // 往里塞几个类型就展成几个值。塞 0 个表示"这个类型直接消失"。
    // 注意它必须注册在 1:1 那条**之后**——后注册的先试。
    if (oneToN) {
      addConversion([](KernelType kty,
                       SmallVectorImpl<Type> &out) -> std::optional<LogicalResult> {
        MLIRContext *ctx = kty.getContext();
        out.push_back(RankedTensorType::get({kty.getRows(), kty.getCols()},
                                            Float32Type::get(ctx)));
        out.push_back(IndexType::get(ctx));   // 半径
        return success();
      });
    }

    if (!withMaterializations)
      return;

    //=== 第 17 章：两个方向的物化 ===//
    //
    // source：把**已经换成新类型的值**转回原来的源类型。
    // 用在"原值还有使用者活到了转换之后"的场合 —— 比如 pix.kernel 已经变成
    // tensor 了，而 pix.convolve 还在等 !pix.kernel。
    addSourceMaterialization([](OpBuilder &b, KernelType want,
                                ValueRange inputs, Location loc) -> Value {
      // 1:N 的关键：inputs **可能有多个**。上面那条 1:N 规则把 !pix.kernel
      // 展成了（权重张量, 半径）两个值，物化时就会收到两个。
      // 我们只需要第一个（半径是从类型里能重新算出来的冗余信息）。
      // 返回 nullptr 表示"我物化不了"，框架会去试别的钩子、都不行才报错——
      // 一开始我这里写了 `if (inputs.size() != 1) return nullptr;`，
      // 于是得到 "failed to legalize unresolved materialization
      // from ('tensor<3x3xf32>', 'index')"，报错里那两个类型就是线索。
      if (inputs.empty())
        return nullptr;
      if (!isa<RankedTensorType>(inputs.front().getType()))
        return nullptr;
      return ToKernelOp::create(b, loc, want, inputs.front());
    });

    // target：把一个值转成**目标**类型。方向和上面相反。
    // 两个都要装：转换是逐步的，两种"接不上"都会发生。
    addTargetMaterialization([](OpBuilder &b, RankedTensorType want,
                                ValueRange inputs, Location loc) -> Value {
      if (inputs.size() != 1)
        return nullptr;
      if (!isa<KernelType>(inputs.front().getType()))
        return nullptr;          // 不是我们这条规则该管的，交给别人
      return FromKernelOp::create(b, loc, want, inputs.front());
    });
  }
};

//===----------------------------------------------------------------------===//
// 部件三：ConversionPattern —— 和普通 RewritePattern 差在哪
//
// 差别只有一个，但很关键：多出一个 **adaptor** 参数，里面装的是
// **已经被转换过的操作数**。直接用 op.getLhs() 拿到的是**旧**的值，
// 在转换过程中它可能已经没人认了。这是 conversion pattern 最常见的错误来源。
//===----------------------------------------------------------------------===//

// pix.add → arith.addf（arith 的浮点运算在 tensor 上就是逐点的）
struct AddLowering : OpConversionPattern<AddOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(AddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::AddFOp>(op, adaptor.getLhs(),
                                               adaptor.getRhs());
    return success();
  }
};

// pix.mul → arith.mulf
struct MulLowering : OpConversionPattern<MulOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(MulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::MulFOp>(op, adaptor.getLhs(),
                                               adaptor.getRhs());
    return success();
  }
};

// pix.scale %img, %f → arith.mulf %img, tensor.splat %f
// 一个 pix op 变成两个 op，这在 conversion pattern 里完全正常。
struct ScaleLowering : OpConversionPattern<ScaleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ScaleOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto ty = dyn_cast<RankedTensorType>(adaptor.getImage().getType());
    if (!ty)
      return rewriter.notifyMatchFailure(op, "图不是有阶张量，splat 不出来");

    Value splat = tensor::SplatOp::create(rewriter, op.getLoc(), ty,
                                         adaptor.getFactor());
    rewriter.replaceOpWithNewOp<arith::MulFOp>(op, adaptor.getImage(), splat);
    return success();
  }
};

// pix.kernel → arith.constant dense<...> : tensor<RxCxf32>
// 这一条会让**结果类型发生变化**（!pix.kernel<3 x 3> → tensor<3x3xf32>），
// 于是 TypeConverter 里那条 kernel 规则终于被用上。
// 但下游 pix.convolve 还在等一个 !pix.kernel —— 第 17 章就从这里开始。
struct KernelLowering : OpConversionPattern<KernelOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(KernelOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // 问 TypeConverter：这个结果类型该变成什么？
    Type newTy = getTypeConverter()->convertType(op.getType());
    if (!newTy)
      return rewriter.notifyMatchFailure(op, "kernel 类型转换不出来");

    auto weights = dyn_cast<DenseFPElementsAttr>(op.getWeights());
    if (!weights)
      return rewriter.notifyMatchFailure(op, "权重不是 DenseFPElementsAttr");

    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, newTy, weights);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// 部件一 + 部件四：ConversionTarget 与 driver
//===----------------------------------------------------------------------===//

struct PixToArithPass : impl::PixToArithBase<PixToArithPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    // 部件一：什么是"合法"的。
    ConversionTarget target(*ctx);
    target.addLegalDialect<arith::ArithDialect, tensor::TensorDialect,
                           func::FuncDialect>();

    // 只把**有下降路径**的那几个标成非法。其余 pix op 我们一个字都不说——
    // 于是它们处于第四种状态 **Unknown**（既没标合法也没标非法）。
    // 合法性其实是四态（DialectConversion.h:1265 的注释写得很清楚）：
    //   Legal(递归) / Legal(非递归) / Illegal / Unknown
    // 而 Unknown「视上下文当作合法或非法」——partial 当它合法（原样留下），
    // full 当它非法（整个转换失败）。这就是 partial 与 full 的真正分界。
    target.addIllegalOp<AddOp, MulOp, ScaleOp>();

    PixTypeConverter converter(/*withMaterializations=*/!noMaterialize,
                               /*oneToN=*/oneToN);
    RewritePatternSet patterns(ctx);
    patterns.add<AddLowering, MulLowering, ScaleLowering>(converter, ctx);

    if (lowerKernels) {
      target.addIllegalOp<KernelOp>();
      patterns.add<KernelLowering>(converter, ctx);
    }

    // 第 17 章：函数签名转换。
    //
    // 签名里的类型不会被上面那些 pattern 碰到——函数的参数和返回类型不是
    // 任何 op 的操作数。要换它们得专门有一条 pattern，上游给了现成的。
    //
    // 配套的关键一步是把 func.func 标成**动态合法**：签名还没转完时它非法，
    // 转完了就合法。少了这一步，driver 永远不知道该不该处理它。
    if (convertSignatures) {
      populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                    converter);
      target.addDynamicallyLegalOp<func::FuncOp>([&converter](func::FuncOp fn) {
        return converter.isSignatureLegal(fn.getFunctionType());
      });
    }

    // 部件四：driver。
    // partial —— "把非法的换掉，别的别管"，绝大多数真实下降都用这个。
    // full    —— "转换完之后整个 IR 必须全部合法"，任何漏网的 op 都算失败。
    if (full) {
      if (failed(applyFullConversion(getOperation(), target,
                                     std::move(patterns))))
        signalPassFailure();
      return;
    }
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace pix
} // namespace mlir
