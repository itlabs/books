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
  PixTypeConverter() {
    // ① 身份规则：默认原样保留。必须有。
    addConversion([](Type ty) { return ty; });

    // ② !pix.kernel<R x C> → tensor<RxCxf32>
    //    返回 std::nullopt 表示"我不管这个类型"，返回空 Type 表示"转换失败"。
    //    两者含义不同，别混。
    addConversion([](KernelType kty) -> std::optional<Type> {
      return RankedTensorType::get({kty.getRows(), kty.getCols()},
                                   Float32Type::get(kty.getContext()));
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

    PixTypeConverter converter;
    RewritePatternSet patterns(ctx);
    patterns.add<AddLowering, MulLowering, ScaleLowering>(converter, ctx);

    if (lowerKernels) {
      target.addIllegalOp<KernelOp>();
      patterns.add<KernelLowering>(converter, ctx);
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
