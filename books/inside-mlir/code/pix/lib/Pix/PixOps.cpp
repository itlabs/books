//===- PixOps.cpp - pix 方言的 op ------------------------------*- C++ -*-===//
//
// op 的骨架（builder、parser、printer、verifyInvariants）全部由 TableGen 生成；
// 这里只放"必须手写"的部分：第 10 章的 verifier 与接口实现，第 11 章的 folder。
//
//===----------------------------------------------------------------------===//
#include "Pix/PixOps.h"
#include "Pix/PixDialect.h"

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
