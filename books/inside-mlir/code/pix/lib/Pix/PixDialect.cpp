//===- PixDialect.cpp - pix 方言 -------------------------------*- C++ -*-===//
#include "Pix/PixDialect.h"
#include "Pix/PixOps.h"

// 第 11 章：.td 里写了 dependentDialects，生成的 PixOpsDialect.cpp.inc 里就会出现
// getContext()->loadDialect<::mlir::arith::ArithDialect>()。那句话要能编译，
// 这个头就得在 include 生成代码**之前**出现——否则报 "'arith' is not a member of 'mlir'"。
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace mlir::pix;

// 方言类的实现（构造、名字、initialize 的声明都由 TableGen 生成）
#include "Pix/PixOpsDialect.cpp.inc"

// initialize() 是方言唯一必须手写的部分：把本方言的 op（以及以后的类型、属性）
// 注册进来。注册的意思就是"让 MLIRContext 知道 pix.add 这个名字对应哪个 C++ 类"。
void PixDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Pix/PixOps.cpp.inc"
      >();
  // 第 9 章：类型与属性也要注册。实现分别在 PixTypes.cpp / PixAttrs.cpp，
  // 拆开是为了让每个文件只 include 自己需要的那套生成代码。
  registerTypes();
  registerAttributes();
}

//===----------------------------------------------------------------------===//
// 第 11 章：把 fold 折出来的 Attribute 落地成一条 op
//===----------------------------------------------------------------------===//

// pix.reduce 折出来的是一个 f32 标量常量，而 pix 自己没有标量常量 op，
// 于是借 arith.constant 落地。返回 nullptr 表示"这个常量我落不了"，
// 框架会据此放弃那次折叠（而不是崩掉）。
Operation *PixDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type type, Location loc) {
  auto typed = dyn_cast<TypedAttr>(value);
  if (!typed || typed.getType() != type)
    return nullptr;
  if (!isa<FloatType>(type))
    return nullptr;
  return arith::ConstantOp::create(builder, loc, typed);
}
