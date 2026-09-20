//===- PixDialect.cpp - pix 方言 -------------------------------*- C++ -*-===//
#include "Pix/PixDialect.h"
#include "Pix/PixOps.h"

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
