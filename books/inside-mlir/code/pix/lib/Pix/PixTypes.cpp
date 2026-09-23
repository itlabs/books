//===- PixTypes.cpp - pix 方言的自定义类型 ---------------------*- C++ -*-===//
#include "Pix/PixTypes.h"
#include "Pix/PixDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"  // 生成的 parser/printer 要用
#include "llvm/ADT/TypeSwitch.h"            // 生成的分派代码要用

using namespace mlir;
using namespace mlir::pix;

#define GET_TYPEDEF_CLASSES
#include "Pix/PixOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// !pix.kernel 的参数验证（ODS 里 genVerifyDecl = 1 换来这个函数的声明）
//===----------------------------------------------------------------------===//

// 第一个参数是"怎么报错"的回调——类型验证发生在还没有 op 的时候（类型可以独立存在），
// 所以不能像 op 那样用 emitOpError，只能用框架给的这个发射器。
LogicalResult KernelType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 int64_t rows, int64_t cols) {
  if (rows <= 0 || cols <= 0)
    return emitError() << "卷积核的行列必须是正数，收到 " << rows << "x" << cols;
  if (rows % 2 == 0 || cols % 2 == 0)
    return emitError() << "卷积核的行列必须是奇数（才有明确的中心点），收到 "
                       << rows << "x" << cols;
  return success();
}

// 把生成出来的类型列表注册进方言。和 op 的 addOperations 是一个套路。
void PixDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Pix/PixOpsTypes.cpp.inc"
      >();
}
