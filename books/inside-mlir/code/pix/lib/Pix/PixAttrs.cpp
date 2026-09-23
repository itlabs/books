//===- PixAttrs.cpp - pix 方言的自定义属性 ---------------------*- C++ -*-===//
#include "Pix/PixAttrs.h"
#include "Pix/PixDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::pix;

// 枚举的字符串互转（stringifyBorderMode / symbolizeBorderMode）
#include "Pix/PixEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Pix/PixAttrs.cpp.inc"

void PixDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Pix/PixAttrs.cpp.inc"
      >();
}
