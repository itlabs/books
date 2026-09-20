//===- PixAttrs.h - pix 方言的自定义属性 -----------------------*- C++ -*-===//
#ifndef PIX_PIXATTRS_H
#define PIX_PIXATTRS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

// 枚举（enum class BorderMode）与它的字符串互转函数。
// 这个 .inc 由我们在 include/Pix/CMakeLists.txt 里手写的 mlir_tablegen 调用生成。
#include "Pix/PixEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "Pix/PixAttrs.h.inc"

#endif // PIX_PIXATTRS_H
