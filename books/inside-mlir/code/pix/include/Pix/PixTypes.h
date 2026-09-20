//===- PixTypes.h - pix 方言的自定义类型 -----------------------*- C++ -*-===//
#ifndef PIX_PIXTYPES_H
#define PIX_PIXTYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

// 注意生成文件的名字：add_mlir_dialect(PixOps pix) 让所有生成文件都以 PixOps 为前缀，
// 所以类型的声明在 PixOpsTypes.h.inc，而不是 PixTypes.h.inc（第 7 章那个坑）。
#define GET_TYPEDEF_CLASSES
#include "Pix/PixOpsTypes.h.inc"

#endif // PIX_PIXTYPES_H
