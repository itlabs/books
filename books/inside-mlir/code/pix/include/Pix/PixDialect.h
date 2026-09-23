//===- PixDialect.h - pix 方言的 C++ 入口 -----------------------*- C++ -*-===//
#ifndef PIX_PIXDIALECT_H
#define PIX_PIXDIALECT_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"

// TableGen 生成的方言类声明。名字里的 "PixOps" 来自 add_mlir_dialect(PixOps pix)
// 的第一个参数——生成文件都以那个名字为前缀，这一点第 7 章特意提醒过。
#include "Pix/PixOpsDialect.h.inc"

#endif // PIX_PIXDIALECT_H
