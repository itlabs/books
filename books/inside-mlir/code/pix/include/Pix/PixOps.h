//===- PixOps.h - pix 方言的 op ---------------------------------*- C++ -*-===//
#ifndef PIX_PIXOPS_H
#define PIX_PIXOPS_H

// 带属性（尤其是存进 properties 的固有属性）的 op，生成的代码会引用
// BytecodeOpInterface（用于把属性读写进字节码格式）。上游各方言的头文件也都 include 它。
#include "mlir/Bytecode/BytecodeOpInterface.h"

#include "Pix/PixAttrs.h"     // #pix.border
#include "Pix/PixTypes.h"     // !pix.kernel
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
// assemblyFormat 生成出来的 parse()/print() 用到 OpAsmParser / OpAsmPrinter，
// 它们声明在这个头文件里。漏了它，编译会报一堆
// "incomplete type 'mlir::OpAsmParser' used in nested name specifier"——
// 报错指向生成的 .inc，很容易误以为是 TableGen 出了问题，其实是 include 少了。
#include "mlir/IR/OpImplementation.h"
// 规律：ODS 里 include 了某个接口的 .td，C++ 这边就要 include 对应的 .h。
// 这是两套平行的 include 世界，漏一边就会在生成的 .inc 里报"某个接口不是 mlir 的成员"。
#include "mlir/Interfaces/ControlFlowInterfaces.h"  // ReturnLike 背后的 RegionBranchTerminatorOpInterface
#include "mlir/Interfaces/InferTypeOpInterface.h"   // SameOperandsAndResultType
#include "mlir/Interfaces/SideEffectInterfaces.h"   // Pure

// 这个宏 + include 的组合是 MLIR 的惯用手法：GET_OP_CLASSES 决定生成文件里
// 哪一段被展开进来（声明 or 定义）。第 8 章会带你看一眼生成出来的东西。
#define GET_OP_CLASSES
#include "Pix/PixOps.h.inc"

#endif // PIX_PIXOPS_H
