//===- PixDialect.td - pix 方言的定义 ----------------------*- tablegen -*-===//
//
// 《深入 MLIR》第 7 章。这里只做两件事：
//   1. 声明方言本身（名字、命名空间、摘要）
//   2. 给本方言的 op 定义一个基类 Pix_Op，省去每个 op 重复写方言参数
//
//===----------------------------------------------------------------------===//

#ifndef PIX_DIALECT
#define PIX_DIALECT

include "mlir/IR/OpBase.td"

def Pix_Dialect : Dialect {
  // 这个名字就是 IR 里的前缀：pix.add、pix.convolve……
  let name = "pix";
  let summary = "一个小型图像处理方言（《深入 MLIR》贯穿案例）";
  let description = [{
    `pix` 用来表达图像处理里的三类计算：逐点运算、邻域（stencil）运算、以及全图归约。
    选这三类是因为它们足以演示 tiling、融合、bufferization、向量化与 GPU 下降——
    而纯标量的玩具语言演示不了这些。
  }];
  // 生成的 C++ 类会放进这个命名空间：mlir::pix::PixDialect
  let cppNamespace = "::mlir::pix";

  // 第 9 章：有了自定义类型与属性之后必须打开这两个开关，
  // 否则 !pix.kernel<3x3> / #pix.border<clamp> 这种写法没人解析。
  // 打开后 TableGen 会替方言生成 parseType/printType、parseAttribute/printAttribute。
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;

  // 这两个函数由我们自己在 .cpp 里实现（把生成的类型/属性列表注册进方言）。
  let extraClassDeclaration = [{
    void registerTypes();
    void registerAttributes();
  }];
}

// 本方言所有 op 的基类。mnemonic 是 op 名里点号后面的部分。
class Pix_Op<string mnemonic, list<Trait> traits = []>
    : Op<Pix_Dialect, mnemonic, traits>;

#endif // PIX_DIALECT
