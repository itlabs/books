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

  // 第 11 章：我们的 canonicalize pattern 会**造** arith.constant / arith.mulf。
  // 想创建某个方言的 op，那个方言必须已经加载进 MLIRContext——这里声明一下，
  // 于是"加载 pix"就顺带加载 arith。
  //
  // 漏掉它的后果很有欺骗性：只要输入 IR 里本来就出现过 arith 的 op，arith 就被
  // 解析器顺手加载了，一切正常；等你喂一个不含 arith 的输入，才会撞上
  //   LLVM ERROR: arith.constant created with unregistered dialect
  // 而且它不是诊断，是直接 abort。
  let dependentDialects = ["::mlir::arith::ArithDialect"];

  // 第 11 章：fold 返回一个 Attribute（比如 pix.reduce 把常量图求和成 f32）时，
  // 框架需要把那个 Attribute 变回一条真的 op 才能替换掉原来的 op——这个"落地"
  // 动作就问方言的 materializeConstant。**不实现它，返回 Attribute 的 fold 会
  // 静悄悄地不起作用**：折是折出来了，但没人能把它放回 IR 里。
  let hasConstantMaterializer = 1;

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
