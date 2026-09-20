//===- PixTypes.td - pix 方言的自定义类型 ------------------*- tablegen -*-===//
//
// 《深入 MLIR》第 9 章。这里只有一个类型：!pix.kernel<RxC>，stencil 的权重矩阵。
//
// 为什么图像本身不做成自定义类型、而直接用内建的 tensor？因为第四部分要把 pix
// 降到 linalg，而 linalg 就在 tensor 上工作。图像若是自定义类型，中间就要插一整套
// 毫无教学价值的类型转换，还会挡住 SameOperandsAndResultType 这类现成 trait。
// 而"卷积核"确实是一个独立概念（尺寸受限、语义不同），值得单独成类型。
//
//===----------------------------------------------------------------------===//

#ifndef PIX_TYPES
#define PIX_TYPES

include "mlir/IR/AttrTypeBase.td"
include "Pix/PixDialect.td"

class Pix_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Pix_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Pix_KernelType : Pix_Type<"Kernel", "kernel"> {
  let summary = "stencil 权重矩阵的类型";
  let description = [{
    `!pix.kernel<RxC>` 表示一个 R 行 C 列的卷积核。**尺寸在类型里**，
    于是"这个卷积核多大"在编译期就是已知的——第 21 章算 halo（边界扩边）时
    要靠它，而如果尺寸只存在运行时的数据里，那一步就只能保守处理。

    例子：`!pix.kernel<3 x 3>`、`!pix.kernel<1 x 5>`

    **注意 `x` 两侧的空格**：MLIR 的词法分析器会把 `3x3` 里的 `x3` 当成一个标识符，
    于是字面量 `x` 匹配不上，报 `expected 'x'`。内建的 `tensor<4x4xf32>` 之所以不用空格，
    是因为它有一套专门的 dimension-list 解析器；自定义类型想要那种紧凑写法，
    得自己写 parser/printer（`let hasCustomAssemblyFormat = 1`）。
    上游 `!spirv.array<4 x f32>` 用的就是这里的空格写法。
  }];

  // 参数就是类型的"内容"。类型是唯一化的不可变值（第 4 章），
  // 所以这些参数一旦定下就不能改——要改只能造一个新类型。
  let parameters = (ins "int64_t":$rows, "int64_t":$cols);

  // 自定义格式：`<` 3 `x` 3 `>`。写法与 op 的 assemblyFormat 是同一套小语言。
  // x 两侧必须留空格，原因见上面 description 里那段。
  let assemblyFormat = "`<` $rows `x` $cols `>`";

  // 第 10 章：类型参数也要验。奇数边长是 stencil 的常识（有明确的中心点），
  // 但没人会自动帮你查——打开这个开关后要在 .cpp 里实现 verify()。
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// 卷积核一共有多少个权重。
    int64_t getNumWeights() const { return getRows() * getCols(); }
    /// 是不是方形核。
    bool isSquare() const { return getRows() == getCols(); }
  }];
}

#endif // PIX_TYPES
