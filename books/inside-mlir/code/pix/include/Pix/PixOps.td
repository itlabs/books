//===- PixOps.td - pix 方言的 op ---------------------------*- tablegen -*-===//
//
// 《深入 MLIR》第 7–8 章。
//   第 7 章：pix.add —— 把"改 .td → 重新生成 → 编译 → pix-opt 能认"这条链路跑通。
//   第 8 章：补上 mul / scale / transpose / reduce / pipeline+yield，
//           它们各自演示 ODS 的一个方面（类型不一致、标量操作数、属性、region）。
//
//===----------------------------------------------------------------------===//

#ifndef PIX_OPS
#define PIX_OPS

include "Pix/PixDialect.td"
include "Pix/PixTypes.td"                          // !pix.kernel
include "Pix/PixAttrs.td"                          // #pix.border
include "mlir/Interfaces/ControlFlowInterfaces.td" // ReturnLike
include "mlir/Interfaces/InferTypeOpInterface.td"  // SameOperandsAndResultType
include "mlir/Interfaces/SideEffectInterfaces.td"  // Pure

def Pix_AddOp : Pix_Op<"add", [Pure, Commutative, SameOperandsAndResultType]> {
  let summary = "两张图逐点相加";
  let description = [{
    `pix.add` 把两张同样大小的图逐点相加。

    trait 的含义（第 5 章）：
      · `Pure`                      —— 不碰内存、结果只由操作数决定，于是 CSE/DCE 白拿
      · `Commutative`               —— 两个操作数可交换
      · `SameOperandsAndResultType` —— 操作数与结果类型必须一致，这条验证由 trait 自动生成

    例子：

    ```mlir
    %sum = pix.add %a, %b : tensor<4x4xf32>
    ```
  }];

  let arguments = (ins F32Tensor:$lhs, F32Tensor:$rhs);
  let results = (outs F32Tensor:$result);

  // 自定义格式：让 IR 读起来像 `pix.add %a, %b : tensor<4x4xf32>`
  // 而不是通用格式那一长串（第 3 章讲过两者的关系）。
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";

  // 第 11 章：这个 op 两条路都用到，正好对照。
  //   fold          —— 加一张全零图等于没加，返回**已有的值**
  //   canonicalizer —— add %x, %x → scale %x, 2.0，要**造新 op**，fold 做不到
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// 第 8 章新增：逐点乘
//===----------------------------------------------------------------------===//

def Pix_MulOp : Pix_Op<"mul", [Pure, Commutative, SameOperandsAndResultType]> {
  let summary = "两张图逐点相乘";
  let arguments = (ins F32Tensor:$lhs, F32Tensor:$rhs);
  let results = (outs F32Tensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// 第 8 章新增：整张图乘一个标量
//   看点：两个操作数**类型不同**（图是张量，系数是标量），所以不能再用
//   SameOperandsAndResultType；改用 AllTypesMatch 只约束"图与结果"这一对。
//===----------------------------------------------------------------------===//

def Pix_ScaleOp : Pix_Op<"scale", [Pure, AllTypesMatch<["image", "result"]>]> {
  let summary = "整张图乘一个标量系数";
  let arguments = (ins F32Tensor:$image, F32:$factor);
  let results = (outs F32Tensor:$result);
  // 只打印一个类型：结果类型与 image 相同，由 AllTypesMatch 保证，
  // 解析时能据此推出 image 的类型——少写一个类型，IR 更好读。
  let assemblyFormat = "$image `,` $factor attr-dict `:` type($result)";

  // 第 11 章：这个 op 同时要 fold 和 canonicalize，正好演示两者的分工。
  //   fold          —— 系数为 1 时，整个 op 就等于它的输入，返回**已有的值**
  //   canonicalizer —— 嵌套的两层 scale 合成一层，要造新 op
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// 第 8 章新增：转置
//   看点：结果类型与操作数类型**不同**（HxW → WxH），所以两个类型都得打印。
//   形状是否真的互为转置，要等第 10 章写 verifier 才检查。
//===----------------------------------------------------------------------===//

def Pix_TransposeOp : Pix_Op<"transpose", [Pure]> {
  let summary = "把图沿主对角线翻转";
  let arguments = (ins F32Tensor:$image);
  let results = (outs F32Tensor:$result);
  let assemblyFormat = "$image attr-dict `:` type($image) `to` type($result)";

  // 第 10 章：ODS 只知道两边都是 F32Tensor，"形状必须互为转置"要自己查。
  let hasVerifier = 1;

  // 第 11 章：transpose(transpose(%x)) → %x。返回的是**已有的值**，
  // 不造任何 op，所以 fold 就够了。
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// 第 8 章新增：全图归约
//   看点：结果是**标量**，不是图。
// 第 12 章：归约方式由枚举属性给出（sum/max/min），兑现第 8 章留的那句话。
//   kind 带默认值，所以老的 `pix.reduce %img : … -> f32` 写法仍然能解析——
//   给已有 op 加属性时这是个值得学的招（详见第 12 章）。
//===----------------------------------------------------------------------===//

def Pix_ReduceOp : Pix_Op<"reduce", [Pure]> {
  let summary = "把整张图归约成一个标量";
  let arguments = (ins F32Tensor:$image,
                       DefaultValuedAttr<Pix_ReduceKindAttr,
                                         "ReduceKind::Sum">:$kind);
  let results = (outs F32:$result);

  // kind 走 attr-dict 打印：默认值（sum）时整个字典是空的，IR 一个字都不多；
  // 写了 max/min 才会出现 {kind = #pix.reduce_kind<max>}。
  let assemblyFormat = "$image attr-dict `:` type($image) `->` type($result)";

  // 第 11 章：常量图可以在编译期直接算完。这是 fold 返回**Attribute** 的例子
  // ——真正意义上的"常量折叠"。第 12 章起它要按 kind 分三种算法。
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// 第 8 章新增：带 region 的容器 op
//   看点：ODS 里怎么声明 region（regions 字段）、怎么规定"单块 + 隐式终结符"，
//   以及配套的 terminator op 怎么用 HasParent 把自己钉在父 op 上。
//   注意它**没有** IsolatedFromAbove：region 内可以直接引用外面的值（第 2 章），
//   代价是不能被直接抽成独立函数——第 25 章做 GPU kernel outlining 时会需要那个 trait。
//===----------------------------------------------------------------------===//

def Pix_PipelineOp : Pix_Op<"pipeline", [
    SingleBlockImplicitTerminator<"YieldOp">,
    RecursiveMemoryEffects,
    // 第 10 章：声明"控制流怎么在我和我的 region 之间走"，以及"每条边的落点接收
    // 哪些值"。实现这两个接口方法，通用机制就会替我们检查"yield 出来的值与本 op 的
    // 结果是否个数、类型都对齐"——正是第 2 章 scf.if 那条报错的来源。
    // getSuccessorInputs 在接口里带默认实现（返回空），要覆盖就得显式列出来。
    DeclareOpInterfaceMethods<RegionBranchOpInterface,
                              ["getSuccessorInputs"]>]> {
  let summary = "把一串图像处理步骤打包成一个可整体变换的单元";
  let description = [{
    `pix.pipeline` 自己不算任何东西，它只是把一段处理流程围起来，
    好让变换（第三部分）能把它当一个整体来 tile、融合或搬到设备上。
    body 的最后一个 op 必须是 `pix.yield`，它交出这个 pipeline 的结果。

    例子：

    ```mlir
    %out = pix.pipeline : tensor<4x4xf32> {
      %t = pix.add %a, %b : tensor<4x4xf32>
      pix.yield %t : tensor<4x4xf32>
    }
    ```
  }];

  let results = (outs F32Tensor:$result);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict `:` type($result) $body";
}

def Pix_YieldOp : Pix_Op<"yield", [Pure, ReturnLike, Terminator,
                                   HasParent<"PipelineOp">]> {
  let summary = "交出 pix.pipeline 的结果";
  let description = [{
    `pix.yield` 是 `pix.pipeline` 的终结符。注意两个细节，它们都是被编译器逼出来的：

    1. 操作数是 `Variadic`，因为 `SingleBlockImplicitTerminator` 允许"省略不写"，
       那种情况下框架插进来的是一个**没有操作数**的 yield。
    2. 必须给一个**零参数的 builder**。隐式终结符机制要能用 `build(builder, state)`
       把它造出来；缺了这个重载，编译会在 OpDefinition.h 里报
       "no matching function for call to 'YieldOp::build(OpBuilder&, OperationState&)'"。
       上游 `scf.yield` 用的是同一招。
  }];

  let arguments = (ins Variadic<F32Tensor>:$values);
  let builders = [OpBuilder<(ins), [{ /* 什么都不用做：造一个空的 yield */ }]>];

  // 可选组：有操作数时打印 `pix.yield %t : tensor<…>`，没有时只打印 `pix.yield`。
  // `^` 标记的是这一组的"锚"——它有值，整组才打印。
  let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
}

//===----------------------------------------------------------------------===//
// 第 9 章新增：卷积核常量
//   看点：结果是**自定义类型**；权重是 DenseElementsAttr（编译期常量数据，第 4 章）。
//   ConstantLike 让它被当作常量对待——第 11 章讲折叠时这个 trait 很关键。
//===----------------------------------------------------------------------===//

def Pix_KernelOp : Pix_Op<"kernel", [Pure, ConstantLike]> {
  let summary = "由一组常量权重造出一个卷积核";
  let description = [{
    例子：

    ```mlir
    %k = pix.kernel dense<[[0.0, 1.0, 0.0],
                           [1.0, -4.0, 1.0],
                           [0.0, 1.0, 0.0]]> : tensor<3x3xf32> -> !pix.kernel<3 x 3>
    ```

    权重存在属性里（编译期已知），尺寸存在类型里。两者必须一致——
    这条检查第 10 章会用手写 verifier 补上。

    为什么要写两个类型？因为 `dense<…>` 这种 **ElementsAttr 必须配一个 shaped 类型**
    （tensor / vector），而 `!pix.kernel` 不是 shaped 类型。如果把格式写成
    `$weights attr-dict `:` type($result)`，解析器会拿 `!pix.kernel<3 x 3>` 去当那个
    elements 字面量的类型，于是报 `elements literal must be a shaped type`。
    所以这里让属性带上自己的 `tensor<3x3xf32>`，再用 `->` 指向结果类型。
  }];

  let arguments = (ins ElementsAttr:$weights);
  let results = (outs Pix_KernelType:$result);
  let assemblyFormat = "$weights attr-dict `->` type($result)";

  // 第 10 章：权重的形状必须与 !pix.kernel<R x C> 里的 R、C 一致。
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 第 9 章新增：邻域（stencil）运算
//   看点：自定义类型当操作数；枚举属性带默认值；可选组让默认值不必打印出来。
//===----------------------------------------------------------------------===//

def Pix_ConvolveOp : Pix_Op<"convolve", [Pure, AllTypesMatch<["image", "result"]>]> {
  let summary = "用一个卷积核对图做邻域运算";
  let description = [{
    输出与输入同样大小（边界按 `border` 指定的方式处理），所以用
    `AllTypesMatch<["image", "result"]>` 约束两者类型相同。

    例子：

    ```mlir
    %out = pix.convolve %img, %k border<zero> : tensor<8x8xf32>
    %dfl = pix.convolve %img, %k : tensor<8x8xf32>   // border 默认 clamp，不打印
    ```
  }];

  let arguments = (ins F32Tensor:$image,
                       Pix_KernelType:$kernel,
                       DefaultValuedAttr<Pix_BorderModeAttr,
                                         "::mlir::pix::BorderMode::Clamp">:$border);
  let results = (outs F32Tensor:$result);

  // 两件事：
  //   1. 可选组：border 取默认值时整组不打印，IR 更干净（`^` 是这一组的锚）。
  //   2. **kernel 的类型必须打印出来**。它是带参数的自定义类型（!pix.kernel<3x3>），
  //      解析器没法凭空造一个出来；漏了它，TableGen 会直接拒绝这份 ODS：
  //        error: type of operand #1, named 'kernel', is not buildable
  //                and a buildable type cannot be inferred
  //      而 image/result 的类型只打一个就够，因为 AllTypesMatch 能把另一个推出来。
  let assemblyFormat = [{
    $image `,` $kernel (`border` $border^)? attr-dict
    `:` type($kernel) `,` type($result)
  }];
}

#endif // PIX_OPS
