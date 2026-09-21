//===- Passes.td - pix 自己的 pass -------------------------*- tablegen -*-===//
//
// 《深入 MLIR》第 13 章。这里的三个 pass 不是为了做什么高深的优化，
// 而是各自演示 Pass 框架的一个机制：
//   pix-cost-report    —— OperationPass<func::FuncOp> + 用分析 + 精确声明保留
//   pix-cost-recheck   —— getCachedAnalysis：分析到底有没有被复用
//   pix-hoist-kernels  —— 真的改 IR 的 pass，用来看"改了之后分析就没了"
//
//===----------------------------------------------------------------------===//

#ifndef PIX_TRANSFORMS_PASSES
#define PIX_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def PixCostReport : Pass<"pix-cost-report", "::mlir::func::FuncOp"> {
  let summary = "统计本函数的 pix 计算量，写成属性";
  let description = [{
    把 `PixCostAnalysis` 的结果写到函数上：

    ```mlir
    func.func @f(...) attributes {pix.num_ops = 3 : i64, pix.elements = 48 : i64}
    ```

    第二个参数 `::mlir::func::FuncOp` 决定了这是个**函数级** pass：
    框架只会把 `func.func` 交给它，而且**不同函数之间可以并发跑**。
    想处理整个模块就写 `"::mlir::ModuleOp"`，什么都不写则是"任意 op"。

    关键一行在实现里：跑完调用 `markAnalysesPreserved<PixCostAnalysis>()`。
    我们只加了一个可丢弃属性，op 的构成一点没变，所以那份分析仍然有效——
    这件事只有我们自己知道，必须显式告诉框架，否则它会保守地全部作废。
  }];
}

def PixCostRecheck : Pass<"pix-cost-recheck", "::mlir::func::FuncOp"> {
  let summary = "看上一个 pass 算的分析还在不在缓存里";
  let description = [{
    用 `getCachedAnalysis`（而不是 `getAnalysis`）查询，把结果写成
    `pix.cost_cached = true/false`。这个 pass 唯一的用途是让"分析缓存"
    这件看不见的事变得可观察——第 13 章拿它做三组对照实验。
  }];
}

def PixHoistKernels : Pass<"pix-hoist-kernels", "::mlir::func::FuncOp"> {
  let summary = "把 pix.kernel 提到函数开头";
  let description = [{
    `pix.kernel` 是 `ConstantLike` 的常量，散落在函数各处不利于阅读，
    也不利于后面的 CSE。这个 pass 把它们统一挪到函数入口块的最前面。

    它是这一章唯一**真的改 IR** 的 pass，用来演示两件事：改动之后分析
    默认会被作废；以及"没改动时要记得 `markAllAnalysesPreserved()`"
    ——漏了它，一个什么都没做的 pass 也会把后面所有分析的缓存冲掉。
  }];
}

def PixRewriteAdd : Pass<"pix-rewrite-add", "::mlir::func::FuncOp"> {
  let summary = "第 14 章：两条都能匹配 pix.add 的 pattern，靠 benefit 决出胜者";
  let description = [{
    两条 pattern 都匹配 `pix.add %x, %x`，但改写成不同的东西：

    - `AddSelfToScale` → `pix.scale %x, 2.0`
    - `AddSelfToMul`   → `pix.mul %x, <全 2 的图>`

    `benefit` 高的先试，成功了后面的就不再试。`swap` 选项把两条的 benefit
    对调，于是同一份输入会得到两种不同的输出——benefit 的作用一眼可见。

    这个 pass 还装了一个 `RewriterBase::Listener`，把 driver 发出的通知
    （插入 / 替换 / 擦除 / 原地修改）计数后写成函数属性，让"通知机制"
    这件平时看不见的事变得可观察、可测试。
  }];
  let options = [
    Option<"swap", "swap", "bool", /*default=*/"false",
           "把两条 pattern 的 benefit 对调">
  ];
}

def PixDRR : Pass<"pix-drr", "::mlir::func::FuncOp"> {
  let summary = "第 15 章：跑一套用 DRR（TableGen）声明出来的 pattern";
  let description = [{
    这个 pass 一条 C++ pattern 都没写，全部来自 `PixPatterns.td` 里的
    `Pat<>`，由 `mlir-tblgen -gen-rewriters` 生成，用
    `populateWithGenerated(patterns)` 一句装进来。

    和第 14 章的 `pix-rewrite-add` 对照着看：同样的改写，一个写在 C++ 里、
    一个写在 TableGen 里，各自的舒服与难受都在第 15 章正文里摊开。
  }];
}

def PixToArith : Pass<"pix-to-arith", "::mlir::ModuleOp"> {
  let summary = "第 16 章：把能下降的 pix op 换成 arith/tensor（部分转换）";
  let description = [{
    Dialect Conversion 的四个部件在这一个 pass 里都能看到：

    - `ConversionTarget` —— 声明什么合法。arith/tensor/func 合法；
      `pix.add` / `pix.mul` / `pix.scale` 非法（必须换掉）；
      `pix.reduce` / `pix.convolve` / `pix.pipeline` **仍然合法**，因为这一章
      还没有它们的下降路径——这正是"部分转换"的意义。
    - `TypeConverter` —— 大部分类型原样保留（必须显式写这条身份规则！），
      外加 `!pix.kernel<R x C>` → `tensor<RxCxf32>`。
    - `ConversionPattern` —— 和普通 RewritePattern 差在多一个 adaptor 参数，
      给的是**已经转换过的操作数**。
    - driver —— `applyPartialConversion`；`full=true` 时换成
      `applyFullConversion`，于是 pix.reduce 这种没人管的 op 会让整个转换失败，
      报错信息正是第 17 章要教你读的那种。
  }];
  // 第 16 章的坑：这个 pass 会**造** arith 和 tensor 的 op，必须声明出来。
  // 注意和第 11 章那次的区别：那次是方言的 canonicalizer 在造 op，声明写在
  // Dialect 上；这次是 pass 在造，就写在 Pass 上（PassBase.td 里的注释很准确：
  // "A list of dialects this pass may produce entities in"）。
  // 漏了它同样是 LLVM ERROR + abort，不是诊断。
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::tensor::TensorDialect"
  ];
  let options = [
    Option<"full", "full", "bool", /*default=*/"false",
           "用 applyFullConversion 代替 applyPartialConversion">,
    Option<"lowerKernels", "lower-kernels", "bool", /*default=*/"false",
           "顺带把 pix.kernel 降成 tensor 常量（类型会变，于是需要物化）">,
    Option<"noMaterialize", "no-materialize", "bool", /*default=*/"false",
           "故意不装物化钩子，复现第 16 章末尾那次 unresolved materialization 失败">,
    Option<"convertSignatures", "convert-signatures", "bool", /*default=*/"false",
           "顺带转换函数签名里的 !pix.kernel（第 17 章）">,
    Option<"oneToN", "one-to-n", "bool", /*default=*/"false",
           "把 !pix.kernel 展成两个值（权重 + 半径），演示 1:N 类型转换">
  ];
}

def PixAnalyze : Pass<"pix-analyze", "::mlir::func::FuncOp"> {
  let summary = "第 19 章：用 MLIR 自带的三样分析，结果写成属性";
  let description = [{
    第 13 章那个分析是我们自己遍历数出来的。这个 pass 用现成的：

    - `DominanceInfo` —— 有多少 pix op 支配着所有其他 pix op
    - 副作用查询 —— `isPure` / `isMemoryEffectFree` / `wouldOpBeTriviallyDead`
      三者各计一个数，因为它们问的是**三件不同的事**
    - `DataFlowSolver` + 一个真的稀疏前向分析 —— 有多少个值"已知是全零图"

    结果全写成函数属性，好用 FileCheck 断言。
  }];
  let dependentDialects = ["::mlir::arith::ArithDialect"];
}

#endif // PIX_TRANSFORMS_PASSES
