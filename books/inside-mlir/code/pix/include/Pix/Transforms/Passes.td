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

#endif // PIX_TRANSFORMS_PASSES
