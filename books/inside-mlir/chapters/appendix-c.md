# 附录 C 源码导览地图：想看某个机制去 `mlir/` 的哪个目录

## 这份附录怎么用

<div class="goal-box">
<p>这也是一份<strong>查的</strong>附录。用法是：你想搞清楚某个机制到底怎么工作的（"greedy 驱动器凭什么决定下一个处理谁"、"ODS 那些 <code>let</code> 到底生成了什么代码"），到这里找路径，然后直接读源码。</p>
<p>为什么强调读源码：这本书写下来最大的一条体会是，<strong>MLIR 的文档说清了"怎么用"，但"为什么会是这个行为"基本只在源码和注释里。</strong>书里那些坑（第 28 章那张表）有一半是靠读 <code>lib/</code> 里的实现和注释定位的，不是靠文档。</p>
<p>所有路径都在 LLVM <strong>23.1.2</strong> 上核对过。目录结构在大版本之间会动，但下面这个<strong>分层规律</strong>很稳定，规律比路径更值得记。</p>
</div>

## 一、先记住三条规律，比记路径有用

MLIR 的目录布局有强规律，摸清这三条，大部分东西不用查也能猜到。

**规律一：`include/` 放声明和 `.td`，`lib/` 放实现，两边目录名一一对应。**

```
mlir/include/mlir/XXX/    <-> mlir/lib/XXX/
```

想找 `DialectConversion` 的实现？声明在 `include/mlir/Transforms/DialectConversion.h`，实现就在 `lib/Transforms/Utils/DialectConversion.cpp`。（注意 `Transforms` 下的实现多在 `Utils/` 子目录里——那里放的是"给别人调用的工具"，`lib/Transforms/` 根目录下放的是"成品 pass"。这个区分很有用：`CSE.cpp` 两边都有，一个是可复用的实现，一个是包成 pass 的壳。）

**规律二：一个方言的目录结构是固定的四件套。**

```
mlir/include/mlir/Dialect/<名字>/
    IR/          <- 方言本体：.td 定义、类型、属性、op
    Transforms/  <- 这个方言自己的 pass
    Utils/       <- 给别人用的辅助函数
    TransformOps/  <- 它给 transform 方言贡献的 op（第 18 章）
```

不是每个方言四件套都全（`TransformOps/` 只有 16 个方言有）。另外 pass 声明的位置有一个要留意的例外：**通常在 `<方言>/Transforms/Passes.td`（28 个方言是这样），但 `Linalg` 和 `Async` 放在 `<方言>/Passes.td`。** 找不到就往上一级看。

看到一个陌生方言，直接进 `IR/` 找那个 `.td`，`let summary` 和 `let description` 通常是最好的入口——第 26、27 章那些设计动机全是从这里读来的。

**规律三：跨方言的转换在 `Conversion/`，不在任何一个方言下面。**

```
mlir/lib/Conversion/<源>To<目标>/
```

第 24 章那条管线上每一步都能这么找到：`SCFToControlFlow/`、`ArithToLLVM/`、`MemRefToLLVM/`、`FuncToLLVM/`。命名格式就是 `XToY`，**这是找下降实现最快的方式**——不用猜是放在源方言还是目标方言下面，答案是两者都不是。

## 二、核心基础设施：`include/mlir/IR/`

这是全书最该熟的一个目录。看到文件名就知道该读哪个：

| 你想知道 | 文件 | 书里讲在 |
| --- | --- | --- |
| op 到底是什么数据结构 | `Operation.h`、`OperationSupport.h` | 第 3 章 |
| 值、use-def 链怎么组织 | `Value.h`、`UseDefLists.h`、`ValueRange.h` | 第 3 章 |
| Block 与 Region | `Block.h`、`Region.h`、`RegionKindInterface.h` | 第 4 章 |
| 属性与类型的存储、唯一化 | `Attributes.h`、`Types.h`、`StorageUniquerSupport.h` | 第 5 章 |
| 内建类型/属性有哪些 | `BuiltinTypes.td`、`BuiltinAttributes.td` | 第 5 章 |
| **ODS 的全部可用写法** | **`OpBase.td`** | 第 9 章 |
| trait 清单 | `OpDefinition.h`、`Traits.td` | 第 10 章 |
| `fold` / `OpFoldResult` 的定义 | `OpDefinition.h` | 第 11 章 |
| pattern 的基类与 rewriter | `PatternMatch.h` | 第 11、14 章 |
| `m_Constant` 之类的匹配器 | `Matchers.h` | 第 11 章 |
| 支配关系 | `Dominance.h` | 第 19 章 |
| 方言注册、`dependentDialects` 的机制 | `Dialect.h`、`DialectRegistry.h` | 第 11、16 章 |
| **`ExternalModel` / `attachInterface`** | `Interfaces.td`（配 `mlir/docs/Interfaces.md`） | 第 28 章 |
| 报错、`emitError` 那一套 | `Diagnostics.h` | 第 12 章 |
| `loc(...)` 是什么 | `Location.h`、`BuiltinLocationAttributes.td` | 第 12 章 |
| verifier 什么时候跑 | `Verifier.h` | 第 11 章 |

`OpBase.td` 那一行我加粗了，因为它是**整本第 9 章最有价值的参考**。ODS 文档写了常用的那些，但 `OpBase.td` 是唯一的完整清单——所有 `*Attr` 约束、所有 trait、所有 `Op` 可用的 `let`。写 `.td` 卡住的时候在这个文件里 `grep`，比搜文档快。

## 三、按机制查：从书里的章节回到源码

下面这张表是这份附录的主体。左边是书里讲过的机制，右边是去读哪个文件。

### 变换与 pattern

| 机制 | 路径 | 章 |
| --- | --- | --- |
| **greedy 驱动器**（`maxIterations`、`maxNumRewrites`、worklist 顺序） | `lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp` | 14 |
| 另一种驱动器：只走一遍，不到不动点 | `lib/Transforms/Utils/WalkPatternRewriteDriver.cpp` | 14 |
| **Dialect Conversion**（四态合法性、`TypeConverter`、materialization） | `lib/Transforms/Utils/DialectConversion.cpp` | 17 |
| 常量折叠的实现、`materializeConstant` 怎么被调用 | `lib/Transforms/Utils/FoldUtils.cpp` | 11 |
| canonicalize pass 本身 | `lib/Transforms/Canonicalizer.cpp` | 11 |
| CSE 怎么判断两个 op 等价 | `lib/Transforms/CSE.cpp` | 19 |
| 死代码消除的判据 | `lib/Transforms/TrivialDeadCodeElimination.cpp`、`RemoveDeadValues.cpp` | 19 |
| 循环不变量提升 | `lib/Transforms/Utils/LoopInvariantCodeMotionUtils.cpp` | 19 |
| 内联器 | `lib/Transforms/Utils/Inliner.cpp`、`InliningUtils.cpp` | 19 |
| `cf` 反推回 `scf`（**下降不总是单向的**） | `lib/Transforms/Utils/CFGToSCF.cpp` | 24 |

`GreedyPatternRewriteDriver.cpp` 值得专门读一遍。第 14 章那个"互逆 pattern 会挂住而不是报不收敛"的结论，就是从这个文件里 `processWorklist` 的注释确认的——文档没说这件事。

### Pass 与分析

| 机制 | 路径 | 章 |
| --- | --- | --- |
| `Pass`/`OperationPass` 的定义 | `include/mlir/Pass/Pass.h` | 13 |
| **写 pass 的 `.td` 有哪些字段** | `include/mlir/Pass/PassBase.td` | 13 |
| 嵌套 PassManager、并行怎么调度 | `include/mlir/Pass/PassManager.h`、`lib/Pass/Pass.cpp` | 13 |
| **分析缓存与失效**（`markAnalysesPreserved`） | `include/mlir/Pass/AnalysisManager.h` | 13 |
| `--mlir-timing` 的实现 | `lib/Pass/PassTiming.cpp` | 12 |
| `--mlir-print-ir-after-all` 的实现 | `lib/Pass/IRPrinting.cpp` | 12 |
| 崩溃时自动缩减复现 | `lib/Pass/PassCrashRecovery.cpp` | 12 |
| pass 选项怎么解析（那些带引号的参数） | `include/mlir/Pass/PassOptions.h` | 16 |
| 副作用查询（`isPure`/`isMemoryEffectFree`/`wouldOpBeTriviallyDead`） | `include/mlir/Interfaces/SideEffectInterfaces.h` | 19 |
| **数据流框架**（solver、lattice、传播） | `include/mlir/Analysis/DataFlowFramework.h` | 19 |
| 稀疏/稠密分析基类 | `include/mlir/Analysis/DataFlow/SparseAnalysis.h`、`DenseAnalysis.h` | 19 |
| 可达性分析（几乎所有稀疏分析的前提） | `include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h` | 19 |
| 一个现成的完整分析，照着抄最省事 | `include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h` | 19 |
| 别名分析 | `include/mlir/Analysis/AliasAnalysis.h` | 19 |
| 活跃性 | `include/mlir/Analysis/Liveness.h` | 19 |
| 切片（找一个值的前向/后向依赖） | `include/mlir/Analysis/SliceAnalysis.h` | 19 |

`AnalysisManager.h` 要读，因为第 13 章那个区分——`markAnalysesPreserved<T>()` 只保住 `T`，`markAllAnalysesPreserved()` 保住全部——写错了不会报错，只会让后面的 pass 拿到过期结果。这个语义只在这个头文件的注释里说清楚。

### 接口（第 10 章那句话的实现）

第 10 章讲 trait 和 interface 的区别，第 28 章看到它在生产规模上兑现。想知道某个"通用变换"凭什么能作用于你的 op，答案都在 `include/mlir/Interfaces/`：

| 接口 | 它让你白拿什么 | 章 |
| --- | --- | --- |
| `SideEffectInterfaces.h` | CSE、DCE、LICM 全都认它 | 19 |
| **`TilingInterface.h`** | tile 和融合（第 21 章那整章） | 21 |
| `DestinationStyleOpInterface.h` | destination-passing style，bufferization 靠它 | 20、22 |
| `IndexingMapOpInterface.h` | indexing map 的通用查询 | 20 |
| `LoopLikeInterface.h` | 循环相关的通用变换（LICM 等） | 19 |
| `ControlFlowInterfaces.h` | `RegionBranchOpInterface` 在这里——第 11 章踩的那个坑 | 11 |
| `CallInterfaces.h` | 内联器认它 | 19 |
| `InferTypeOpInterface.h` | 结果类型自动推导 | 9 |
| `ViewLikeInterface.h`、`SubsetOpInterface.h` | `extract_slice` 那类 op 的通用处理 | 21、22 |
| `VectorInterfaces.h` | 向量化相关 | 23 |
| `ParallelCombiningOpInterface.h` | `scf.forall` 的 `parallel_insert_slice` 靠它 | 21 |
| `ValueBoundsOpInterface.h` | 推导"这个下标不会越界"——`in_bounds` 的来源 | 23 |
| `DataLayoutInterfaces.h` | 目标的数据布局（对齐、指针宽度） | 24 |

`ValueBoundsOpInterface.h` 是个容易被忽略但很有用的：第 23 章 `transfer_read` 上那个 `in_bounds` 属性能不能设成 `true`，判断逻辑就走这套。

### 方言：44 个，怎么挑

上游 `include/mlir/Dialect/` 下有 46 个目录，其中 `Utils/` 和 `OpenACCMPCommon/` 不是独立方言，所以实际是 **44 个方言**。按"什么时候需要看它"分组：

**几乎一定会用到的**

| 方言 | 管什么 | 章 |
| --- | --- | --- |
| `Func` | 函数与调用 | 全书 |
| `Arith` | 标量算术、`arith.constant` | 16 |
| `Tensor` | 值语义张量、`extract_slice`/`pad`/`empty` | 20、21 |
| `MemRef` | 内存语义缓冲 | 22、24 |
| `SCF` | 结构化控制流（`for`/`if`/`forall`） | 21、24 |
| `ControlFlow` | 基本块与分支（`cf`） | 24 |
| `LLVMIR` | 最后那一层 | 24 |
| `Linalg` | 迭代空间显式化 | 20、21 |
| `Vector` | 向量化、mask | 23 |
| `Bufferization` | 值语义 → 内存语义 | 22 |
| `Transform` | 把调度写成 IR | 18 |

**做 GPU 才看的**：`GPU`、`NVVM`、`ROCDL`（在 `LLVMIR/` 下）、`SPIRV`、`AMDGPU`、`NVGPU`、`XeGPU`（第 25 章）。

**做 CPU 向量化才看的**：`ArmNeon`、`ArmSVE`、`ArmSME`、`X86`（第 23 章）。

**写 pattern 才看的**：`PDL`、`PDLInterp`（第 15 章）。

**特定领域**：`Tosa`（框架前端）、`SparseTensor`、`Quant`、`Complex`、`Math`、`Shape`、`Async`、`EmitC`（生成 C 代码）、`OpenMP`/`OpenACC`（第 27、28 章那条共享线）、`MPI`、`Shard`（分布式）、`SMT`、`WasmSSA`、`IRDL`（在 IR 里定义方言）。

**读方言源码的入口固定是这三个文件**：

```
<方言>/IR/<方言>Base.td 或 <方言>Dialect.td   <- let summary / let description，设计动机
<方言>/IR/<方言>Ops.td                        <- op 清单
<方言>/Transforms/Passes.td                   <- 它提供哪些 pass（Linalg/Async 在上一级）
```

第 26、27、28 章所有对 Triton/IREE/flang/ClangIR 的判断，都是从这三类文件读出来的。**`let description` 是被严重低估的资源**——很多方言的作者把设计取舍写在了里面（第 27 章引用的 `stream` 那段、ClangIR `cir` 那段关于 `hasConstantMaterializer` 的注释，都是这么来的）。

### 转换：`lib/Conversion/`

按 `XToY` 命名，第 24、25 章那两条管线上每一步都在这里：

```
lib/Conversion/
    SCFToControlFlow/      <- 结构化控制流压平（不可逆的那一步）
    ArithToLLVM/
    MemRefToLLVM/          <- memref 描述符怎么变成 { ptr, ptr, i64, [N], [N] }
    FuncToLLVM/            <- 函数签名与调用约定
    ControlFlowToLLVM/     <- 第 24 章那个 "Dialect 'cf' not found" 的解药
    VectorToLLVM/
    GPUToNVVM/ GPUToROCDL/ GPUToSPIRV/   <- 第 25 章三个出口
    TosaToLinalg/          <- 框架前端进来的入口
```

`MemRefToLLVM/` 特别值得读：第 24 章那个描述符结构（`{ ptr, ptr, i64, [N x i64], [N x i64] }`）不是文档里的示意图，它就是这里的代码定的。想知道为什么 JIT 的调用约定长那样，答案在这个目录。

### 落地与执行

| 想知道 | 路径 | 章 |
| --- | --- | --- |
| MLIR 的 `llvm` 方言怎么变成真的 LLVM IR | `lib/Target/LLVMIR/` | 24 |
| `registerAllToLLVMIRTranslations` 到底注册了什么 | `include/mlir/Target/LLVMIR/Dialect/` | 25 |
| JIT 怎么起来的、`mlir_runner_utils` 是什么 | `lib/ExecutionEngine/` | 24 |
| 生成 SPIR-V 二进制 | `lib/Target/SPIRV/` | 25 |

### 工具本身

`mlir/tools/` 下面这些，有几个值得知道存在：

| 工具 | 用途 | 章 |
| --- | --- | --- |
| `mlir-opt` | 主力 | 全书 |
| `mlir-translate` | IR ↔ 外部格式（出 LLVM IR 用它） | 24 |
| `mlir-runner` | JIT 执行 | 24 |
| **`mlir-tblgen`** | `.td` → C++，见下 | 9 |
| `mlir-pdll` | PDLL 编译器 | 15 |
| `mlir-reduce` | 把一个崩溃的用例自动缩到最小 | 12 |
| `mlir-query` | 交互式查询 IR（像 clang-query） | — |
| `mlir-rewrite` | 批量改写 IR 文本 | — |
| `mlir-lsp-server` / `tblgen-lsp-server` | 编辑器里补全 `.mlir` 和 `.td` | 附录 A |

`mlir-query` 和 `mlir-rewrite` 这两个书里没用上，但值得知道：调查一个大模块"哪些 op 满足某个条件"时比写 pass 快得多。

## 四、想知道 ODS 到底生成了什么

这是第 9 到 11 章最容易卡住的地方：你在 `.td` 里写了几行 `let`，生成的 C++ 长什么样？

**第一招，直接看生成的代码。** 构建目录里就有：

```
<build>/tools/mlir/include/mlir/Dialect/<方言>/IR/*.h.inc
<你的项目 build>/include/Pix/PixOps.h.inc      <- 我们自己的
```

`.h.inc`/`.cpp.inc` 是纯文本，可读。想知道 `hasFolder = 1` 生成了什么声明，`grep fold` 就看到了。

**第二招，自己跑一遍 `mlir-tblgen`。** 想看某个 `.td` 生成什么，不必构建整个项目：

```
mlir-tblgen --gen-op-decls -I <mlir的include> your.td
mlir-tblgen --gen-op-defs  -I <mlir的include> your.td
```

**第三招，读生成器本身。** 这是最硬但最彻底的办法，`mlir/tools/mlir-tblgen/` 下按功能分得很清楚：

| 你想知道 | 生成器 |
| --- | --- |
| op 的 C++ 类是怎么拼出来的（accessor、builder、verifier） | `OpDefinitionsGen.cpp` |
| **自定义 assembly（`assemblyFormat`）的全部语法** | `OpFormatGen.cpp` |
| 类型/属性的自定义格式 | `AttrTypeFormatGen.cpp` |
| 接口生成的那些 `Adaptor`、`Model` 类 | `OpInterfacesGen.cpp` |
| 类型与属性的类怎么生成 | `AttrOrTypeDefGen.cpp` |
| **DRR（`Pat<>`）到底变成了什么 C++** | `RewriterGen.cpp` |
| pass 的样板（`impl::XxxBase`） | `PassGen.cpp` |
| 枚举属性 | `EnumsGen.cpp` |

`OpFormatGen.cpp` 那一行值得记：`assemblyFormat` 的可用指令（`$operands`、`attr-dict`、`type($result)`、`custom<...>`……）散落在文档各处，但这个文件里有完整的解析逻辑。第 9 章写 `pix.convolve` 那个格式时我就是在这里查的。

`RewriterGen.cpp` 对应第 15 章：DRR 报的那些错（"argument number mismatch"、"referencing unbound symbol"）都是这个文件发出来的，`grep` 报错原文就能直接定位到判断逻辑，比猜快得多。

## 五、文档：哪几篇值得完整读

`mlir/docs/` 下有二十多篇。大部分是查的，但有几篇**值得完整读一遍**，因为它们讲的是"为什么"而不只是"怎么用"：

| 文档 | 为什么值得读 | 对应章 |
| --- | --- | --- |
| `LangRef.md` | MLIR 的语言规范。读完前 8 章之后回头看这篇，很多事会对上 | 1–8 |
| **`Canonicalization.md`** | `fold` 和 canonicalize pattern 的分工准则，写得很清楚 | 11 |
| **`DialectConversion.md`** | 四态合法性、materialization 的完整语义。第 17 章那些坑基本都在这篇里有交代 | 17 |
| `PatternRewriter.md` | pattern 与 rewriter 的约定（什么时候必须用 rewriter 而不能直接改 IR） | 11、14 |
| **`Interfaces.md`** | 包括第 28 章补讲的 `ExternalModel`、`attachInterface`、`declarePromisedInterface` | 10、28 |
| `PassManagement.md` | 嵌套、并行、分析失效的完整规则 | 13 |
| `Bufferization.md` | One-Shot Bufferize 的 in-place 分析怎么工作 | 22 |
| `DeclarativeRewrites.md` | DRR 的完整语法 | 15 |
| `PDLL.md` | PDLL 语法 | 15 |
| `TargetLLVMIR.md` | memref 描述符、调用约定的规范说明 | 24 |
| `Diagnostics.md` | 怎么写出好的报错（自己造方言时用） | 12 |
| `Passes.md` | 上游所有 pass 的清单，`--help` 的可读版 | 全书 |

加粗那三篇是投入产出比最高的。如果时间只够读三篇文档，读 `Canonicalization.md`、`DialectConversion.md`、`Interfaces.md`。

## 六、最后一条建议：怎么读一个陌生方言

这本书第五部分（Triton、IREE、flang、ClangIR）用的就是下面这个固定流程，四个项目都是这么拆开的。放在这里当模板：

1. **先数方言，再数层。** `ls` 那个项目的方言目录，然后判断哪几个是真正的"层"、哪几个是横切基础设施或上游扩展。第 27 章数 IREE 时，九个目录里只有四个是分层决策——**直接数目录会高估复杂度**。

2. **读每个方言的 `let summary` 和 `let description`。** 这一步的信息密度最高。作者经常把设计动机、和相邻层的分工、甚至架构图（`stream` 那个 ASCII 图）写在这里。

3. **列 pass 清单，分成"下降"和"层内优化"两类。** 层内优化的数量就是这一层的价值——第 27 章数 HLFIR 的 12 个 pass，8 个是层内优化，那 8 个就是加这一层的全部理由。

4. **找那个"压平"的 pass。** 几乎每个项目都有一步"结构化控制流 → 基本块"，找到它就知道这个项目的"不可逆分界线"在哪。上游是 `SCFToControlFlow`，ClangIR 是 `cir-flatten-cfg`。

5. **看默认值。** 第 28 章那条教训：`cir` 在树里，但 `UseClangIRPipeline(false)`。**"在不在树里"说明不了成熟度，默认值才说明。** 找项目的 options 结构体或者驱动层的默认参数。

6. **找它的 `MissingFeatures` 等价物。** 不一定叫这个名字，但成熟项目通常有——一份明码标价的缺口清单。没有这个的项目，缺口不是不存在，只是没人写下来。

7. **最后才读实现。** 前六步下来你已经知道该读哪个文件了。反过来（一上来就读 `.cpp`）会淹死。

这个流程对你自己的项目也成立——反过来用，就是第 27 章那四个问题的落地检查表。
