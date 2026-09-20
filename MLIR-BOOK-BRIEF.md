# MLIR 书 · 规划与章节作者简报（写作时必读）

本文件是《深入 MLIR》（`books/inside-mlir/`，slug `inside-mlir`）的总体规划 + 写章节时的作者简报。
定位：本仓库第五本书，独立成册，风格对标《深入 LLVM 与 SYCL》（`books/inside-llvm-sycl/`）。

## 1. 定位与读者

- **读者**：有经验的 C++ / 编译器 / 系统工程师。懂 SSA、支配关系、Pass、指令选择；**不需要**再解释什么是 AST、什么是寄存器分配。MLIR 零基础或只跑过 Toy 教程。
- **痛点**（全书要解决的三件事，每章都要对得上其中之一）：
  1. Toy 教程只演示 API 序列，不讲**为什么 MLIR 长这样**、每个设计在解决什么问题。
  2. 真实项目（Triton / IREE / flang / ClangIR）几十个方言、上百个 pass，不知从哪拉线头。
  3. 最难的几块——**Interface、Dialect Conversion、bufferization、Transform 方言**——文档最薄、坑最深。
- **规格**：28 章 + 3 附录，五部分。每章正文约 250–450 行 markdown，与 `inside-llvm-sycl` 同量级。
- **源码锚点**：上游 `llvm/llvm-project` 的 `mlir/` 子树（不是 `intel/llvm`）。真实项目章另锚 `triton-lang/triton`、`iree-org/iree`、`llvm-project/flang`、`llvm/clangir`。

## 2. 动手前必做

1. **通读 `books/inside-llvm-sycl/chapters/ch01.md`、`ch04.md`、`ch25.md`** —— 声音、结构、图示、「源码为锚」写法的黄金样板。ch25（ClangIR）尤其接近本书要的调子：讲一个正在演进的 MLIR 层。
2. 读 `STYLE.md` 了解排版约定（**但语气不适用**：那是青少年书的语气，本书见 §4）。
3. 读 `books/inside-mlir/book.json` 确认章节顺序，交叉引用时（「第 N 章会讲…」）必须准确。
4. 读本文件 §5 的 **pix 方言规格**——从第 7 章起每章都在同一个项目上增量推进，不能各写各的。
5. **手边备一份可 grep 的源码树**。批次 1 是这样拉的（只取 mlir 子树，约 134 MB，与基线同分支）：
   ```bash
   git clone --filter=blob:none --sparse --depth=1 --branch release/23.x \
     https://github.com/llvm/llvm-project.git ~/llvm-ref/llvm-project
   cd ~/llvm-ref/llvm-project && git sparse-checkout set mlir/include mlir/lib mlir/test mlir/docs
   ```
   正文里每个类名、trait 名、接口文件名都要在这份树里 grep 确认过再写。注意 `/tmp` 会被清空，别放那儿。

## 3. 版本基线（已定：LLVM/MLIR 23.1.x）

**基线 = LLVM 23.1.x**（23.1.1 发布于 2026-09-08，本书写作时的最新发布版），已写进附录 A 第 0 节，正文统一按它写。网页代码块跑的是 Compiler Explorer 的 **trunk** 版工具，两者极少数情况下会不一致——这类地方用 `warn` 框提醒读者去 `--help` 核对。

MLIR 演进快，pass 名、API、目录都会变，所以：

- 不要凭记忆写 pass 名。每个 pass 选项都要用 `mlir-opt --help`、`mlir-opt --list-passes`（或对应目录下的 `Passes.td`）核对过。
- 已知的易变点，写到哪里就要提醒读者去核对：
  - `mlir-cpu-runner` 已更名为 `mlir-runner`（旧名在老版本里）。
  - `--convert-*-to-*` 系列 pass 名与选项时有调整；`--one-shot-bufferize` 的选项变动频繁。
  - `linalg` 的 named op 集合、`transform` 方言的 op 名都在动。
  - `LLVM_ENABLE_PROJECTS=mlir` vs `LLVM_ENABLE_RUNTIMES` 等构建开关。
- 不确定的 API 一律用 `-norun` 代码块并保持通用；**绝不编造不存在的 op、pass 或类名**。

## 4. 声音与语气

- 用「你」称呼读者，像一位资深同事在白板前讲解。**专业、精炼、有洞察**，不啰嗦、不灌水。
- 核心手法是**对照 + 追问「为什么」**：
  - 先用读者已会的 **LLVM IR / Pass / 指令选择**建立锚点，再讲 MLIR 怎么做、为什么这么做（例：block argument 取代 phi；region 嵌套取代 CFG-only；一切是 op 取代固定指令集）。
  - 每引入一个设计，先说它在解决什么问题，再给机制。**不要 API 罗列**。
- 允许英文术语，首次出现给「中文（English）」并加粗，如 **方言（dialect）**、**部分转换（partial conversion）**。约定见 §9。
- 承认现实：MLIR 有难用、不稳、文档缺失的地方。该说「这里坑在哪、社区正在改」就直说，这是本书的信任来源之一。

## 5. 贯穿案例：pix 方言（第 7 章起，每章增量）

一个小型**图像处理方言**，选它的理由：语义简单到能一眼看懂，但同时具备**逐点、邻域（stencil）、归约**三类计算，足以真实地演示 tiling / fusion / bufferization / 向量化 / GPU 下降——Toy 那种纯标量语言做不到这些。

**类型与属性**（批次 2 落地时做的修正，与初版规格不同）
- pix 的图像**直接用内建的 `tensor<HxWxf32>`**，不再自定义 `!pix.image`。理由：第四部分要把 pix 降到 `linalg`，而 `linalg` 本来就在 `tensor` 上工作；如果图像是自定义类型，中间要插一整套毫无教学价值的类型转换，还会挡住 `SameOperandsAndResultType` 这类现成 trait。
- 自定义类型改用 **`!pix.kernel<3x3>`**（stencil 权重：参数化类型，带行列两个参数与自定义格式）——它确实是一个独立概念，值得单独成类型，正好当第 9 章的教材。
- 自定义属性用 **`#pix.border<clamp|zero|wrap>`**（枚举属性，`EnumAttr` 封装），给 `pix.convolve` 描述边界处理方式。第 9 章讲属性时用它。

**Op 集与章节归属**（每章只加一点，避免来回改）
| 章 | 新增 | 讲什么 |
| --- | --- | --- |
| ch07 | `pix.add` | 项目骨架跑通；`Pure` 换来免费 CSE |
| ch08 | `pix.mul`、`pix.scale`、`pix.transpose`、`pix.reduce`、`pix.pipeline`+`pix.yield` | ODS：操作数/结果类型不一致的情况、属性、region、builder、`extraClassDeclaration` |
| ch09 | `!pix.kernel<3x3>`、`#pix.border`、`pix.convolve` | 自定义参数化类型与枚举属性 |
| ch10 | （不加 op）transpose/convolve/pipeline 的 verifier | 不变式与诊断 |
| ch11 | （不加 op）folder 与 canonicalizer | 常量折叠、两次转置消去、乘 1 消去 |
| ch12 | `pix.constant` + 测试体系 | lit/FileCheck、`check-pix` |

**下降路径（第四部分的主干）**
```
pix  --(ch20)-->  linalg on tensor  --(ch21)-->  tiled scf.forall + linalg
     --(ch22)-->  memref           --(ch23)-->  vector
     --(ch24)-->  scf → cf → llvm  --> JIT 运行，出图
     --(ch25)-->  gpu.launch → NVVM / ROCDL / SPIR-V
```

**代码落地**：`books/inside-mlir/code/pix/` —— 一个完整的 out-of-tree MLIR 项目（`CMakeLists.txt`、`include/Pix/`、`lib/`、`pix-opt/`、`test/`）。每章在正文里只贴关键片段，完整工程在这里；`verify/verify-pix.sh` 负责构建它并跑 lit 测试。

## 6. 每章固定结构

1. `# 第 N 章 标题`（与 `book.json` 完全一致）
2. `## 这一章想让你带走什么` + 紧跟 `<div class="goal-box"><p>…纯 HTML 段落…</p></div>`
3. 正文：**问题 → 机制 → 源码锚点 → 动手**。每章至少：
   - **1 张概念 SVG 图**（§8）
   - **1 处真实源码锚点**：`mlir/include/...` 或 `mlir/lib/...` 的路径 + 类名/函数签名
   - **1 段可交互内容**：`mlir` 代码块（点开在 CE 里跑 `mlir-opt`）或浏览器可运行的 `python` 模拟器
4. `## 动手：…` 一节：给命令 + 预期输出（「运行结果：」块），并引导读者改一处再跑。
5. `---` 后 `**这一章的要点**` 要点列表回顾，并**预告下一章**。
6. 第 7 章起，末尾加一行 **pix 进度**：本章给 pix 加了什么、项目现在能干什么。

## 7. 代码块约定

### 7.1 build.js 支持（已实现）

`build.js` 里已加 `mlirSpec` + `mlirRunArgs()`（紧挨 `gpuSpec`）：

| 围栏 | 用途 | 按钮 / 底部命令 |
| --- | --- | --- |
| ` ```mlir ` | 完整、能 round-trip 的 MLIR 模块 | 「🔗 在 Compiler Explorer 打开（mlir-opt）」+ 本地 `mlir-opt input.mlir` |
| ` ```mlir-lower ` | 展示一条下降管线的输入 | 「🔬 看下降结果（mlir-opt）」，参数取自块首 `// RUN:` 行（§7.3），默认 `--canonicalize` |
| ` ```mlir-translate ` | IR → LLVM IR 那一步（ch24） | 「🔬 看 LLVM IR（mlir-translate）」，默认 `--mlir-to-llvmir` |
| ` ```tablegen ` | ODS 片段（.td） | 无按钮 + `mlir-tblgen -gen-op-decls …`（CE 没有 tblgen） |
| ` ```mlir-norun ` | IR 片段、伪 IR（原有） | 无 |
| ` ```cpp ` / ` ```cpp-norun ` | Pass / pattern 的 C++ 实现 | `cpp` 有 godbolt 按钮，但 MLIR 的 C++ 缺头文件编不过，**本书一律用 `cpp-norun`** |
| ` ```python ` | 浏览器真运行的概念模拟器（Pyodide） | ▶ 运行 |
| ` ```bash ` | 终端命令 | 无 |

MLIR 系列围栏（`mlir` / `mlir-lower` / `mlir-translate` / `mlir-norun`）用 **`language-mlir`**；`llvm` / `spirv-norun` / `tablegen` 仍用 `language-llvm`。

**高亮已配好**（批次 1 之后补的）：`shared/vendor/prism-llvm.min.js`（Prism 1.29 官方组件）+ `shared/mlir-syntax.js`（我们自己的 30 行派生脚本）。后者在 llvm 语法基础上派生出 `mlir`：`//` 行注释、`^bb0` 块标签、`tensor<4x?xf32>` 里的形状、以及 `index`/`f32`/`memref` 这批内建类型；并用 `(?!\.)` 区分「`memref<8xf32>` 里的 memref 是类型」与「`memref.load` 里的 memref 是方言名」。**不动 `llvm` 语法本身**，所以《深入 LLVM 与 SYCL》的 LLVM IR 块不受影响（它的 `mlir-norun`/`cir-norun` 块反而一起受益）。改动点：`build.js` 模板里在 prism-bash 之后加两行 script（顺序要求：core → llvm → mlir-syntax）。

**CE 接线已核实**（`https://godbolt.org/api/compilers/mlir`）：语言 id `mlir`，编译器 id **`mliropttrunk`**、**`mlirtranslatetrunk`**（另有 14.0.0/14.0.5/16.0.0 的固定版本，太老，不用）。已实测 trunk 版接受 `--canonicalize`、`--verify-roundtrip`、`--pass-pipeline=…`、`--convert-to-llvm`、`--one-shot-bufferize`、`--mlir-print-ir-after-change`、`--mlir-print-ir-tree-dir=`、`--show-dialects`、`--mlir-print-op-generic`，verifier 报错会带诊断原样返回。

> **注意**：CE 那份是**关断言**的构建，`--debug-only=` 在它上面不存在（实测报 `Unknown command line argument`）。所以讲调试的段落（ch14、ch17）不要用 `mlir` 块配 `--debug-only` 的 RUN 行——那个按钮点开必然报错。调试命令一律用 ` ```bash ` 展示，并说明需要开断言的本地构建（附录 A 已写明）。

### 7.2 随书校验脚本（已实现，见 `books/inside-mlir/verify/README.md`）

```bash
books/inside-mlir/verify/verify-mlir.sh                # 抽 mlir / mlir-lower / mlir-translate 块逐个跑
books/inside-mlir/verify/verify-mlir.sh ch16 ch17      # 只校验指定章
MLIR_BIN=~/llvm-project/build/bin books/…/verify-mlir.sh
CE=1 books/inside-mlir/verify/verify-mlir.sh           # 没有本地 MLIR：走 godbolt 的 trunk 工具
books/inside-mlir/verify/verify-pix.sh                 # 构建 code/pix/ 并跑 check-pix
```

- `mlir` 块默认 `--verify-roundtrip`；`mlir-lower` 默认 `--canonicalize`；`mlir-translate` 默认 `--mlir-to-llvmir`；块首有 `// RUN:` 行则以它为准。
- 章节列表**按 `chapters/*.md` 实际存在的文件来**，新增章不用改脚本。
- `CE=1` 走 `ce-compile.js`（Node 内建 https，不依赖 curl），把块发给 godbolt 跑——**没有本地 LLVM 也能验证 pass 名和语法**，这条路实测可用。网络/服务异常与"代码真错了"分开计（退出码 2 vs 1）。
- 工具链缺失时干净退出（127）并指向附录 A；全通过 0，任一块失败 1，可接 CI。不参与 `node build.js`。

### 7.3 块首指令约定

`mlir-lower` 块第一行用 lit 风格注释写出管线，校验脚本据此执行，读者也能直接照抄：

```
// RUN: mlir-opt %s --pass-pipeline='builtin.module(func.func(...))'
```

## 8. SVG 概念图规范

沿用 `inside-llvm-sycl`/GPU 书的写法：
`<div class="concept"><svg viewBox="0 0 宽 高" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif"> … </svg><div class="caption">图 N-x　说明</div></div>`

- 配色：**MLIR / 高层方言用蓝 `#2b6cb0`**，**LLVM / 低层用灰 `#6b7280`**，强调/变换动作用橙 `#e06a3b`，成功路径/结论用绿 `#3f9b5c`。
- 本书必画的图（别画装饰图）：op/region/block 的嵌套树、def-use 边、方言分层「塔」、pattern worklist 的收敛过程、dialect conversion 的类型物化、linalg indexing map 的迭代空间、tile 后的分块与 halo、memref descriptor 的字段布局、GPU 三个出口的分叉。
- 图号「图 N-1、N-2」（N=章号）。
- 双栏对照：`<div class="compare"><div class="known"><div class="compare-head">LLVM IR</div>…</div><div class="cpp-side"><div class="compare-head">MLIR</div>…</div></div>`，div 内的代码块前后要留空行。
- 提示框：`<div class="note">`（蓝，补充）、`<div class="warn">`（橙，陷阱/易变 API）、`<div class="keypoint">`（绿，核心观念）。内部纯 HTML，`<strong>` 开头做小标题。

## 9. 术语统一

- **操作（operation / op）**、**方言（dialect）**、**区域（region）**、**块（block）**、**块参数（block argument）**、**特征（trait）**、**接口（interface）**。
- 正文里 op 名一律写全限定小写形式（`arith.addf`、`scf.for`、`linalg.generic`、`memref.alloc`），不写成「AddFOp」除非在讲 C++ 类名。
- C++ 类名保持原样（`OpBuilder`、`RewritePatternSet`、`ConversionTarget`、`TypeConverter`）。
- **ODS**（Operation Definition Specification）、**DRR**（Declarative Rewrite Rule）、**PDL/PDLL**、**IRDL** 首次出现给全称。
- 「下降」= lowering，「合法化」= legalization，「物化」= materialization，「缓冲化」= bufferization（首次给英文）。
- 与《深入 LLVM 与 SYCL》的对照术语保持一致（附录 B 给 LLVM ↔ MLIR 概念全表）。

## 10. 各部分要点与源码锚点（写作提纲）

**第一部分（ch01–06）· 核心抽象**
- ch01：从真实痛点切入——每个 DSL 都重造前端、过早 lower 丢信息、多目标各写一套。MLIR 的三个回答：可扩展 op、方言共存、渐进下降（progressive lowering）。不要一上来堆术语。
- ch02：`Operation` 的实际内存布局（尾部分配 operands/results）、`OpOperand` 的 def-use 链、region/block/block argument。与 LLVM IR 逐项对照表。锚 `mlir/include/mlir/IR/Operation.h`。
- ch03：generic form ↔ custom form 的关系（generic 是唯一真相）、location 信息、`mlir-opt` 的 `--pass-pipeline`、`--mlir-print-ir-after-all`、`--debug-only`。本章要把读者「能自己动手看」这件事立住。
- ch04：`StorageUniquer` 与不可变、context 生命周期、`tensor`/`memref`/`vector`/`index`/`DenseElementsAttr`。讲清「类型和属性都是唯一化的不可变值」这个反直觉点。
- ch05：全书最重要的一章之一。trait（编译期 mixin，如 `Pure`、`Terminator`、`IsolatedFromAbove`）vs interface（运行时多态，如 `MemoryEffectOpInterface`、`LoopLikeOpInterface`、`DestinationStyleOpInterface`）。讲清「pass 为什么能在不认识你的 op 的情况下变换它」。锚 `mlir/include/mlir/IR/OpDefinition.h`、`Interfaces/`。
- ch06：内建方言巡览与分层：`builtin`/`func`/`arith`/`math`/`cf`/`scf`/`affine`/`tensor`/`memref`/`vector`/`linalg`/`gpu`/`spirv`/`llvm`/`transform`。给一张「塔」图，标出各自的抽象层与典型下降边。

**第二部分（ch07–12）· 造方言**
- ch07：out-of-tree 骨架、`find_package(MLIR)`、`add_mlir_dialect`、TableGen → `.h.inc`/`.cpp.inc` 的生成链（讲清 `#define GET_OP_CLASSES` 这套宏把生成代码塞进你的类里的把戏）。
- ch08：ODS 的 `Op<>`、`arguments`/`results`、`assemblyFormat`、自动/自定义 builder、`extraClassDeclaration`。讲「生成了什么」而不只是「怎么写」——带读者看一眼生成的 `.inc`。
- ch09：`TypeDef`/`AttrDef`、参数化类型、自定义 parser/printer、类型别名。`!pix.image` 落地。
- ch10：`verify()` 与 trait 自带验证的执行顺序、`InferTypeOpInterface`、诊断（`emitOpError`）、`--verify-diagnostics` 与 `-split-input-file` 测试写法。
- ch11：`fold()` vs `getCanonicalizationPatterns()` 的分工、`OpFoldResult`、canonicalize 的不动点与「两个互逆 pattern 打死循环」这个经典坑。
- ch12：把 pix 补齐成能用的方言：`pix-opt` 驱动、lit + FileCheck 测试目录、`ninja check-pix`。本章交付一个可复现的完整项目。

**第三部分（ch13–19）· 变换**
- ch13：`Pass`/`OperationPass<T>`、嵌套 pass manager 与 op 树的对应、`AnalysisManager` 的缓存与失效、为什么 MLIR 能**并发**跑 pass（`IsolatedFromAbove` 的作用）。与 LLVM 新 PM 对照。
- ch14：`RewritePattern`/`matchAndRewrite`、`PatternRewriter` 的通知机制、`benefit`、greedy driver 的 worklist 与收敛、`--debug-only=greedy-rewriter` 看它到底在干什么。
- ch15：DRR（`Pat<>`）与 PDLL 各自的定位与局限；明确给出「什么时候老老实实写 C++」的判断标准。
- ch16：dialect conversion 的四个部件：`ConversionTarget`（legal/dynamically legal/illegal）、`TypeConverter`、`ConversionPattern`、driver。partial vs full conversion 的区别与选择。
- ch17：全书最实用的一章。`addSourceMaterialization`/`addTargetMaterialization`、1:N 类型展开（`SignatureConversion`）、`applyPartialConversion` 失败信息怎么读、`failed to legalize operation` 的系统化排查法。锚 `mlir/lib/Transforms/Utils/DialectConversion.cpp`。
- ch18：transform 方言的动机（把调度从 pass 里解放出来）、handle/payload 模型、`transform.structured.tile_using_forall` 等常用 op、怎么在 pix 上驱动 tiling。
- ch19：`DominanceInfo`、`MemoryEffects`（side effect 建模是很多变换的前提）、稀疏数据流分析框架、`DataFlowSolver`。

**第四部分（ch20–25）· 下降**
- ch20：linalg 的核心是「用 indexing map 描述迭代空间与访存」，因此 tiling/fusion 可以通用地实现。named op vs `linalg.generic`，destination-passing style 为什么长那样。
- ch21：tiling 的数学（迭代空间切分 + 访存切片）、`scf.forall`、producer-consumer fusion、stencil 的 halo 处理。用 transform 方言脚本驱动，读者能改 tile size 重跑。
- ch22：值语义 → 内存语义这一步为什么难（别名、就地更新、所有权）；one-shot bufferize 的 in-place 分析、`BufferizableOpInterface`、常见的多余 copy 怎么来的。
- ch23：向量化的两条路（linalg vectorizer / 手写 pattern）、`vector.transfer_read/write` 与 mask、unroll 到目标寄存器宽度、`vector` → LLVM intrinsic。
- ch24：`scf`→`cf`→`llvm`、memref descriptor 的字段（allocated/aligned/offset/sizes/strides）与 ABI、`mlir-translate --mlir-to-llvmir`、`mlir-runner`/ExecutionEngine JIT 真跑出结果。
- ch25：`gpu` 方言、`gpu.launch` 与 kernel outlining、三个出口（NVVM→PTX、ROCDL、SPIR-V）及其共性/差异。与 GPU 书、SYCL 书交叉引用。

**第五部分（ch26–28）· 真实世界**
- ch26：Triton 的分层（`tt` → `ttg` → LLVM/PTX）；**重点讲 layout/encoding 系统**——这是 Triton 对 MLIR 的原创贡献，也是理解它性能的钥匙。与 GPU 书的 Triton 章交叉引用，不重复讲 Triton 语言用法。
- ch27：两种截然不同的用法对照——IREE（flow/stream/hal 分层 + 自带运行时，MLIR 用到极致）vs flang（fir/hlfir，传统前端只借 MLIR 做中端）。从中提炼「什么时候该造方言、造几层」。
- ch28：ClangIR（与 `inside-llvm-sycl` 第五部分接上，不重复，只讲「作为 MLIR 用户」的视角）；再诚实收尾：编译时间、API 不稳、文档、调试体验、方言碎片化——MLIR 还没解决的问题。

## 11. 取舍：不单独成章的话题及归属

明确不给单独章节，但要在正文/附录有一次交代和指路，避免读者以为漏了：
- **PDL / PDLL** → ch15 里讲。**IRDL** → ch06 末尾一段 + 附录 C。
- **Python bindings** → ch03 一段（读者可能用它探索 IR）+ 附录 C 指路。
- **bytecode 格式** → ch03 一段（`--emit-bytecode`）。
- **affine 方言与多面体** → ch06 概览 + ch21 对照（说明本书主线走 linalg/scf 而非 affine，并给理由）。
- **sparse tensor / sparsifier** → ch22 一段指路。
- **async / openmp / emitc / 硬件专用方言（amx、nvgpu、amdgpu）** → ch06 表格里列出 + 附录 C。
- **`--mlir-print-ir-tree-dir`、Action/tracing、reproducer** → ch03 与 ch17 的调试小节。
- 不写「MLIR 编程语言语法大全」式的参考章——那种内容读者去官网 LangRef 更合适，本书的价值在机制与判断。

## 12. 写作顺序与批次

照仓库现有节奏，一个部分一个 PR/批次（参考 `clangir-part5` 那次）：

| 批次 | 内容 | 前置 |
| --- | --- | --- |
| 0 ✅ | `build.js` 的 `mlir`/`mlir-lower`/`mlir-translate`/`tablegen` 支持 + `verify/`（含 `CE=1` 兜底）+ 附录 A（基线定为 LLVM 23.1.x） | 已完成 |
| 1 ✅ | ch01–06（第一部分）+ 附录 B | 已完成，16 个 MLIR 块 + 6 个 python 块全部实测通过 |
| 2 | ch07–12 + `code/pix/` 可编译项目 + `verify-pix.sh` | 批次 1；pix op 集定稿 |
| 3 | ch13–19（第三部分） | 批次 2 |
| 4 | ch20–25（第四部分） | 批次 3；下降管线在真机跑通 |
| 5 | ch26–28 + 附录 C + 全书交叉引用校对 | 批次 4 |

每批次结束跑 `node build.js inside-mlir` 确认网页生成，并跑对应 verify 脚本。

## 13. 质量红线

- **代码必须真的过 `mlir-opt`**：每个 `mlir` 块在写进正文前先本地跑一遍（至少 `--verify-roundtrip`）。展示「运行结果」时贴**真实输出**，不要手写想象的 IR。
- **pass 名、op 名、C++ 类名不得编造**，一律核对过源码或 `--help`。不确定就改用 `-norun` 并写得通用。
- **源码路径必须存在**：引用 `mlir/...` 路径时对着当前版本确认；行号不要写死（版本一变就错），引用**类名/函数名**。
- 可运行 `python` 模拟器：完整、可独立运行、只用标准库、不依赖前文变量；配套「运行结果：」块必须与真实输出一致。
- 交叉引用（「第 N 章」）与 `book.json` 一致；与 GPU 书、SYCL 书的交叉引用写明是哪本书。
- **只用公开信息**：不含任何公司内部代码、内部项目名或未公开信息。
- 篇幅 250–450 行/章；宁可讲透一个机制，不要摊平罗列 API。
- **每章写完必跑三项自检**（批次 1 全过）：
  1. `CE=1 books/inside-mlir/verify/verify-mlir.sh <ch>` —— 所有 MLIR 块过工具；
  2. 抽出 python 块真跑一遍，与正文「运行结果：」逐字节 diff；
  3. `node build.js inside-mlir` 后检查生成的 HTML：**每章至少 1 张 SVG**、SVG 内没有被误插 `<p>`、CE 按钮和运行按钮数量符合预期。
- **课后题的答案必须自己先跑一遍。** 批次 1 里有三道题的真实报错与初稿的暗示不符（parser 先于 verifier 报错、隐式 terminator、`--convert-to-llvm` 不报错但留下 `scf.for`），都按实测重写了——这类"意料之外"往往是最好的教学点，但前提是你验过。
