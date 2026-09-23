# 附录 B 术语与对照表：LLVM ↔ MLIR 概念对照与缩写

## 这份附录怎么用

<div class="goal-box">
<p>这是一份<strong>查的</strong>附录，不用读。四张表：从 LLVM 概念翻译到 MLIR（你已有的知识怎么迁移）、缩写与中英术语（读社区讨论和源码注释时用）、方言速查（看到方言名立刻定位它在哪一层）、以及<strong>常见报错 → 真正含义 → 去哪一章</strong>（卡住时最快的入口）。</p>
<p>报错那张表里的每条信息都是在本书基线版本（LLVM 23.1.x，见附录 A）上实际跑出来的，不是从文档抄的措辞。</p>
</div>

## 一、LLVM ↔ MLIR 概念对照

| LLVM | MLIR | 关键差别 | 细讲 |
| --- | --- | --- | --- |
| `Module` | `builtin.module` op | 模块是普通 op，不是特殊容器 | 第 2 章 |
| `Function` | `func.func` op（还有 `gpu.func`、`llvm.func`） | 函数是 op；"函数"不止一种，靠 `FunctionOpInterface` 统一 | 第 2、6 章 |
| `BasicBlock` | `Block` | block 可以有**参数** | 第 2 章 |
| `Instruction` | `Operation` | 指令集不固定；op 可嵌套 region | 第 2 章 |
| `Value` | `Value`（`OpResult` 或 `BlockArgument`） | 只有这两种来源，函数参数不是特例 | 第 2 章 |
| `Use` / `User` | `OpOperand` | def-use 边是对象，值维护侵入式 use 链表 | 第 2 章 |
| `phi` 指令 | **块参数 + 分支传参** | MLIR 没有 phi | 第 2 章 |
| `Type`（固定集合） | `Type`（方言可注册） | 唯一化的不可变值；比较 = 指针比较 | 第 4 章 |
| metadata（可丢） | **固有属性** `<{…}>` / **可丢弃属性** `{…}` | 固有属性有 verifier 保证，不会被"合法地丢掉" | 第 2、4 章 |
| intrinsic | 就是普通 op | 不需要"特殊指令"这个概念 | 第 1 章 |
| `Pass` / `PassManager` | `Pass` / 嵌套 `PassManager` | 管线**锚定**在 op 类型上，可并发 | 第 3、13 章 |
| `InstVisitor` / 手写 switch | **trait + interface** | pass 不问"你是谁"，只问"你声明了什么" | 第 5 章 |
| `-O2` 之类的固定管线 | `--pass-pipeline='…(…)'` 字符串 | 管线是数据，可自由拼装 | 第 3、13 章 |
| `opt` | `mlir-opt` | 同一个位置的工具 | 第 3 章 |
| `llc` / 后端 | `--convert-*-to-llvm` + `mlir-translate` | 先降到 `llvm` 方言，再翻成 LLVM IR | 第 24 章 |
| LLVM IR 的 `.ll` | `.mlir`（文本）/ 字节码 | 两种等价序列化形态 | 第 3 章 |
| `RAUW`（`replaceAllUsesWith`） | 同名 | 概念一致，是所有重写的原子动作 | 第 2、14 章 |
| 没有对应物 | **Region** | MLIR 最大的结构性新东西：IR 成为一棵树 | 第 2 章 |
| 没有对应物 | **方言（dialect）** | 命名空间 + 可扩展的 op/类型/属性集合 | 第 6 章 |
| 没有对应物 | **渐进下降** | 高层和低层 op 合法共存，一步步降 | 第 1、6 章 |

## 二、缩写与中英术语

### 缩写

| 缩写 | 全称 | 是什么 |
| --- | --- | --- |
| **ODS** | Operation Definition Specification | 用 TableGen 声明 op/类型/属性的那套 DSL（第 8、9 章） |
| **DRR** | Declarative Rewrite Rule | TableGen 里的 `Pat<>`，声明式写重写规则（第 15 章） |
| **PDL / PDLL** | Pattern Descriptor Language | 把"匹配什么模式"本身写成 IR / 写成一门小语言（第 15 章） |
| **IRDL** | IR Definition Language | 把方言定义本身写成 IR（第 6 章提及） |
| **TD** | TableGen definition（`.td` 文件） | ODS、DRR、接口定义都写在 `.td` 里 |
| **SSA** | Static Single Assignment | 每个值只被定义一次 |
| **CFG** | Control Flow Graph | 基本块 + 边 |
| **CSE** | Common Subexpression Elimination | 合并重复计算（第 5 章实测） |
| **DCE** | Dead Code Elimination | 删掉没人用又没副作用的 op（第 5 章） |
| **LICM** | Loop Invariant Code Motion | 把循环不变量提出循环 |
| **RAUW** | Replace All Uses With | 把"所有用我的地方"改成用别人（第 2 章） |
| **UB** | Undefined Behavior | MLIR 里有个 `ub` 方言显式表达它 |

### 中英术语（本书统一用法）

| 中文 | English | 说明 |
| --- | --- | --- |
| 方言 | dialect | op / 类型 / 属性的命名空间与集合 |
| 操作 | operation, op | MLIR 的基本单位 |
| 区域 | region | op 内部的 block 列表 |
| 块参数 | block argument | 取代 phi 的机制 |
| 特征 | trait | 编译期静态声明，回答"是/不是"（第 5 章） |
| 接口 | interface | 运行时虚表，回答"具体是多少"（第 5 章） |
| 唯一化 | uniquing | 内容相同的类型/属性只存一份（第 4 章） |
| 下降 | lowering | 从高抽象层换到低抽象层 |
| 渐进下降 | progressive lowering | 每次只降一点，中途始终是合法 IR |
| 折叠 | folding | 用常量或已有值替换 op 的结果（第 11 章） |
| 规范化 | canonicalization | 把 IR 变成约定的标准形态（第 11 章） |
| 合法化 | legalization | 把 IR 变成目标方言允许的形态（第 16 章） |
| 物化 | materialization | 类型转换时插入桥接 op（第 17 章） |
| 缓冲化 | bufferization | 从 tensor（值语义）到 memref（内存语义）（第 22 章） |
| 支配 | dominance | 定义必须支配使用（SSACFG region 的要求） |
| 副作用 | side effect | 读/写/分配内存等（第 5 章） |
| 可推测执行 | speculatable | 能否被无条件提前执行（`AlwaysSpeculatable`） |
| 迭代空间 | iteration space | `linalg` 用 indexing map 描述它（第 20 章） |
| 分块 | tiling | 把大迭代空间切成小块（第 21 章） |
| 融合 | fusion | 把生产者消费者合到一个循环里（第 21 章） |
| 外提（抽成函数） | outlining | 比如把 kernel 抽成 `gpu.func`（第 25 章） |

## 三、方言速查

层号对应第 6 章图 6-1：0 最抽象，6 最贴近机器。

| 方言 | 层 | 代表 op / 类型 | 一句话 |
| --- | --- | --- | --- |
| `builtin` | — | `builtin.module`、`tensor`/`memref`/`vector` 类型 | 结构容器与内建类型 |
| `func` | 4 | `func.func`、`func.call`、`func.return` | 函数与调用 |
| `arith` | 3 | `addi`/`addf`/`cmpi`/`select`/`constant` | 标量与逐元素算术；**整数 signless** |
| `math` | 3 | `exp`/`log`/`sqrt`/`fma` | 超越函数 |
| `index` | 3 | `index.add`、`index.mul` | 专门处理 `index` 类型运算 |
| `scf` | 2 | `scf.for`/`while`/`if`/`forall`/`yield` | **结构化**控制流 |
| `affine` | 2 | `affine.for`、`affine.load`、`affine_map` | 仿射约束下的循环，依赖分析精确 |
| `cf` | 4 | `cf.br`、`cf.cond_br`、`cf.switch` | **非结构化**控制流（像 LLVM IR） |
| `tensor` | 1 | `tensor.empty`、`extract_slice`、`insert_slice` | 张量，**值语义** |
| `memref` | 2 | `alloc`/`load`/`store`/`subview`/`dim` | 内存缓冲，**内存语义** |
| `bufferization` | 2 | `to_memref`、`to_tensor`、`alloc_tensor` | tensor↔memref 那条边（第 22 章） |
| `vector` | 3 | `transfer_read`/`transfer_write`、`contract` | 宽度已定的向量运算（第 23 章） |
| `linalg` | 1 | `linalg.generic`、`matmul`、`fill` | 结构化计算枢纽，indexing map（第 20 章） |
| `tosa` | 0 | 标准算子集 | 常作前端入口 |
| `sparse_tensor` | 1 | 稀疏编码 + 自动稀疏化 | 稀疏张量 |
| `shape` | 1 | `shape.shape_of` 等 | 动态形状推理 |
| `gpu` | 5 | `gpu.module`/`func`/`launch_func`/`thread_id`/`barrier` | 厂商无关的设备抽象（第 25 章） |
| `nvvm` / `rocdl` | 6 | 目标内建函数 | NVIDIA / AMD 出口 |
| `spirv` | 6 | SPIR-V op 与类型 | Vulkan/OpenCL/Intel GPU 出口 |
| `llvm` | 6 | `llvm.func`、`llvm.load`、`!llvm.ptr`、`!llvm.struct` | LLVM IR 的 MLIR 镜像（**不是** LLVM IR） |
| `emitc` | 6 | 生成 C/C++ 源码 | 反方向出口 |
| `transform` | 横切 | `transform.structured.tile_using_forall` 等 | 把"怎么变换"写成 IR（第 18 章） |
| `pdl` / `pdl_interp` | 横切 | 把"匹配什么"写成 IR | 声明式重写底座（第 15 章） |
| `ub` | 横切 | `ub.poison` | 显式的未定义行为 |
| `async` / `omp` / `acc` | 横切 | 并发与卸载模型 | — |

> 另有一批硬件专用方言：`X86`、`ArmNeon`/`ArmSVE`/`ArmSME`、`AMDGPU`、`NVGPU`、`XeGPU`；以及 `MPI`、`Shard`、`SMT`、`WasmSSA`、`MLProgram`、`Ptr`、`DLTI`、`Quant`。想知道你手上的二进制注册了哪些，敲 `mlir-opt --show-dialects`。

## 四、常见报错 → 真正含义 → 去哪一章

本书前六章实测出现过的报错，按"看到它先想什么"整理：

| 报错片段 | 真正含义 | 先做什么 | 章 |
| --- | --- | --- | --- |
| `operand #0 must be signless-non-zero-bitwidth-integer-like, but got 'tensor<4x4xf32>'` | op 的**类型约束**不满足（这句话由 ODS 里的约束自动生成，不是手写检查） | 核对 op 的 ODS 定义要求什么类型 | 第 1、4、8 章 |
| `use of value '%x' expects different type than prior uses: 'f32' vs 'i1'` | **解析器**报的：同一个 SSA 名在文本里有两种类型 | 这是语法层错误，verifier 还没上场 | 第 2、3 章 |
| `use of undeclared SSA value name` | 文本格式里 SSA 名**按 region 作用域**，你在外面引用了内部的名字 | 想跨 region 传值就走块参数/结果 | 第 2 章 |
| `operand #0 does not dominate this use` | 真正的**支配关系**错误（SSACFG region 的要求） | 看它附带的 `note: operand defined here` | 第 2 章 |
| `'scf.if' op along control flow edge … region successor needs 1 inputs` | region 之间传值的**个数/类型**不匹配；常见于忘了 yield（隐式 terminator 帮你插了个空的） | 检查每个 region 的 yield 值与 op 结果是否对齐 | 第 2、5 章 |
| `operation being parsed with an unregistered dialect` | 这个方言没注册 | 加 `--allow-unregistered-dialect`，或者该给它建个方言了 | 第 3、7 章 |
| `Unknown command line argument '--xxx'` | pass 名/选项在**这个版本**不存在（或需要开断言的构建，比如 `--debug-only`） | `mlir-opt --help \| grep` 核对 | 附录 A |
| **没有任何报错，但 IR 一点没变** | ①管线 anchor 拼错（整串 pass 静默跳过）②pass 顺序不对，跑的时候目标 op 还不存在 | 用 `--mlir-print-ir-after-all`：它一行都不打印就是一个都没跑 | 第 3、6 章 |
| **没有报错，但 IR 里还剩别的方言的 op** | 下降不完整。`--convert-to-llvm` 不覆盖所有方言（例如 `scf` 不在其中） | 搜一下输出里还有哪些方言前缀 | 第 6、24 章 |

<div class="keypoint">
<strong>最后两行值得单独记住。</strong>
<p>MLIR 里最难查的问题通常不是报错，而是<strong>"什么都没发生"</strong>或<strong>"降了一半"</strong>。养成两个习惯：pass 不生效先查 anchor；下降完成后先确认 IR 里只剩目标方言。</p>
</div>
