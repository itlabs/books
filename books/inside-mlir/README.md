# 深入 MLIR

> **Inside MLIR** · 写给编译器工程师的多层 IR 之旅：方言、变换与下降。
> 你已经读过 LLVM IR、写过 Pass，也大概知道「MLIR 是个能自定义 IR 的框架」——这本书要带你真的造一个方言出来，给它写变换，再把它一路下降到 CPU 和 GPU 上跑起来。

## 这本书写给谁

你是一名有经验的 C++ / 编译器 / 系统工程师。你理解 SSA、支配关系、Pass、指令选择；如果你读过《[深入 LLVM 与 SYCL](../inside-llvm-sycl/README.md)》的第一部分，那正好是本书的前置知识（但不是必须的——第 1、2 章会把需要的 LLVM 概念重新摆一遍）。

你缺的不是编译器经验，而是对 MLIR 这套**「什么都能自己定义」的基础设施**的手感：

- 官方 Toy 教程七章看完，能跑，但合上就忘——因为它只演示了 API 序列，没讲清**为什么 MLIR 长这样**、每个设计在解决什么问题。
- 真实项目（Triton、IREE、flang、ClangIR）的代码一打开就是几十个方言、上百个 pass，不知道从哪根线头拉。
- 最难的几块——**Interface、Dialect Conversion、bufferization、Transform 方言**——恰恰是文档最薄、坑最深的地方。

这本书就是冲着这三件事写的。

## 这本书的特别之处

- **源码为锚**：每个机制都对应 `llvm-project/mlir/` 里的**真实文件路径、类名与函数签名**（如 `mlir/include/mlir/IR/PatternMatch.h` 的 `RewritePattern`、`mlir/lib/Transforms/Utils/DialectConversion.cpp`）。不是伪代码，是你机器上就有的代码。
- **一个方言贯穿全书**：第 7 章起，我们从零造一个图像处理方言 **`pix`**（逐点运算 + stencil + 归约），后面每一章都在它身上加东西：类型、verifier、folder、Pass、pattern、conversion、tiling、bufferization、向量化，最后落到 LLVM JIT 和 GPU。全书结束时你手上是一个**能编译、能跑、带 lit 测试的 out-of-tree MLIR 项目**，而不是一堆散片段。
- **看得见的内部结构**：Region 的嵌套、pattern 的 worklist 怎么收敛、dialect conversion 的类型物化、linalg 的 indexing map、tile 后的迭代空间、memref descriptor 的内存布局——全部配**图示**。
- **可运行 + 可验证**：`mlir` 代码块带 **🔗 在 Compiler Explorer 打开（mlir-opt）**，点开就能看一个 pass 跑前跑后的 IR 差别；抽象机制（pattern 重写的收敛、conversion 的合法性判定、tiling 的迭代空间）用**浏览器里能跑的 Python 小模拟器**画出来。全书 MLIR 代码带随书校验脚本，可在任何装了 `mlir-opt` 的机器上一次性验证。

## 目录

### 第一部分 · 多层 IR 的世界：MLIR 的核心抽象
- [第 1 章 为什么还要一层 IR：LLVM IR 的天花板与 MLIR 的答案](chapters/ch01.md)
- [第 2 章 IR 的形状：Operation / Region / Block / Value 与 LLVM IR 对照](chapters/ch02.md)
- [第 3 章 读写 MLIR：通用格式、自定义格式与 mlir-opt 的日常用法](chapters/ch03.md)
- [第 4 章 类型与属性：唯一化、MLIRContext 与内建类型](chapters/ch04.md)
- [第 5 章 Trait 与 Interface：MLIR 复用的真正机关](chapters/ch05.md)
- [第 6 章 方言全景：内建方言巡览与它们的分工](chapters/ch06.md)

### 第二部分 · 造一个方言：TableGen/ODS 与 pix 方言
- [第 7 章 起一个 out-of-tree 方言：目录、CMake 与 mlir-tblgen 构建链](chapters/ch07.md)
- [第 8 章 ODS（一）：用 TableGen 声明 Op、参数、结果与 assemblyFormat](chapters/ch08.md)
- [第 9 章 ODS（二）：自定义类型与属性，!pix.kernel 是怎么来的](chapters/ch09.md)
- [第 10 章 verifier 与不变式：把「不该出现的 IR」挡在门外](chapters/ch10.md)
- [第 11 章 fold 与 canonicalize：常量折叠、恒等化简与收敛](chapters/ch11.md)
- [第 12 章 pix 方言成型：完整 op 集、pix-opt 与 lit 测试](chapters/ch12.md)

### 第三部分 · 变换：Pattern、Pass 与 Dialect Conversion
- [第 13 章 Pass 框架：嵌套 PassManager、分析缓存与并发执行](chapters/ch13.md)
- [第 14 章 Pattern Rewrite 引擎：RewritePattern、benefit 与 greedy driver](chapters/ch14.md)
- [第 15 章 声明式重写：DRR 与 PDLL，以及什么时候别用它们](chapters/ch15.md)
- [第 16 章 Dialect Conversion（一）：TypeConverter、ConversionTarget 与部分转换](chapters/ch16.md)
- [第 17 章 Dialect Conversion（二）：materialization、1:N 转换与调试「转不过去」](chapters/ch17.md)
- [第 18 章 Transform 方言：把「怎么变换」本身写成 IR](chapters/ch18.md)
- [第 19 章 分析与副作用：支配关系、MemoryEffects 与数据流框架](chapters/ch19.md)

### 第四部分 · 下降：从张量一路走到机器
- [第 20 章 linalg：结构化操作、indexing map 与「为什么它好变换」](chapters/ch20.md)
- [第 21 章 tiling 与 fusion：把 pix 的逐点与 stencil 变成分块循环](chapters/ch21.md)
- [第 22 章 Bufferization：从 tensor 的值语义到 memref 的内存语义](chapters/ch22.md)
- [第 23 章 vector 方言：向量化、mask 与按目标宽度展开](chapters/ch23.md)
- [第 24 章 落到 LLVM：scf→cf、memref 描述符与 JIT 跑起来](chapters/ch24.md)
- [第 25 章 落到 GPU：gpu 方言、kernel outlining 与三个出口](chapters/ch25.md)

### 第五部分 · 真实世界：别人的编译器怎么用 MLIR
- [第 26 章 Triton：Python DSL 到 PTX，layout 系统是它的核心创造](chapters/ch26.md)
- [第 27 章 IREE 与 flang：端到端 AI 编译栈与 Fortran 前端的两种用法](chapters/ch27.md)
- [第 28 章 ClangIR 与未来：C++ 也变成方言，以及 MLIR 还没解决的问题](chapters/ch28.md)

### 附录
- [附录 A 搭建：构建 LLVM/MLIR、out-of-tree 模板与常用命令](chapters/appendix-a.md)
- [附录 B 术语与对照表：LLVM ↔ MLIR 概念对照与缩写](chapters/appendix-b.md)
- [附录 C 源码导览地图：想看某个机制去 mlir/ 的哪个目录](chapters/appendix-c.md)

> MLIR 仍在快速演进，API 与 pass 名字会变。本书正文以**附录 A 里写明的那个 LLVM 版本**为准；读的时候请以你手上那份 `llvm-project` 为准，正文会在容易变的地方提醒你去哪里核对。

## 怎么用这本书

1. **第一、二部分按顺序读，别跳。** 第 1–6 章立起「一切都是 op、方言可自定义、Trait/Interface 负责复用」这套心智模型；第 7–12 章把它变成你自己能编译的代码。这两部分是后面所有内容的地基。
2. **把 pix 项目跟着敲出来。** 第 7 章给出的 out-of-tree 骨架，后面每章都在其上增量推进。跟着做一遍，胜过读十遍文档。
3. **点开代码块动手。** MLIR 段落点 **🔗 在 Compiler Explorer 打开（mlir-opt）**，亲眼看一个 pass 把 IR 改成了什么；机制段落点 **▶ 运行** 跑浏览器里的 Python 模拟器。
4. **第三、四部分可以按需跳读。** 遇到实际问题（「我的 conversion 报 failed to legalize」「tensor 怎么变 memref」）时直接查对应章——第 17、22 章就是照着真实踩坑写的。
5. **第五部分当地图用。** 读完前四部分再去翻 Triton / IREE / flang 的源码，你会发现它们只是把同一套机制用到了不同深度。

准备好从「用 MLIR 的人」变成「设计 IR 的人」了吗？从[第 1 章](chapters/ch01.md)开始。
