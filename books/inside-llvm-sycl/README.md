# 深入 LLVM 与 SYCL

> **Inside LLVM and SYCL** ·  写给编译器工程师的源码之旅：LLVM、DPC++/SYCL 实现与扩展。
> 你已经会写 C++、读得懂编译原理教材上的「前端—优化—后端」了——这本书要带你打开 `intel/llvm` 这个真实的大型代码库，看清 LLVM 与 SYCL 到底是**怎么实现**的。

## 这本书写给谁

你是一名有经验的 C++ / 系统工程师，理解编译、链接、指针、虚函数、模板这些概念，也大致知道「编译器分前端、中端、后端」。你**不需要**别人再教你什么是抽象语法树、什么是寄存器。

你缺的不是编程经验，而是一张**通往真实编译器源码的地图**：

- 教科书讲的是「概念上的编译器」；这本书讲的是**你能 `cd` 进去、能 `grep`、能改一行重编**的那个编译器——`intel/llvm`（DPC++ 的上游）。
- 每一个抽象概念（Pass、IR、SPIR-V、offload、SYCL kernel、扩展），我们都会给出**具体的源码路径、真实的类名和函数签名**，你可以边读边翻。

## 这本书的特别之处

- **源码为锚**：每个机制都对应到 `intel/llvm` 里的**真实文件路径与类名**（如 `llvm/include/llvm/IR/PassManager.h` 的 `PassManager<IRUnitT>`、`sycl/source/detail/scheduler/`）。不是伪代码，是你机器上就有的代码。
- **看得见的内部结构**：IR 的 def-use 图、Pass 之间的依赖与分析缓存、命令组的依赖图、两趟编译的数据流、SPIR-V 与 PTX 的分叉——全部配**图示**，而不是只有文字。
- **从 LLVM 到 SYCL 再到扩展，一条线**：先立起 LLVM 的「IR 中心」心智模型，再看 SYCL 如何把一份 `.cpp` 拆成主机端与设备端两趟编译、如何在运行时建依赖图上设备，最后手把手实现一个扩展。
- **可运行 + 可验证**：C++/SYCL 代码块带 **🔗 在 Compiler Explorer 打开**，附本地编译命令（`clang` / `icpx -fsycl` / `opt` / `llvm-spirv`）；抽象机制（Pass 管线、依赖图、SSA、调度）用**浏览器里能跑的 Python 小模拟器**画出来——不需要搭好整套工具链也能理解。

## 目录

### 第一部分 · LLVM：IR 中心的编译器基础设施
- [第 1 章 LLVM 是什么：三段式架构与「IR 中心」哲学](chapters/ch01.md)
- [第 2 章 LLVM IR：Module / Function / BasicBlock / Instruction 与 SSA](chapters/ch02.md)
- [第 3 章 IR 的三种形态：内存对象、文本 .ll 与位码 .bc](chapters/ch03.md)
- [第 4 章 Pass 框架（一）：新 PassManager 的设计](chapters/ch04.md)
- [第 5 章 Pass 框架（二）：写一个 Pass，并用插件挂进 opt](chapters/ch05.md)
- [第 6 章 优化管线：PassBuilder 与 O0–O3 默认流水线](chapters/ch06.md)
- [第 7 章 后端与代码生成：Target、指令选择到机器码](chapters/ch07.md)

### 第二部分 · 异构：SPIR-V、offload 工具链与运行时
- [第 8 章 SPIR-V：异构世界的可移植中间表示与 llvm-spirv](chapters/ch08.md)
- [第 9 章 异构 offload：bundler / wrapper / linker-wrapper 工具链](chapters/ch09.md)
- [第 10 章 运行时 liboffload 与 plugins-nextgen：kernel 怎么上设备](chapters/ch10.md)
- [第 11 章 NVIDIA 支持：从 IR 到 PTX，libclc 与 NVPTX 后端](chapters/ch11.md)

### 第三部分 · SYCL（DPC++）的真实实现
- [第 12 章 SYCL 全景：单一源码、两趟编译与规范到实现的映射](chapters/ch12.md)
- [第 13 章 头文件地图：sycl.hpp 里到底有什么](chapters/ch13.md)
- [第 14 章 设备端编译：integration header/footer 与 kernel 命名](chapters/ch14.md)
- [第 15 章 属性与 attributes：[[sycl::…]] 如何落到 IR](chapters/ch15.md)
- [第 16 章 运行时（一）：queue / context / device / handler 对象模型](chapters/ch16.md)
- [第 17 章 运行时（二）：命令组、调度器与依赖图](chapters/ch17.md)
- [第 18 章 Unified Runtime：SYCL 如何对接 Level Zero / CUDA / OpenCL](chapters/ch18.md)
- [第 19 章 设备库与 libdevice：数学函数、fallback 与 SPIR-V 内建](chapters/ch19.md)

### 第四部分 · 实现一个 SYCL 扩展
- [第 20 章 扩展的解剖：命名约定、feature-test 宏与 doc/extensions](chapters/ch20.md)
- [第 21 章 实战：free function kernels 扩展](chapters/ch21.md)
- [第 22 章 需要编译器配合的扩展：SemaSYCL 与前端支持](chapters/ch22.md)
- [第 23 章 案例巡礼：bindless images 与 kernel_compiler（运行时编译）](chapters/ch23.md)
- [第 24 章 从提案到落地：写扩展的完整流程与测试](chapters/ch24.md)

### 附录
- [附录 A 搭建与构建 intel/llvm：config、compile 与常用命令](chapters/appendix-a.md)
- [附录 B 术语与缩写对照：LLVM / SYCL / SPIR-V / UR](chapters/appendix-b.md)
- [附录 C 源码导览地图：想看某个功能去哪个目录](chapters/appendix-c.md)

## 怎么用这本书

1. **按顺序读第一部分。** 第 1–7 章是全书地基：LLVM 的「IR 中心」哲学、IR 数据结构、Pass 框架与后端。这套模型立住了，后面 SYCL 的设备端编译、SPIR-V、offload 都只是它的具体应用。
2. **边读边翻源码。** 每个机制我们都给出 `intel/llvm` 里的真实路径。建议在另一个窗口打开代码库，读到类名就去 `grep` 一下，亲眼确认。
3. **点开代码块动手。** C++/SYCL 段落点 **🔗 在 Compiler Explorer 打开**；机制段落点 **▶ 运行** 跑浏览器里的 Python 模拟器，亲眼看 Pass 管线、依赖图、SSA 是怎么动的。
4. **想改编译器就照着扩展章走。** 第四部分从「扩展是什么」一路到「写一个 free function kernels 扩展并加测试」，是把前三部分的知识用起来的实战。

准备好从「概念上的编译器」走进**真实的编译器源码**了吗？从[第 1 章](chapters/ch01.md)开始。
