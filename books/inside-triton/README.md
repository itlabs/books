# 深入 Triton

> **Inside Triton** · 写给 kernel 作者的 Triton：从写对、写快到看懂编译器。
> 你会用 PyTorch，跑通过 Triton 的官方 tutorial，也许还写过一点 CUDA。这本书要带你从「能跑」走到「知道它为什么快、为什么慢」。

## 这本书写给谁

你是一名资深工程师，熟练使用 Python 和 PyTorch，懂多线程、缓存和 SIMD。写过 CUDA 当然好，从没碰过 GPU 也没关系：够用的 GPU 硬件模型和 MLIR 知识，书里都从头讲。

你卡住的地方，多半是下面四件事之一：

- 官方 tutorial 能跑通，可自己写出来的 kernel **比 cuBLAS / FlashAttention 慢一截，不知道差在哪**。
- 改了 `BLOCK_SIZE`、`num_warps`、`num_stages`，性能乱跳，**没有一个能预测结果的心智模型**，只能盲调。
- 编译器是个黑盒：TTIR / TTGIR 看不懂，**layout 是什么、`convert_layout` 为什么贵**，没人讲清楚。
- 生态在分层：往上有 Helion，往下有 Gluon / TLX，旁边还有 CuTe DSL、TileLang。**不知道什么时候该离开 Triton，该往哪一层走**。

这本书的每一章，都对应其中至少一件。

## 这本书的特别之处

- **一个项目贯穿全书**：第 4 章起，我们用 Triton 一点点写出一个简化的 decoder-only Transformer 层（**minillm**）。逐点、归约、GEMM、attention、MoE 路由、decode attention，每一类计算对应一种典型的性能瓶颈。全书结束时，你手上有一个带测试、带性能测量、能和 PyTorch eager / Inductor 做端到端对比的小项目。
- **先预测，再测量，再解释**：第三部分讲性能时，每改一个参数，先让你猜它会快还是慢，再测，再用硬件模型解释为什么。目标是让你下次不用测也能猜对。
- **源码为锚，IR 是真的**：每个机制都对应 Triton 3.9.0 源码里的真实路径和类名；正文里的每一段 TTIR / TTGIR 都是真 dump 出来的，并且能指回它来自哪一行 `tl.*`。
- **不用 GPU 也能看 IR**：`triton-ir` 代码块带 **🔬 在 Compiler Explorer 看 IR** 按钮，点开就是源码、编译器、各级 IR 三栏，CE 还会给源码行和对应的 IR 行上同样的底色。
- **两种 GPU 上实测**：代码在 NVIDIA A10 和 Intel Data Center GPU Max 1100 上跑过，性能数字都注明是哪一台测的。同一个 kernel 换了后端会遇到什么，第 23 章专门讲。
- **诚实**：Triton 有写不快、不稳定、文档缺失的地方。该说「这里不如手写 CUDA」「这个 API 下个版本可能就没了」的地方，书里会直说。

## 目录

### 第一部分 · 心智模型
- [第 1 章 为什么是 Triton：从 PyTorch eager 到手写 kernel 之间的鸿沟](chapters/ch01.md)
- [第 2 章 够用的 GPU 硬件模型](chapters/ch02.md)
- [第 3 章 块级编程：一个 program 处理一块数据](chapters/ch03.md)
- [第 4 章 第一个 kernel：vector add 的完整生命周期](chapters/ch04.md)

### 第二部分 · 写对 Triton
- [第 5 章 tl 语言核心](chapters/ch05.md)
- [第 6 章 归约与逐行算子](chapters/ch06.md)
- [第 7 章 tl.dot 与第一个 GEMM](chapters/ch07.md)
- [第 8 章 访存的三种写法](chapters/ch08.md)
- [第 9 章 调试与测试](chapters/ch09.md)
- [第 10 章 接入 PyTorch](chapters/ch10.md)

### 第三部分 · 写快 Triton
- [第 11 章 性能模型与测量](chapters/ch11.md)
- [第 12 章 GEMM 优化（一）：tile 尺寸与执行顺序](chapters/ch12.md)
- [第 13 章 GEMM 优化（二）：软件流水](chapters/ch13.md)
- [第 14 章 autotune](chapters/ch14.md)
- [第 15 章 新架构特性在 Triton 里怎么用](chapters/ch15.md)
- [第 16 章 低精度](chapters/ch16.md)
- [第 17 章 FlashAttention 前向](chapters/ch17.md)

### 第四部分 · 看懂 Triton 编译器
- [第 18 章 够用的 MLIR](chapters/ch18.md)
- [第 19 章 编译管线全景](chapters/ch19.md)
- [第 20 章 Layout 系统（一）](chapters/ch20.md)
- [第 21 章 Layout 系统（二）：linear layout 与 convert_layout](chapters/ch21.md)
- [第 22 章 关键 pass 与性能定位](chapters/ch22.md)
- [第 23 章 三个后端](chapters/ch23.md)

### 第五部分 · Triton 的上一层与下一层
- [第 24 章 Gluon（一）：为什么需要再往下一层](chapters/ch24.md)
- [第 25 章 Gluon（二）：手写高性能 GEMM](chapters/ch25.md)
- [第 26 章 Helion：带 tile 的 PyTorch](chapters/ch26.md)
- [第 27 章 同一个 kernel 的另几种写法](chapters/ch27.md)

### 第六部分 · 实战
- [第 28 章 MoE：grouped GEMM 与路由](chapters/ch28.md)
- [第 29 章 推理 kernel：decode attention](chapters/ch29.md)
- [第 30 章 收官：串起一层，再谈 Triton 还没解决的问题](chapters/ch30.md)

### 附录
- [附录 A 环境搭建](chapters/appendix-a.md)
- [附录 B 概念对照表](chapters/appendix-b.md)
- [附录 C 环境变量与调试开关速查](chapters/appendix-c.md)
- [附录 D 源码导览地图](chapters/appendix-d.md)

> Triton 演进很快：访存 API 在从指针迁到 tensor descriptor，Gluon 还在 `experimental` 下，环境变量也时有增减。本书正文以**附录 A 写明的 Triton 3.9.0** 为准，容易变的地方会提醒你去哪里核对。

## 怎么用这本书

1. **先装环境。** 照[附录 A](chapters/appendix-a.md) 装好，跑一遍体检脚本 `probe-env.py`，14 项全过再往下读。没有 GPU 也能读：第四部分和所有 `triton-ir` 代码块都不需要 GPU。
2. **第一、二部分按顺序读。** 第 1–4 章立起「一个 program 处理一块数据」的心智模型，第 5–10 章教你把 kernel 写对、接进 PyTorch。后面所有内容都建立在这里。
3. **跟着把 minillm 敲出来。** 正文只贴关键片段，完整代码在 `code/minillm/`。每章末尾有一行「minillm 进度」。
4. **第三部分动手测。** 先猜，再跑 `bench/` 里的脚本，再对照正文的解释。你的 GPU 和本书的不一样，数字会不同，但趋势应当一致；不一致的地方往往最值得琢磨。
5. **第四、五部分按需读。** 碰到「为什么这里多了一个 `convert_layout`」「Triton 推导的 layout 不够好怎么办」这类问题时，直接去对应的章。

准备好打开这个黑盒了吗？从[第 1 章](chapters/ch01.md)开始。
