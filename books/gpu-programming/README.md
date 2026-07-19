# 看得见的 GPU 编程

> 写给资深程序员的并行心智模型：CUDA、SYCL、Triton 与 CuTe DSL。
> 你已经会写多线程、懂缓存和内存带宽了——这本书要讲的，是当成千上万个线程同时跑起来时，你脑子里那套 CPU 模型要怎么改。

## 这本书写给谁

你是一名资深工程师，能熟练用 C++ 或 Python 解决真实问题，理解线程、锁、缓存、SIMD 这些概念。你**不需要**别人再教你什么是循环、什么是数组。

你缺的不是编程经验，而是一套新的**心智模型**：

- 在 CPU 上，你习惯让**几个**线程各干各的活，靠缓存和乱序执行把单线程跑到极致；
- 在 GPU 上，你要让**成千上万个**线程齐步走，靠海量并行和内存带宽取胜——单个线程慢得可怜，但一起上就快得惊人。

这本书帮你完成这次切换。我们不从"什么是并行"讲起，而是不断把 GPU 的做法和你已经会的 **CPU 多线程 / SIMD** 做法对照，讲清**差异**和**差异背后的原因**。

## 这本书的特别之处

- **对照式讲解**：每个新概念都尽量先用一段你熟悉的 CPU 代码建立锚点，再看 GPU 怎么做、为什么这么做。
- **看得见的并行**：Grid/Block/Warp/Thread、内存层级、合并访问、bank conflict、warp 分歧——这些抽象概念全部配**图示**，而不是只有文字。
- **一个概念，四种写法**：同一件事（比如向量加法、softmax），分别用 **CUDA / SYCL / Triton / CuTe DSL** 写出来横向对比，你会看清它们其实是同一个模型的不同外衣。
- **可运行 + 看得见编译器**：
  - **CUDA / SYCL** 代码块带 **🔗 在 Compiler Explorer 打开**：点开就能在线编译、看生成的 PTX；每段都附**本地编译命令**（`nvcc` / `icpx`）。
  - **概念模拟**：SIMT 调度、warp 分歧、归约树这些机制，用**能在浏览器里直接跑的 Python**做成小模拟器，点"▶ 运行"就能看到线程们怎么动——**不需要 GPU 也能理解**。
  - **Triton / CuTe DSL** 附完整可跑的 Python 程序与运行命令，在你自己的 GPU 上执行。
- **示例全部可编译可执行**：书中的完整程序都能独立编译运行，不是伪代码。

## 目录

### 第一部分 · 换个心智模型
- [第 1 章 你会并行编程了——那 GPU 到底不一样在哪](chapters/ch01.md)
- [第 2 章 SIMT 与 SIMD：一条指令，一大群线程](chapters/ch02.md)
- [第 3 章 GPU 硬件的心智模型：SM、核心、内存与带宽](chapters/ch03.md)

### 第二部分 · 核心编程模型（本书的地基）
- [第 4 章 执行层级（一）：Grid、Block、Thread](chapters/ch04.md)
- [第 5 章 执行层级（二）：NDRange、Work-group、Work-item](chapters/ch05.md)
- [第 6 章 真正的调度单位：Warp 与 Sub-group](chapters/ch06.md)
- [第 7 章 内存层级：global、shared、register 与合并访问](chapters/ch07.md)
- [第 8 章 协作与同步：barrier、shuffle 与原子操作](chapters/ch08.md)
- [第 9 章 你的第一个真 kernel：向量加法的完整生命周期](chapters/ch09.md)

### 第三部分 · 四大技术，逐一展开
- [第 10 章 CUDA：最贴近硬件的编程模型](chapters/ch10.md)
- [第 11 章 CUDA 进阶：用 shared memory 做 tiling 矩阵乘](chapters/ch11.md)
- [第 12 章 SYCL：单一源码的跨厂商 C++](chapters/ch12.md)
- [第 13 章 Triton：用 Python 写块级 kernel](chapters/ch13.md)
- [第 14 章 Triton 进阶：算子融合与自动调优](chapters/ch14.md)
- [第 15 章 CuTe DSL：Layout 与 Tensor 的代数](chapters/ch15.md)

### 第四部分 · 性能与优化
- [第 16 章 性能模型：occupancy、带宽与 Roofline](chapters/ch16.md)
- [第 17 章 访存优化：合并、bank conflict 与向量化](chapters/ch17.md)
- [第 18 章 并行原语：归约与扫描的经典模式](chapters/ch18.md)
- [第 19 章 现代硬件：Tensor Core、异步拷贝与流水线](chapters/ch19.md)

### 第五部分 · 融会贯通
- [第 20 章 同一个 softmax，四种技术怎么写](chapters/ch20.md)
- [第 21 章 综合项目：从零实现 fused attention 前向](chapters/ch21.md)

### 附录
- [附录 A 搭好环境：CUDA Toolkit、oneAPI、Triton 与在线工具](chapters/appendix-a.md)
- [附录 B 术语对照表：CUDA ↔ SYCL ↔ OpenCL ↔ Triton](chapters/appendix-b.md)
- [附录 C 读懂 GPU 报错与用好 profiler](chapters/appendix-c.md)
- [附录 D 继续前进：文档、书目与工具链](chapters/appendix-d.md)

## 怎么用这本书

1. **按顺序读第一、二部分。** 第 1—9 章是全书地基：SIMT 模型、执行层级、warp、内存层级、同步。这套模型立住了，后面四大技术都只是它的不同语法。
2. **点开代码块动手改。** CUDA/SYCL 段落点 **🔗 在 Compiler Explorer 打开** 看生成的 PTX；概念段落点 **▶ 运行** 跑浏览器里的 Python 模拟，亲眼看线程怎么调度。
3. **用附录 B 查术语。** CUDA 的 block = SYCL 的 work-group = Triton 的 program，一张表帮你在四套术语间自由换算。
4. **没有 GPU 也能学。** 概念、模拟、PTX 都不需要显卡；等你有了 GPU（或云实例），再回来跑 Triton / CuTe 的完整例子。

准备好从"几个线程"升级到"成千上万个线程"了吗？从[第 1 章](chapters/ch01.md)开始。
