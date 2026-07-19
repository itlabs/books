# 附录 D 继续前进：文档、书目与工具链

## 这一章想让你带走什么

<div class="goal-box">
<p>这本书给了你一套<strong>心智模型</strong>——从"海量弱小线程"到 SIMT、执行层级、内存层级，再到四种技术怎么把同一个模型换不同外衣写出来。但一本书装不下整个领域。这一节是一张<strong>带注解的路标图</strong>：官方文档、关键库、值得读的经典论文/思想、社区，以及针对不同目标（ML 算子、HPC、系统底层）的<strong>进阶路线</strong>。</p>
<p>每条都标明"这是什么、什么时候该用它"，而不是一堆干链接。URL 只在我确信是官方规范地址时才给。</p>
</div>

## 官方文档：先读这些

任何时候有疑问，官方文档都应是第一信源——它们比二手教程更准、更新更快。

- **CUDA C++ Programming Guide**（`docs.nvidia.com/cuda`）：CUDA 编程模型的权威参考，执行层级、内存模型、同步、协作组、Tensor Core 等都以此为准。**当你对某个 CUDA 概念的确切语义不确定时查它**，是本书第 4–11、19 章的官方底本。
- **CUDA C++ Best Practices Guide**（同站）：从"能跑"到"跑得快"的官方优化清单——合并访问、occupancy、传输优化、指令级建议。**本书第 16、17 章的思路和它一脉相承**，读完本书后通读它一遍收获很大。
- **CUDA Runtime / Driver API Reference**：查具体 API 签名和错误码（配合附录 C）。
- **SYCL 规范（Khronos）与 Intel oneAPI 文档**（`spec.oneapi.io`、Intel oneAPI 编程指南）：SYCL 语言语义的权威来源，`nd_range`、accessor、USM 的确切行为查这里（第 5、12 章）。AdaptiveCpp 有自己的文档讲它的后端和编译选项。
- **OpenCL 规范（Khronos）**：需要跨最广厂商、或维护既有 OpenCL 代码时的参考（附录 B 的对照基准之一）。
- **Triton 文档与教程**（`triton-lang.org`）：官方教程尤其好——向量加法、融合 softmax、矩阵乘、dropout、layer norm 一路带你走，**是第 13、14 章之后最该做的动手练习**。`triton.language` 的 API 参考也在这里。
- **CUTLASS / CuTe 文档**（CUTLASS 的 GitHub 仓库与其中的 `media/docs`）：CuTe 的 Layout 代数、Tensor、以及 CUTLASS 的模板化 GEMM。第 15 章之后深入的官方材料。

<div class="note">
<strong>读文档的姿势</strong>
不要试图通读 Programming Guide——太长。<strong>把它当字典</strong>：带着一个具体问题去查（"shared memory 的 bank 到底怎么分？""nd_range 的 global 是总数还是组数？"），读那一节，验证在代码里。Best Practices Guide 则相反，值得从头到尾读一遍建立全局优化观。
</div>

## 关键库：别重复造轮子

生产代码里，很多高性能原语已经有厂商调到极致的实现，直接用往往比自己写快得多。

- **cuBLAS**：稠密线性代数（GEMM 等）的 NVIDIA 官方实现。**要做矩阵乘先想它**，你手写的 tiling 版（第 11 章）是为了理解原理，生产上很难拼过 cuBLAS。
- **cuDNN**：深度学习原语（卷积、pooling、归一化、attention 等）。框架底层大量依赖它。
- **CUB**：CUDA 的**设备级/块级并行原语**模板库——归约、扫描、排序、直方图。**第 18 章那些归约/扫描模式，生产中直接用 CUB**，它替你处理了所有边界和架构差异。
- **Thrust**：CUDA 上的 STL 风格高层库（`thrust::sort`、`reduce`、`transform`），快速原型和 host 端友好的接口。
- **oneMKL**：Intel oneAPI 的数学核心库，SYCL 世界里 BLAS/FFT/随机数的对应物（第 12 章之后）。
- **PyTorch custom ops / `torch.compile`**：把你写的 Triton/CUDA kernel 接进 PyTorch 的正道。**ML 方向的读者**这是把第 13–14 章成果用起来的关键——写个自定义算子替换掉框架里慢的那一环。
- **CUTLASS**：当 cuBLAS 的固定接口满足不了你（要融合、要特殊数据类型、要自定义 epilogue）时，用 CUTLASS 的模板拼一个定制 GEMM。CuTe（第 15 章）是它的底层。

## 值得读的经典思想与论文

工具会过时，思想不会。下面几组是这个领域的"必读经典"，按主题给出——都能按名字搜到原文。

- **Roofline 模型**（Williams、Waterman、Patterson 提出的性能模型）：本书第 16 章的理论根基。理解"算术强度决定你受限于带宽还是算力"这一个模型，胜过记一百条优化技巧。
- **并行归约与扫描（scan）的经典讲法**：Mark Harris 的《Optimizing Parallel Reduction in CUDA》是归约优化的启蒙经典，一步步把归约从朴素版优化到打满带宽；Blelloch 的 work-efficient scan 是前缀和的理论基础（第 18 章）。
- **FlashAttention**（Dao 等）：把 attention 从 memory-bound 的多趟读写，通过分块 + 在线 softmax + 算子融合变成一趟搞定的里程碑工作。**它几乎用上了本书后半本的每一个概念**（tiling、shared memory、融合、在线归约），是第 20、21 章 fused attention 的思想来源。读它的论文，你会对"为什么融合、为什么分块"有豁然开朗的理解。
- **在线 softmax / 数值稳定的 softmax**（online softmax，Milakov & Gimelshein 一类工作）：FlashAttention 的关键子技术，第 20 章 softmax 的进阶版。
- **memory-bound 优化文献**：围绕"多数真实 kernel 卡在访存"展开的一系列工作，呼应本书从第 3 章就立下的主线。

<div class="keypoint">
<strong>读论文的性价比</strong>
不必啃全部证明。<strong>先读 FlashAttention 和 Mark Harris 的归约那篇</strong>——它们把本书教的抽象概念落到了两个真实、影响巨大的 kernel 上，读完你会发现本书的每一章都在那里各就各位。这种"概念→真实 kernel"的闭环，比再读十篇教程都值。
</div>

## 社区与持续跟进

- **NVIDIA Developer 论坛 与 CUDA 相关的技术博客**：官方工程师答疑、新架构特性的第一手解读。
- **Triton 的 GitHub 仓库（issue / discussion）**：Triton 迭代快，遇到 API 变动或 bug，仓库里往往有最新答案。
- **PyTorch 论坛与其 GPU/编译相关讨论**：ML 算子方向的问题集散地。
- **各技术的 GitHub 仓库本身**：CUTLASS、Triton、AdaptiveCpp 的 example / test 目录是最好的"实战代码库"，比教程更贴近真实用法。

## 三条进阶路线：按你的目标选

学完本书，往哪走取决于你想解决什么问题。三条典型路线：

<div class="concept">
<svg viewBox="0 0 560 240" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">
  <text x="280" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#2b6cb0">本书之后，三条路</text>
  <!-- 共同起点 -->
  <rect x="200" y="32" width="160" height="30" rx="6" fill="#eef6ff" stroke="#2b6cb0"/>
  <text x="280" y="52" text-anchor="middle" font-size="11" fill="#2b6cb0">本书：GPU 心智模型 + 四种技术</text>
  <!-- 三条分叉 -->
  <g stroke="#6b7280" stroke-width="1.2" fill="none">
    <path d="M240 62 L110 95" marker-end="url(#ad)"/>
    <path d="M280 62 L280 95" marker-end="url(#ad)"/>
    <path d="M320 62 L450 95" marker-end="url(#ad)"/>
  </g>
  <!-- ML -->
  <rect x="20" y="98" width="170" height="120" rx="6" fill="#f0fdf4" stroke="#3f9b5c"/>
  <text x="105" y="116" text-anchor="middle" font-size="12" font-weight="bold" fill="#2f7a45">ML 算子作者</text>
  <text x="105" y="138" text-anchor="middle" font-size="10" fill="#2f7a45">Triton → CuTe DSL</text>
  <text x="105" y="156" text-anchor="middle" font-size="10" fill="#2f7a45">+ PyTorch custom ops</text>
  <text x="105" y="174" text-anchor="middle" font-size="10" fill="#2f7a45">读 FlashAttention</text>
  <text x="105" y="192" text-anchor="middle" font-size="10" fill="#2f7a45">目标：写快过框架的</text>
  <text x="105" y="207" text-anchor="middle" font-size="10" fill="#2f7a45">融合算子</text>
  <!-- HPC -->
  <rect x="200" y="98" width="160" height="120" rx="6" fill="#fff4ec" stroke="#e06a3b"/>
  <text x="280" y="116" text-anchor="middle" font-size="12" font-weight="bold" fill="#a53f1c">HPC / 科学计算</text>
  <text x="280" y="138" text-anchor="middle" font-size="10" fill="#a53f1c">CUDA + SYCL</text>
  <text x="280" y="156" text-anchor="middle" font-size="10" fill="#a53f1c">跨厂商可移植</text>
  <text x="280" y="174" text-anchor="middle" font-size="10" fill="#a53f1c">cuBLAS / oneMKL</text>
  <text x="280" y="192" text-anchor="middle" font-size="10" fill="#a53f1c">目标：把物理模拟</text>
  <text x="280" y="207" text-anchor="middle" font-size="10" fill="#a53f1c">/ 求解器搬上 GPU</text>
  <!-- Systems -->
  <rect x="370" y="98" width="170" height="120" rx="6" fill="#eef6ff" stroke="#2b6cb0"/>
  <text x="455" y="116" text-anchor="middle" font-size="12" font-weight="bold" fill="#2b6cb0">系统 / 底层</text>
  <text x="455" y="138" text-anchor="middle" font-size="10" fill="#2b6cb0">CUDA 深挖</text>
  <text x="455" y="156" text-anchor="middle" font-size="10" fill="#2b6cb0">PTX / SASS、异步拷贝</text>
  <text x="455" y="174" text-anchor="middle" font-size="10" fill="#2b6cb0">CUTLASS / CuTe</text>
  <text x="455" y="192" text-anchor="middle" font-size="10" fill="#2b6cb0">目标：榨干硬件、</text>
  <text x="455" y="207" text-anchor="middle" font-size="10" fill="#2b6cb0">写库级 kernel</text>
  <defs>
    <marker id="ad" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#6b7280"/></marker>
  </defs>
</svg>
<div class="caption">图 D-1　三条进阶路线共享本书这套心智模型，再按目标各自深入不同的技术栈</div>
</div>

- **ML 算子作者**（想给模型写更快的算子）：深耕 **Triton**（第 13、14 章）打好块级 kernel 的手感，再上 **CuTe DSL**（第 15 章）应对更复杂的张量布局；学会用 **PyTorch custom ops / `torch.compile`** 把 kernel 接进真实模型；精读 **FlashAttention** 作为范本。第 21 章的 fused attention 就是这条路的缩影。
- **HPC / 科学计算**（想把模拟、求解器搬上 GPU）：以 **CUDA** 为主力打底，用 **SYCL**（第 12 章）获得跨厂商可移植性；重度依赖 **cuBLAS / oneMKL** 等成熟库；把精力放在算法的并行化和访存优化（第 16–18 章）上。
- **系统 / 底层**（想写库、榨干硬件）：把 **CUDA 往深里挖**——读 PTX/SASS、玩转异步拷贝与流水线（第 19 章）、精通 profiler（附录 C）；进入 **CUTLASS / CuTe** 的模板世界写库级 GEMM。这条路最硬核，也最接近硬件的极限。

## 写在最后

如果你从第 1 章一路走到这里，回头看看你已经跨过的那道坎：一开始，"启动一百万个线程、每个只算一个元素"听起来近乎荒谬；现在，你不但接受了它，还能推理出为什么分支会串行、为什么访存顺序比次数重要、为什么多数 kernel 卡在带宽、为什么融合能救命。你换掉的不是几个 API，而是**整套关于"一个线程值多少钱"的直觉**。

四种技术——CUDA、SYCL、Triton、CuTe DSL——你现在应该能看穿：它们是同一个模型的不同外衣，学会看穿外衣，比记住任何一套语法都重要，因为语法会变，模型不会。

真正的功夫在书外的键盘上。找一个你手头真实的、又多又独立的计算，把它搬上 GPU，用附录 C 的 profiler 量一量，再对着 Roofline 优化到打满带宽。那一刻你会明白，这本书教的从来不是"怎么写 GPU 代码"，而是**怎么像 GPU 一样思考**。

去写吧。GPU 在等你喂满它的带宽。

---

**这一章的要点**

- **官方文档是第一信源**：CUDA Programming Guide / Best Practices Guide、SYCL 规范与 oneAPI 文档、Triton 文档（`triton-lang.org`）、CUTLASS/CuTe 文档。当字典查，Best Practices 值得通读。
- **别重复造轮子**：矩阵乘用 cuBLAS/CUTLASS，归约扫描用 CUB/Thrust，SYCL 数学用 oneMKL，接进 ML 用 PyTorch custom ops。
- **必读经典**：Roofline 模型、Mark Harris 的并行归约、FlashAttention——后者几乎用上了本书后半本的全部概念。
- **三条路线**：ML 算子 → Triton + CuTe；HPC → CUDA + SYCL；系统底层 → CUDA 深挖 + CUTLASS。都共享本书这套心智模型。
- **看穿外衣**：四种技术是同一个模型的不同写法；模型不变，语法会变。剩下的功夫，在你自己的键盘上。
