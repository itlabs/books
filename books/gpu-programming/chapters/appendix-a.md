# 附录 A 搭好环境：CUDA Toolkit、oneAPI、Triton 与在线工具

## 这一章想让你带走什么

<div class="goal-box">
<p>正文里的每一段代码都不是纸上谈兵：CUDA / SYCL 能在浏览器里看编译结果，Triton / CuTe DSL 能在免费云 GPU 上真跑起来。这一节把<strong>四条技术栈的落地路径</strong>一次讲清，从"一分钱不花、什么都不装"开始，到本地装齐全套工具链结束。</p>
<p>读完你会有一张明确的决策表：手头没有 NVIDIA 显卡时哪些能做、哪些做不了，以及每一步该敲哪条命令、怎么验证它真的装好了。</p>
</div>

## 先别急着装：没有 GPU 你也能开始

本书的设计原则之一，就是让你在没有显卡的情况下也能走完大半本。装工具链是有成本的（下载几个 GB、配环境变量、对付驱动版本），所以我们**从零安装的选项讲起**，把真正需要花钱租卡的地方留到最后。

### 浏览器里的 Python 模拟器：什么都不用装

正文每一章那些标着 ` ```python ` 、点 **▶ 运行** 就出结果的代码块，跑在浏览器内置的 **Pyodide** 上，**不需要任何安装、也不需要 GPU**。它们不真用显卡，而是用纯 Python 忠实模拟 SIMT 执行、warp 分歧、访存合并、归约树这些机制——这是本书"可交互"的核心。打开网页就能改一行、重跑、看差别，这条路对所有读者永远可用。

### Compiler Explorer（godbolt.org）：在线看 CUDA→PTX 和 SYCL

正文里的 ` ```cuda ` 和 ` ```sycl ` 代码块都带一个 **🔗 在 Compiler Explorer 打开** 的入口。[Compiler Explorer](https://godbolt.org)（俗称 godbolt）是一个在线编译器，它**不运行**你的程序，但会把源码**编译**给你看中间产物：

- **CUDA → PTX**：选 `nvcc` 编译器，你能看到 kernel 被编译成的 **PTX**（NVIDIA 的中间汇编）。这对理解"编译器把我写的 `c[i]=a[i]+b[i]` 变成了什么"极有价值——第 2、10 章反复用到。它甚至能进一步给出 SASS（真正的机器码）。
- **SYCL**：选 `icpx`（Intel 的 DPC++ 编译器）或 `AdaptiveCpp`，能编译 SYCL 源码看产物。

<div class="keypoint">
<strong>Compiler Explorer 能做什么、不能做什么</strong>
它是<strong>编译器</strong>不是<strong>运行环境</strong>：你能看到 PTX / 汇编、能确认代码能否通过编译、能对比不同优化选项的产物差异，但<strong>不能真的在 GPU 上执行</strong>看到运行结果。要看 kernel 跑出来的数字，你需要一块真实（或云上的）GPU。看编译产物不要 GPU，跑程序才要。
</div>

### Google Colab / Kaggle：免费云 GPU，跑 Triton 和 PyTorch 的推荐路径

这是**没有本地显卡时运行 Triton / CuTe DSL 章节代码的首选**。[Google Colab](https://colab.research.google.com) 和 [Kaggle Notebooks](https://www.kaggle.com/code) 都提供**免费的 NVIDIA GPU**（通常是 T4 一类），只要一个浏览器和账号就能用：

1. 新建一个 notebook；
2. 在菜单里把运行时（runtime / accelerator）改成 **GPU**；
3. 第一个 cell 里验证卡的存在，然后直接 `pip install` 你要的东西。

```bash
# 在 Colab / Kaggle 的一个 cell 里（前面加 ! 让它当 shell 命令跑）
!nvidia-smi                 # 确认真的分到了一块 GPU，看到型号和显存即可
!pip install -q triton torch
```

因为 Colab/Kaggle 用的是 Linux + NVIDIA GPU + 预装 CUDA，**Triton 在上面开箱即用**，省掉了本地配环境的全部麻烦。第 13–15、20、21 章的 Triton / CuTe DSL 实战，推荐就在这里跑。

<div class="note">
<strong>免费额度的现实</strong>
免费 GPU 有使用时长和排队限制，长时间不动会断开、重连后环境要重装（重跑那句 <code>pip install</code> 即可）。做练习和跑本书示例足够；要长时间训练大模型才需要付费实例或自购硬件。
</div>

## CUDA Toolkit：本地跑 CUDA 的地基

要在本地编译运行 ` ```cuda ` 程序，你需要两样东西：**NVIDIA 驱动**（让操作系统认识显卡）和 **CUDA Toolkit**（提供编译器 `nvcc` 和运行时库）。前提是你有一块 NVIDIA 显卡。

### 安装

**Linux**：最省心的方式是用发行版或 NVIDIA 官方仓库装。装完驱动后再装 toolkit：

```bash
# Ubuntu/Debian 示例：先装官方驱动（版本号按官网当前推荐替换）
sudo apt update
sudo apt install nvidia-driver-550       # 装完通常需要重启
# 再装 CUDA Toolkit（也可从 developer.nvidia.com/cuda-downloads 下 .run 或仓库包）
sudo apt install nvidia-cuda-toolkit
```

更通用、版本更可控的做法是去 **developer.nvidia.com/cuda-downloads** 按你的系统选安装方式（仓库 / runfile）。装完把 CUDA 的 `bin` 和 `lib64` 加进 `PATH` 和 `LD_LIBRARY_PATH`：

```bash
# 追加到 ~/.bashrc（路径里的版本号按实际安装替换）
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Windows**：到 **developer.nvidia.com/cuda-downloads** 下载图形化安装包，它会一并处理驱动、`nvcc` 和环境变量。编译 `.cu` 需要一个宿主 C++ 编译器，安装器会引导你装 Visual Studio 的 C++ 生成工具（MSVC）。装完在 **Developer Command Prompt** 或 PowerShell 里用 `nvcc`。用 WSL2 的话，按 Linux 流程装 toolkit，但**驱动装在 Windows 侧**，WSL 里不要另装驱动。

### 验证装好了

```bash
nvidia-smi          # 显示驱动版本、GPU 型号、显存占用——能看到卡就说明驱动 OK
nvcc --version      # 显示 CUDA 编译器版本——能打印版本号说明 toolkit 在 PATH 里
```

`nvidia-smi` 右上角还会显示这块驱动**支持的最高 CUDA 版本**；它要 ≥ 你 `nvcc` 的版本，否则运行时会报驱动太旧。

### Hello, kernel：编译并运行第一个 CUDA 程序

把下面这个完整程序存成 `hello.cu`（它就是第 1、9 章那个向量加法的可运行骨架）：

```cuda
#include <cstdio>

__global__ void add(float* c, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int N = 1 << 20;                 // 约 100 万个元素
    size_t bytes = N * sizeof(float);
    float *a, *b, *c;                      // host 端
    a = (float*)malloc(bytes);
    b = (float*)malloc(bytes);
    c = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) { a[i] = 1.0f; b[i] = 2.0f; }

    float *da, *db, *dc;                   // device 端
    cudaMalloc(&da, bytes);
    cudaMalloc(&db, bytes);
    cudaMalloc(&dc, bytes);
    cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    add<<<blocks, threads>>>(dc, da, db, N);
    cudaDeviceSynchronize();               // 等 kernel 跑完

    cudaMemcpy(c, dc, bytes, cudaMemcpyDeviceToHost);
    printf("c[0] = %.1f, c[%d] = %.1f\n", c[0], N - 1, c[N - 1]);  // 期望 3.0

    cudaFree(da); cudaFree(db); cudaFree(dc);
    free(a); free(b); free(c);
    return 0;
}
```

编译并运行：

```bash
nvcc hello.cu -o hello      # 编译；可加 -arch=sm_80 之类指定目标架构
./hello                     # 期望输出：c[0] = 3.0, c[1048575] = 3.0
```

看到 `c[0] = 3.0` 就说明从驱动、编译器到显存拷贝、kernel 启动的整条链路都通了。

<div class="warn">
<strong>常见第一坑：版本不匹配</strong>
最常见的报错是"CUDA driver version is insufficient for CUDA runtime version"——你的 <code>nvcc</code>（toolkit）比驱动新。要么升级驱动，要么用旧一点的 toolkit。在 WSL2 里则常见于误在 Linux 侧又装了一遍驱动，导致冲突。附录 C 会系统地讲这类报错。
</div>

## Intel oneAPI / DPC++：跑 SYCL

SYCL 是单一源码、跨厂商的 C++ 并行方言（第 12 章）。最主流的实现是 Intel 的 **oneAPI**，其编译器叫 **DPC++**，命令是 `icpx`。

### 安装 oneAPI（DPC++）

去 Intel 官网下 **oneAPI Base Toolkit**（Linux/Windows 都有安装包）。装完在每个新终端里先 source 一下环境脚本，`icpx` 才会进 `PATH`：

```bash
# Linux：安装后每次开终端先加载 oneAPI 环境
source /opt/intel/oneapi/setvars.sh
icpx --version                      # 能打印版本说明 DPC++ 就绪
```

### 编译一个 SYCL 程序

SYCL 用 `-fsycl` 打开：

```bash
icpx -fsycl vadd.sycl.cpp -o vadd   # 编译 SYCL 源码
./vadd
```

一个能编译的最小 SYCL 向量加法骨架（完整程序，对照第 12 章）：

```sycl
#include <sycl/sycl.hpp>
#include <cstdio>

int main() {
    const size_t N = 1 << 20;
    sycl::queue q;                                  // 默认选一个设备
    printf("运行设备：%s\n",
           q.get_device().get_info<sycl::info::device::name>().c_str());

    float *a = sycl::malloc_shared<float>(N, q);    // 统一共享内存，host/device 都能访问
    float *b = sycl::malloc_shared<float>(N, q);
    float *c = sycl::malloc_shared<float>(N, q);
    for (size_t i = 0; i < N; ++i) { a[i] = 1.0f; b[i] = 2.0f; }

    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        c[i] = a[i] + b[i];                         // 一个 work-item 处理一个元素
    }).wait();

    printf("c[0] = %.1f\n", c[0]);                  // 期望 3.0
    sycl::free(a, q); sycl::free(b, q); sycl::free(c, q);
    return 0;
}
```

### 让 SYCL 跑在 NVIDIA GPU 上

oneAPI 默认面向 Intel 的 CPU/GPU。要让 SYCL 代码**跑到 NVIDIA 显卡**上，有两条路：

- 给 DPC++ 装 **oneAPI for NVIDIA GPUs 插件**（Codeplay 提供），然后指定 CUDA 目标编译：

```bash
# 用 CUDA 后端做 AOT 编译（需已装该插件 + CUDA Toolkit）
icpx -fsycl -fsycl-targets=nvptx64-nvidia-cuda vadd.sycl.cpp -o vadd
```

- 用 **AdaptiveCpp**（前身叫 hipSYCL / Open SYCL）——一个独立于 Intel 的开源 SYCL 实现，对 NVIDIA、AMD、CPU 后端都友好，非 Intel 硬件用户尤其适用：

```bash
# AdaptiveCpp 的编译入口（安装后），--acpp-targets 指定后端
acpp -O2 --acpp-targets='cuda:sm_80' vadd.sycl.cpp -o vadd
```

<div class="note">
<strong>选 oneAPI 还是 AdaptiveCpp</strong>
手头是 Intel GPU 或想要最完整的官方工具链（含 profiler、oneMKL），用 <strong>oneAPI/DPC++</strong>。想在 NVIDIA/AMD 上跑 SYCL、或想要一个轻量开源实现，用 <strong>AdaptiveCpp</strong>。两者编译的是同一份 SYCL 源码——这正是 SYCL "单一源码跨厂商"的卖点。
</div>

## Triton：用 Python 写 kernel

Triton（第 13、14 章）是一个 Python 库，`@triton.jit` 装饰的函数会被即时编译成 GPU kernel。它的安装极简单，但有硬性前提。

```bash
pip install triton torch     # triton 常与 PyTorch 搭配（张量、显存都靠 torch）
```

```bash
# 验证：能 import 且 torch 认得到 CUDA 设备
python -c "import triton, torch; print('triton', triton.__version__, 'cuda?', torch.cuda.is_available())"
```

看到 `cuda? True` 才算真正就绪。

<div class="warn">
<strong>Triton 的两条硬前提</strong>
一是<strong>需要 NVIDIA GPU + 已装 CUDA</strong>（Triton 会用系统里的 CUDA 工具链）；二是 <strong>Triton 面向 Linux</strong>——官方支持以 Linux 为主，Windows 上原生支持不完善，最省心的办法是用 <strong>WSL2</strong> 或直接上 <strong>Colab/Kaggle</strong>。这也是前面把云 notebook 定为 Triton 章节推荐路径的原因：它一步到位地满足了 Linux + NVIDIA + CUDA。
</div>

一个最小的 Triton kernel 骨架（片段，完整版见第 13 章）：

```triton-norun
import torch, triton, triton.language as tl

@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)                 # 一个 program 处理一整块数据（≈ 一个 CUDA block）
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n                        # 边界屏蔽，等价于 CUDA 里的 if (i < n)
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(c_ptr + offs, a + b, mask=mask)
```

## CuTe DSL：CUTLASS 的布局代数

CuTe DSL（第 15 章）来自 NVIDIA 的 CUTLASS 项目，用 Layout/Tensor 代数来描述数据摆布和张量运算。它以 Python 包的形式发布：

```bash
pip install nvidia-cutlass-dsl     # 提供 CuTe DSL 的 Python 接口
```

和 Triton 一样，它**需要 NVIDIA GPU + CUDA 环境**才能真正运行 kernel。在 Colab/Kaggle 上安装同样是最省事的做法。验证安装：

```bash
python -c "import cutlass; print('cutlass dsl ok')"   # 能 import 即安装成功
```

## 一张表：有 GPU 和没 GPU，分别能做什么

把上面的路径浓缩成一张决策表——写到某一章卡在环境上时，回头看这里：

| 你想做的事 | 不需要 GPU | 需要 NVIDIA GPU（本地或云） |
|---|---|---|
| 跑本书的浏览器 Python 模拟器 | ✅ Pyodide，什么都不装 | — |
| 看 CUDA 编译成的 PTX / SASS | ✅ Compiler Explorer（nvcc） | 本地 `nvcc -ptx` 也行 |
| 看 SYCL 的编译产物 | ✅ Compiler Explorer（icpx / AdaptiveCpp） | — |
| 检查 CUDA/SYCL 代码能否编译 | ✅ Compiler Explorer | ✅ 本地编译器 |
| **运行** CUDA 程序看结果 | ❌ | ✅ CUDA Toolkit + 显卡 |
| **运行** SYCL 程序看结果 | ❌（Intel GPU 也可） | ✅ oneAPI / AdaptiveCpp + 设备 |
| **运行** Triton kernel | ❌ | ✅ Linux + CUDA（推荐 Colab/Kaggle） |
| **运行** CuTe DSL | ❌ | ✅ CUDA（推荐 Colab/Kaggle） |
| 用 profiler 分析性能（附录 C） | ❌ | ✅ 需真实 GPU |

<div class="keypoint">
<strong>一句话的落地建议</strong>
前 12 章（心智模型 + CUDA/SYCL 概念）用<strong>浏览器模拟器 + Compiler Explorer</strong>就能走完，一分钱不花、一个字节不装。真要动手跑 Triton / CuTe DSL 和做性能剖析（第 13 章起），最省心的路是 <strong>Google Colab 或 Kaggle 的免费 GPU</strong>；需要长期开发或大规模实验时，再考虑本地装齐 CUDA Toolkit 或租用云 GPU 实例。
</div>

---

**这一章的要点**

- **没有 GPU 也能开始**：浏览器 Python 模拟器零安装，Compiler Explorer 能看 CUDA→PTX 和 SYCL 编译产物（但不运行）。
- **要真跑 Triton/CuTe DSL**，最省心的免费路径是 **Google Colab / Kaggle**——现成的 Linux + NVIDIA GPU + CUDA。
- **CUDA Toolkit**：装驱动 + toolkit，用 `nvidia-smi` 和 `nvcc --version` 验证，一个向量加法 `.cu` 跑出 `3.0` 即全链路打通；当心驱动/toolkit 版本不匹配。
- **SYCL**：oneAPI 的 `icpx -fsycl` 是主流；用 `-fsycl-targets=nvptx64-nvidia-cuda` 或 **AdaptiveCpp** 可把 SYCL 跑上 NVIDIA。
- **Triton** 需 NVIDIA GPU + CUDA 且以 Linux 为主（Windows 用 WSL2 或云）；**CuTe DSL** 用 `pip install nvidia-cutlass-dsl`，同样需 CUDA 环境。
