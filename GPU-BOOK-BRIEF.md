# GPU 书 · 章节作者简报（写作时必读）

你在为《看得见的 GPU 编程》写章节。这是一本面向**资深程序员（会 C++/Python、懂多线程/缓存/SIMD，但 GPU 零基础）**的中文书，会被构建脚本转成交互式网页。

## 动手前必做
1. **通读 `books/gpu-programming/chapters/ch01.md`、`ch02.md`、`ch03.md`** —— 这是声音、结构、图示、代码块用法的黄金样板，务必模仿其风格与质量。
2. 读 `STYLE.md` 了解排版约定（但注意：本书读者是资深工程师，不是青少年，语气见下）。
3. 读 `books/gpu-programming/book.json` 了解全书章节顺序，以便正确交叉引用（"第 N 章会讲…"）。

## 声音与语气（关键：不同于 STYLE.md 的青少年语气）
- 用"你"称呼读者，像一位资深同事在白板前讲解。**专业、精炼、有洞察**，不幼稚、不啰嗦。
- 核心手法是**对照**：先用读者已会的 **CPU 多线程 / SIMD / 缓存** 建立锚点，再讲 GPU 怎么做、**为什么**这么做。
- 讲清"为什么"和设计权衡，不只是罗列 API。多用精准类比，但不油腻。
- 允许英文术语，首次出现给"中文（English）"并加粗，如 **合并访问（memory coalescing）**。

## 每章固定结构（照 ch01–03 的样子）
1. `# 第 N 章 标题`（与 book.json 完全一致）
2. `## 这一章想让你带走什么` + 紧跟一个 `<div class="goal-box"><p>…纯 HTML 段落…</p></div>`
3. 正文：概念 → 最小例子 → 讲解 → 动手改。每章**至少 1 张概念 SVG 图**、**至少 1 个双栏对照**、**至少 1 段浏览器可运行的 Python 模拟或实验**。
4. `## 动手：…` 一节，含可运行代码 + "运行结果："块 + 让读者改一行的引导。
5. `---` 后 `**这一章的要点**` 用要点列表回顾，并**预告下一章**。

## 代码块约定（本书构建脚本支持，务必用对）
- ` ```cuda ` —— CUDA C++，带"🔗 在 Compiler Explorer 打开（看 PTX）"+ nvcc 本地命令。**应是完整可编译的 .cu 程序**（含 `#include`、`main`、host 端分配/拷贝/启动）。
- ` ```cuda-ptx ` —— 同上但强调看 PTX（讲编译器把 kernel 编成什么时用）。
- ` ```cuda-norun ` —— 只高亮，不给按钮。用于片段、伪代码、只展示 kernel 体。
- ` ```sycl ` —— SYCL/DPC++，带 Compiler Explorer（icpx -fsycl）+ 本地命令。完整可编译程序。` ```sycl-norun ` 为片段。
- ` ```triton ` / ` ```triton-norun ` —— Triton（Python），只高亮 + `pip install triton torch` 本地命令。需 GPU。
- ` ```cutedsl ` / ` ```cutedsl-norun ` —— CuTe DSL（Python），只高亮 + 本地命令。需 GPU。
- ` ```python ` —— **会在浏览器里真运行（Pyodide）**。用于概念模拟器（如 ch01 的两种世界观、ch02 的 warp 分歧）。**必须完整、可独立运行、不依赖前文变量、只用标准库**（无 numpy/torch）。每个可运行 python 块后面配一个纯文本的"运行结果："代码块，内容与真实输出一致。
- ` ```python-norun ` —— 只高亮的 Python（需要 numpy/torch/GPU 的示范代码放这里）。
- ` ```cpp ` / ` ```cpp-norun ` —— 普通 C++（CPU 对照代码）。
- ` ```bash ` —— 终端命令。

> 关键区分：真正在 GPU 上跑的代码（cuda/sycl/triton/cutedsl）读者需要自己的硬件，我们给 Compiler Explorer 链接或本地命令；而**概念的可交互性**靠浏览器里能跑的 ` ```python ` 模拟器实现——这是本书"可视化/互动"的核心，每章都要有。

## SVG 概念图规范（照 ch01–03）
- 用 `<div class="concept"><svg viewBox="0 0 宽 高" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif"> … </svg><div class="caption">图 N-x　说明</div></div>` 包裹。
- 配色：橙 `#e06a3b`（强调/CPU对照灰 `#6b7280`）、蓝 `#2b6cb0`（GPU）、绿 `#3f9b5c`。GPU 相关用蓝，CPU 对照用灰。
- 用简单矩形、箭头（用 `<marker>` 定义箭头）、等宽字体标注。图要**真的帮助理解**（画出线程网格、内存层级、访存模式、归约树等），不是装饰。图号用"图 N-1、N-2"（N=章号）。
- 双栏对照用 `<div class="compare"><div class="known"><div class="compare-head">CPU…</div>\n\n```代码```\n\n</div><div class="cpp-side"><div class="compare-head">GPU…</div>\n\n```代码```\n\n</div></div>`。注意 div 内的 markdown 代码块前后要空行。
- 提示框：`<div class="note">`（蓝，补充）、`<div class="warn">`（橙，陷阱）、`<div class="keypoint">`（绿，核心观念）。内部用纯 HTML，`<strong>` 开头做小标题。

## 术语统一
- warp = 32 线程（NVIDIA）；sub-group（SYCL/OpenCL）是其对应物。
- CUDA: grid/block/thread、global/shared/register memory、`__global__`/`__shared__`/`__syncthreads()`。
- SYCL: NDRange/work-group/work-item、`nd_item`、`local_accessor`、`item.barrier()`、sub_group。
- 对应关系（附录 B 有全表）：CUDA block = SYCL work-group = Triton program(一个 program 处理一个 block 的数据) ；CUDA thread = SYCL work-item。
- 硬件：SM（流多处理器）、CUDA 核心、Tensor Core、显存/全局内存、shared memory/L1、L2。

## 质量红线
- 代码正确：CUDA 用 `blockIdx.x*blockDim.x+threadIdx.x` 且带边界检查；host 端 `cudaMalloc`/`cudaMemcpy`/`cudaFree` 完整；SYCL 用现代 `sycl::queue`+`parallel_for`。可运行 python 的"运行结果"必须与代码真实输出一致（自己心算核对）。
- 篇幅：每章正文充实，约 250–450 行 markdown，与 ch01–03 相当。
- 不要编造不存在的 API。不确定的 API 用 `-norun` 并保持通用、符合官方文档习惯。
- 只用公开信息，不含任何公司内部代码或信息。
