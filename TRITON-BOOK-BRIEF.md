# Triton 书 · 规划与章节作者简报（写作时必读）

本文件是《深入 Triton》（暂定，`books/inside-triton/`，slug `inside-triton`）的总体规划 + 写章节时的作者简报。
定位：本仓库第六本书，**独立成册**，风格对标《深入 MLIR》（`books/inside-mlir/`）。

> 状态：**草案**（2026-11-14）。§3 的版本基线已按当天 PyPI / GitHub 查过；§14 列出还没拍板的决定。

## 1. 定位与读者

- **读者**：资深工程师，熟练使用 Python 和 PyTorch，懂多线程、缓存、SIMD；可能写过一点 CUDA，也可能完全没碰过 GPU。**不假设读过本仓库的其它书**：GPU 心智模型、够用的 MLIR 知识都在本书里从头讲。
- **痛点**（全书要解决的四件事，每章都要对得上其中之一）：
  1. 官方 tutorial 能跑通，但**写出来的 kernel 比 cuBLAS / FlashAttention 慢一截，不知道差在哪**。
  2. 改了 `BLOCK_SIZE`、`num_warps`、`num_stages` 性能乱跳，**没有能预测结果的心智模型**，只能盲调。
  3. 编译器是黑盒：TTIR / TTGIR 看不懂，**layout 是什么、`convert_layout` 为什么贵**，没人讲清。
  4. 生态在分层（往上是 Helion，往下是 Gluon / TLX，旁边是 CuTe DSL、TileLang），**不知道什么时候该离开 Triton、该往哪一层走**。
- **范围**：Triton 是主体（约 75%）；Triton 生态的上下层 **Gluon、Helion** 各有独立章节（约 15%）；其它 DSL（CUDA、CuTe DSL、TileLang、Pallas、TLX）**只当对照镜子**（约 10%），不写成平铺式综述。
- **规格**：30 章 + 4 附录，六部分。篇幅不设上限，判据同 MLIR 书：「这一段删掉，读者会不会少懂一件事」。
- **与其它书的关系**：内容可以与《看得见的 GPU 编程》（Triton 两章 + softmax / attention 对照）以及《深入 MLIR》第 26 章重叠，**按本书视角重写，不靠交叉引用省篇幅**。交叉引用只作为「延伸阅读」，不能当成前置要求。

## 2. 动手前必做

1. **通读 `books/inside-mlir/chapters/ch01.md`、`ch05.md`、`ch26.md`**，这是声音、结构、「源码为锚」写法的样板。ch26 就是 Triton，但那一章站在 MLIR 用户的角度；本书要站在 kernel 作者的角度。
2. 读 `STYLE.md` 了解排版约定（**语气不适用**，那份是青少年书的语气，本书的语气见 §4）。
3. 读 `books/inside-triton/book.json` 确认章节顺序，交叉引用必须准确。
4. 读 §5 的 **minillm 贯穿项目规格**：每章都在同一个项目上增量推进。
5. **手边备一份可 grep 的源码树**：
   ```bash
   # Triton 主仓（tt/ttg 方言、pass、Python 前端、Gluon、tutorials）
   git clone --filter=blob:none --sparse https://github.com/triton-lang/triton.git ~/repos/triton
   cd ~/repos/triton && git checkout v3.9.0
   git sparse-checkout set include lib python/triton python/tutorials third_party/nvidia third_party/amd third_party/proton
   # Helion（ch26）
   git clone --depth=1 --branch v1.5.2 https://github.com/pytorch/helion.git ~/repos/helion
   # Intel XPU 后端（ch23；只用公开仓库）
   git clone --filter=blob:none --sparse --depth=1 --branch v3.9.0 \
     https://github.com/intel/intel-xpu-backend-for-triton.git ~/repos/intel-xpu-triton
   ```
   `~/repos/triton` 里已有一份 `84fa223`（2026-09-20，MLIR 书 ch26 用的），**写本书要切到 `v3.9.0` 标签**。
   正文里每个 `tl.*` / `gl.*` / `hl.*` API、环境变量、pass 名都要在这几份树里 grep 确认过再写。

## 3. 版本基线（2026-11-14 查过）

| 组件 | 基线 | 发布日期 | 备注 |
| --- | --- | --- | --- |
| Triton | **3.9.0** | 2026-10-27 | `third_party/` 下只有 `nvidia`、`amd`（以及 `proton`、`f2reduce`）两个树内后端 |
| Gluon | 随 Triton 3.9.0 | — | **仍在 `triton.experimental.gluon`**，API 不承诺稳定；官方教程 `python/tutorials/gluon/01–14` |
| Helion | **1.5.2** | 2026-11-01 | 已到 1.x；要求 Python 3.10–3.14、PyTorch ≥ 2.9、Triton ≥ 3.5 |
| intel-xpu-backend-for-triton | **v3.9.0** | 2026-10-29 | 树外后端，跟 Triton 主线同版本号；有 Gluon 测试的 skiplist（说明已部分支持） |
| PyTorch | 待定 | — | 选与 Triton 3.9.0 配套的那一版，批次 0 里定（§14） |
| Compiler Explorer | **Triton 3.7.0**（NVIDIA / AMD） | — | `https://godbolt.org/api/compilers/triton`：`triton_nvidia_370`、`triton_amd_370`，**比基线落后两个小版本** |

### 3.1 验证硬件（2026-11-14 定）

两台机器，都**不在**写作用的这台 WSL 上（WSL 里没有 GPU）：

| 机器 | 架构 | 角色 |
| --- | --- | --- |
| **NVIDIA A10** | Ampere，sm_86，24 GB GDDR6 | **主力平台**：正文的性能数字以它为准；真机 PTX；Gluon 的 Ampere 部分（`gluon.language.nvidia.ampere` 下的 `async_copy`、`mbarrier`，官方教程 01–03） |
| **Intel Data Center GPU Max 1100** | PVC（Xe-HPC） | Intel 后端（ch23）、SPIR-V；示例代码的可移植性验证；`gluon.language.intel` 的 `IntelDPASLayout` |

两台都**跑不了**的内容：TMA、wgmma / tcgen05、warp specialization、cluster / multi-CTA（需要 Hopper 及以上），fp8（需要 sm_89 及以上）和 mxfp，以及 TLX。受影响的是 **ch15、ch16 的大部分、ch25 的后半**。处理方式见 §14 第 1 项。

由此带来的写作约束：
- 示例代码**与设备无关**：统一写 `DEVICE = triton.runtime.driver.active.get_active_torch_device()`（以 3.9.0 源码核对过的写法为准），不写死 `"cuda"`。
- 每个代码块都要标注它所需的最低架构（§7.3），正文里的性能数字要写明是 A10 还是 Max 1100 测的。
- 批次 0 要在两台机器上各搭一遍环境，核对驱动版本能否跑 Triton 3.9.0。GPU 书当时那台 NVIDIA 机器上是 nvcc 11.5；Triton 的 wheel 自带 ptxas，nvcc 版本不要紧，关键是驱动。
- **A10 的驱动约束**（2026-11-15 实测 `probe-env.py`）：驱动 550.163.01，最高支持 CUDA 12.4，**暂不升级**。PyTorch 与 Triton 的配套关系是：torch 2.15.0 配 triton ~=3.9.0，但只有 CUDA 13.2 版，要求驱动 ≥ 580；torch 2.14.1 配 triton ~=3.8.0，有 CUDA 12.6 版。**已实测**（2026-11-15）：torch 2.14.1+cu126，再用 `--no-deps` 强装 triton 3.9.0，在 550 驱动上通过了 vector add、`tl.dot`（`#ttg.nvidia_mma`）、`do_bench`（x*2 约 473 GB/s，标称 600 GB/s）、解释器、Gluon（含 `nvidia/ampere`）和 proton 的检查。**基线保持 Triton 3.9.0。** 复测时 `torch.compile`（Inductor）和 Helion 1.5.2 的固定 Config kernel 也通过了，`probe-env.py` 14/14 全过。**A10 环境就此定下。**
  - Max 1100 不需要绕：torch 2.15.0+xpu 自带依赖 `triton-xpu~=3.9.0`，正好与基线一致。
  - A10 上的装法：`uv pip install --no-config torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126`，再装 `helion packaging`，最后 `uv pip install --no-config triton==3.9.0 --no-deps`。
  - **坑 1**：Helion 1.5.2 运行时要导入 `packaging`，但它的依赖声明里漏写了这个包，要手动装。写进附录 A。
  - **坑 2**：Triton 3.9 在未设置 `TRITON_DEFAULT_FP_FUSION` 时会发警告：3.9 默认**开启** FP fusion（乘加融合成 FMA），3.10 将改为默认关闭。这会影响数值结果的逐位一致性，写进 ch07（精度）、ch09（数值对比）和附录 C。

易变点，写到哪里就要提醒读者去核对：

- Gluon 的模块路径、layout 类名、架构子包（`nvidia/{ampere,hopper,blackwell,rubin}`、`amd/{cdna3,cdna4,cdna5,rdna3,rdna4,gfx1250}`）都可能改。
- 访存 API 正在迁移：指针 → block pointer（`tl.make_block_ptr`）→ tensor descriptor。写之前查清 3.9.0 里哪种是推荐写法、哪种已标弃用。
- 调试和 dump 用的环境变量（`TRITON_INTERPRET`、`MLIR_ENABLE_DUMP`、`TRITON_KERNEL_DUMP`、`TRITON_CACHE_DIR` 等）一律 grep `python/triton/knobs.py`（或对应位置）核对，不凭记忆写。
- Helion 的 `Config` 字段（`block_sizes`、`range_warp_specializes`、`pid_type`、`indexing` 等）以 1.5.2 的 `helion/runtime/config.py` 为准。
- 不确定的 API 一律用 `-norun` 代码块，并写得通用；**绝不编造不存在的 API、pass 或环境变量**。

## 4. 声音与语气

- 用「你」称呼读者，像一位资深同事在白板前讲解：**专业、精炼、有洞察**，不啰嗦。
- 核心手法是**对照 + 追问「为什么」**：
  - 第一、二部分以读者已会的 **PyTorch eager / NumPy / CPU 多线程**为锚点；
  - 第三部分以「**预测 → 测量 → 解释**」为节奏：先让读者猜改一个参数会快还是慢，再测，再用硬件模型解释；
  - 第四部分以 **Python 源码 ↔ IR** 对照为主线：每一段 IR 都要能指回它来自哪一行 `tl.*`。
- 允许英文术语，首次出现写成「中文（English）」并加粗，例如 **程序实例（program instance）**、**软件流水（software pipelining）**。
- 承认现实：Triton 有写不快、不稳、文档缺失的地方。该直说「这里 Triton 不如手写 CUDA」「这个 API 下个版本可能就没了」的地方就直说。

## 5. 贯穿项目：minillm（第 4 章起，每章增量）

用一个**简化的 decoder-only Transformer 层**作为贯穿项目，所有 kernel 都用 Triton 实现，最后串起来跑一个真实的前向。选它的理由：读者最关心的就是这类 kernel；而它同时覆盖了**逐点、归约、GEMM、attention、路由 / 分组**五类计算，每一类都对应一种典型的性能瓶颈。

| 章 | 新增 kernel / 功能 | 主要演示什么 |
| --- | --- | --- |
| ch04 | `add`、`silu_mul`（逐点） | launch、grid、mask，以及 JIT 缓存 |
| ch06 | `rmsnorm`、`softmax` | 行级归约、单遍 / 两遍算法、行太长时怎么办 |
| ch07 | `matmul`（朴素 tile 版） | `tl.dot`、累加精度、正确性基准 |
| ch08 | `matmul` 改用 tensor descriptor | 三种访存写法的对比 |
| ch10 | 全部包成 `torch.library` custom op | 接入 autograd 与 `torch.compile` |
| ch12–15 | `matmul` 逐步优化到接近 cuBLAS | L2 swizzle、流水、autotune、persistent、warp specialization |
| ch16 | `matmul_fp8`、block-scaled | 低精度 |
| ch17 | `flash_attn_fwd` | 把第三部分全部用上 |
| ch21–22 | （不加 kernel）读 ch17 的 TTGIR，定位并修掉一次多余的 `convert_layout` | 用编译器知识解释性能 |
| ch25 | 用 Gluon 重写 `matmul` | 把控制权拿回来，和 Triton 版本对比 IR 与性能 |
| ch26 | 用 Helion 重写 `rmsnorm` + `matmul` | 看生成的 Triton 代码，比较自动调优与手写 |
| ch28 | `moe_grouped_gemm` + 路由 permute | 不规则形状、grouped GEMM |
| ch29 | `decode_attn`（paged、split-k） | 推理场景：小 batch、长 KV |
| ch30 | 一层前向串起来 | 与 PyTorch eager / Inductor 做端到端对比 |

**代码落地**：`books/inside-triton/code/minillm/`，一个能 `pip install -e .` 的小包，结构是 `minillm/kernels/*.py` + `tests/`（pytest，与 PyTorch 参考实现对比数值）+ `bench/`（`triton.testing.do_bench`）。正文只贴关键片段，完整代码放在这里；`verify/verify-minillm.sh` 跑全部测试。

## 6. 每章固定结构

1. `# 第 N 章 标题`（与 `book.json` 完全一致）
2. `## 这一章想让你带走什么` + 紧跟 `<div class="goal-box"><p>…纯 HTML 段落…</p></div>`
3. 正文：**问题 → 机制 → 源码 / IR 锚点 → 动手**。每章至少：
   - **1 张概念 SVG 图**（§8）
   - **1 处真实锚点**：Triton 源码路径 + 类名 / 函数名，或一段**真实 dump 出来的** TTIR / TTGIR
   - **1 段可交互内容**：浏览器可运行的 `python` 模拟器，或带 CE 按钮的 `triton` 块
4. `## 动手：…` 一节：给命令和**真机跑出来的**输出（「运行结果：」块），并引导读者改一处再跑。性能数字必须注明 GPU 型号与版本。
5. `---` 后写 `**这一章的要点**`，用要点列表回顾，并**预告下一章**。
6. 第 4 章起，末尾加一行 **minillm 进度**。

## 7. 代码块约定

### 7.1 现有 `build.js` 支持

| 围栏 | 现状 |
| --- | --- |
| ` ```triton ` / ` ```triton-norun ` | 已有（GPU 书在用）：只高亮 + `pip install triton torch` 本地命令，**没有 CE 按钮** |
| ` ```cutedsl ` / ` ```cutedsl-norun ` | 已有 |
| ` ```cuda ` / ` ```cuda-norun ` | 已有，带 CE 按钮 |
| ` ```python ` | 浏览器真运行（Pyodide），只能用标准库 |
| ` ```python-norun ` / ` ```bash ` | 已有 |
| ` ```mlir-norun ` | 已有，带 MLIR 高亮。用于贴 TTIR / TTGIR |

### 7.2 批次 0 要加的

| 围栏 | 用途 | 按钮 / 底部命令 |
| --- | --- | --- |
| ` ```triton-ir ` | 完整的 Triton 程序，点开在 CE 里看 TTIR / TTGIR / PTX（**不需要 GPU**） | 「🔬 在 Compiler Explorer 看 IR」，编译器 id 默认 `triton_nvidia_370`；块首 `# CE: triton_amd_370` 可切换到 AMD |
| ` ```gluon ` / ` ```gluon-norun ` | Gluon kernel | 只高亮 + 本地命令，标注所需架构（如「需 Hopper 及以上」） |
| ` ```helion ` / ` ```helion-norun ` | Helion kernel | 只高亮 + `pip install helion` 本地命令 |
| ` ```ttgir-norun ` | 可选：如果 `mlir-norun` 的高亮对 `#ttg.blocked<{…}>` 这类 encoding 属性效果不好，就在 `mlir-syntax.js` 里加几条规则 | 无 |

批次 0 必须先实测 CE 的 Triton 接口：它对源码有什么要求（是否需要 `triton.compile` 驱动代码、怎么选看哪一级 IR），以及 3.7.0 和 3.9.0 的 IR 差异会不会影响正文。

### 7.3 随书校验脚本

```bash
books/inside-triton/verify/verify-triton.sh            # 抽 triton / gluon / helion 块，在真机上逐个跑
books/inside-triton/verify/verify-triton.sh ch12 ch13  # 只校验指定章
CE=1 books/inside-triton/verify/verify-triton.sh       # 没有 GPU：triton-ir 块走 godbolt，只验证能编译
books/inside-triton/verify/verify-minillm.sh           # pytest 全部 minillm kernel
books/inside-triton/verify/verify-outputs.py           # 正文贴的运行结果与实跑逐字节比对（性能数字除外，只核量级）
```

每个块按它所需的最低架构打标签（`sm80` / `sm90` / `sm100` / `cdna3` / `xpu`），在不满足条件的机器上自动跳过，并报告「跳过 N 块」，**不能算作通过**。

## 8. SVG 概念图规范

沿用 MLIR 书的写法：`<div class="concept"><svg viewBox="…" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif"> … </svg><div class="caption">图 N-x　说明</div></div>`

- 配色：**Triton / 块级抽象用蓝 `#2b6cb0`**，**硬件 / 线程级用灰 `#6b7280`**，强调或变换动作用橙 `#e06a3b`，成功路径或结论用绿 `#3f9b5c`。
- 本书必画的图（不画装饰图）：
  - program → block 的映射
  - 内存层级与带宽数字
  - roofline
  - mask 覆盖的边界块
  - GEMM 的 tile 分解与 L2 swizzle 的访问顺序
  - 流水线的 stage 时间线
  - warp specialization 的生产者 / 消费者分工
  - layout 把张量元素分给 thread / warp 的方式
  - `convert_layout` 走 shared memory 的数据流
  - 编译管线各级 IR
  - Helion → Triton → Gluon 的抽象层次塔
- 双栏对照用 `<div class="compare">…`，提示框用 `note` / `warn` / `keypoint`，规则同 MLIR 书 §8。

## 9. 术语统一

- **程序实例（program instance）**：Triton 里一个 `program_id` 对应的执行单位，对应 CUDA 的 thread block。正文统一说「program」，**不要**说成「线程」。
- **块（block）**：Triton 张量的一块 tile，即 `tl.arange(0, BLOCK)` 出来的那个形状。注意它和 CUDA 的 thread block 是两回事，**首次出现时必须讲清楚这个撞名**。
- **布局（layout / encoding）**：TTGIR 里张量的分布属性。`#ttg.blocked`、`#ttg.slice`、`#ttg.nvidia_mma`、`#ttg.dot_op`、`#ttg.shared`、linear layout 一律写源码里的原名。
- 方言前缀写全：`tt.load`、`tt.dot`、`ttg.convert_layout`、`ttng.*`（NVIDIA 专有）、`ttag.*` / `amdgpu.*`（AMD，以源码为准）。
- 硬件代际用架构名加 compute capability，例如「Hopper（sm_90）」「Blackwell（sm_100）」「CDNA3（MI300）」。
- Gluon 写 `gl.*`，Helion 写 `hl.*`，Triton 写 `tl.*`，前缀不混用。
- 附录 B 给 CUDA ↔ Triton ↔ Gluon ↔ Helion 的概念对照全表。

## 10. 章节规划

### 第一部分 · 心智模型（ch01–04）
- **ch01 为什么是 Triton：从 PyTorch eager 到手写 kernel 之间的鸿沟。** 从一个真实的慢算子切入：eager 下 softmax 读写显存 N 次，融合以后只要 1 次。然后讲清 CUDA、Triton、`torch.compile` 各站在哪个位置。最后给出本书的地图，即 §1 那四个痛点。
- **ch02 够用的 GPU 硬件模型。** SM、warp、寄存器、shared memory、L2、HBM，以及每一级的容量和带宽数字。讲清「带宽墙」与「算力墙」，引出 roofline 的直觉（正式内容在 ch11）。用浏览器模拟器演示 warp 调度如何隐藏延迟。
- **ch03 块级编程：一个 program 处理一块数据。** 和 CUDA SIMT 双栏对照：同一个向量加法，一边是逐线程写法，一边是逐块写法。讲清 Triton 替你做了什么（线程映射、合并访问、shared memory 分配），又拿走了什么（warp 级控制）。block 撞名问题在这里讲透。
- **ch04 第一个 kernel：vector add 的完整生命周期。** `@triton.jit`、grid、`tl.program_id`、mask、`tl.constexpr`、JIT 特化与缓存目录。minillm 起步。

### 第二部分 · 写对 Triton（ch05–10）
- **ch05 tl 语言核心。** 指针算术、广播、`load` / `store` 的 `mask` 与 `other`、`tl.where`、类型提升规则，以及 2D 张量的 stride 计算。
- **ch06 归约与逐行算子。** softmax、rmsnorm：一行放得进一个 block 和放不进时的两种写法、online softmax。
- **ch07 `tl.dot` 与第一个 GEMM。** tile 分解、累加器精度、TF32 / fp16 / bf16 的误差，以及「为什么结果和 PyTorch 差 1e-3」。
- **ch08 访存的三种写法。** 裸指针、block pointer、tensor descriptor 各有什么取舍，3.9.0 推荐哪种，以及它们在 IR 里长得有什么不同（预告第四部分）。
- **ch09 调试与测试。** `TRITON_INTERPRET=1`、`tl.device_print`、`tl.device_assert`；数值对比的容差怎么定；典型错误（越界、mask 漏写、constexpr 误用）及其报错样子。
- **ch10 接入 PyTorch。** `torch.library.triton_op` / custom op、autograd、在 `torch.compile` 里调用，以及顺便看一眼 Inductor 自己生成的 Triton 代码。

### 第三部分 · 写快 Triton（ch11–17）
- **ch11 性能模型与测量。** roofline、算术强度、`triton.testing.do_bench` 的正确用法（预热、L2 清理）、proton profiler。
- **ch12 GEMM 优化（一）：tile 尺寸与执行顺序。** `BLOCK_M/N/K`、`num_warps` 对寄存器压力和占用率的影响，L2 swizzle（grouped ordering）。
- **ch13 GEMM 优化（二）：软件流水。** `num_stages` 在硬件上到底发生了什么（异步拷贝和多缓冲），以及 stage 为什么不是越多越好（shared memory 预算）。
- **ch14 autotune。** 配置空间、`key`、`prune_configs_by`、缓存，以及调优本身的代价。对照：Helion 的自动搜索空间要大得多（预告 ch26）。
- **ch15 新架构特性在 Triton 里怎么用。** TMA、wgmma / tcgen05、persistent kernel、warp specialization。重点讲清**哪些由编译器自动做，哪些要你给提示**，并注明各自需要的架构。
- **ch16 低精度。** fp8、block-scaled（mxfp4 / mxfp8）GEMM、`tl.dot_scaled`。
- **ch17 FlashAttention 前向。** online softmax 加上两次 GEMM 融合，把本部分讲过的手段全部用上，并与 PyTorch SDPA 对比。

### 第四部分 · 看懂 Triton 编译器（ch18–23）
- **ch18 够用的 MLIR。** op、dialect、region、pass、lowering，只讲读 Triton IR 需要的那部分，自成一体。可以和 MLIR 书重叠。
- **ch19 编译管线全景。** Python AST → TTIR → TTGIR → LLVM IR → PTX / AMDGCN / SPIR-V；`compile()` 的阶段、缓存，以及怎么 dump 每一级 IR。从这一章起开始大量使用 `triton-ir` 块（CE，不需要 GPU）。
- **ch20 Layout 系统（一）。** blocked、slice、mma、dot_operand、shared，各自描述了什么，为什么 GPU 编译器需要这层抽象。配一个浏览器模拟器：输入 layout 参数，画出每个元素归哪个 thread 和 warp。
- **ch21 Layout 系统（二）：linear layout 与 `convert_layout`。** 统一的线性代数表示；`convert_layout` 什么时候走 shuffle、什么时候走 shared memory，以及它的代价。锚 `include/triton/Tools/LinearLayout.h`。
- **ch22 关键 pass 与性能定位。** coalesce、accelerate-matmul、remove-layout-conversions、pipeliner、warp specialization 等 pass；演示怎么读 TTGIR 定位 ch17 里一个真实的性能问题。
- **ch23 三个后端。** NVIDIA、AMD 两个树内后端加 Intel XPU 树外后端：各自在 TTGIR 之后做了什么，同一个 kernel 换后端会遇到哪些性能可移植性问题（warp 宽度、矩阵指令形状、shared memory 大小）。**Intel 部分只用公开仓库**。

### 第五部分 · Triton 的上一层与下一层（ch24–27）
- **ch24 Gluon（一）：为什么需要再往下一层。** Triton 自动推导的 layout 和流水什么时候不够好；Gluon 把哪些决定交还给你（layout、shared memory、warp 分工）；它和 Triton 共用同一套后端。layout 显式化以后，ch20–21 讲的东西在这里变得直观。
- **ch25 Gluon（二）：手写高性能 GEMM。** async copy、TMA、mbarrier、warp specialization；用 Gluon 重写 minillm 的 `matmul`，与 Triton 版本对比 IR 和性能。小节对照 **TLX**（Meta 的 Triton 低层扩展）：同一个问题的另一种解法。
- **ch26 Helion：带 tile 的 PyTorch。** `@helion.kernel`、`hl.tile`、autotune 搜索什么；看它生成的 Triton 代码；可以固定 `Config`；多后端（Triton 是默认，另有 TileIR、CuTe、Pallas 等）。讨论什么时候用 Helion、什么时候退回手写 Triton。
- **ch27 同一个 kernel 的另几种写法。** 用 CUDA、CuTe DSL、TileLang、Pallas 各写一遍 ch12 的 GEMM 或 ch06 的 softmax，比较每种 DSL 把哪些控制权交给用户。结论：一张「什么时候离开 Triton、往哪走」的决策表。**本章只做对照，不展开教学**。

### 第六部分 · 实战（ch28–30）
- **ch28 MoE：grouped GEMM 与路由。** 不规则的 expert 负载、permute / unpermute 融合、grouped GEMM 的调度。
- **ch29 推理 kernel：decode attention。** paged KV cache、split-k / flash-decoding；小 batch 下为什么瓶颈在带宽。
- **ch30 收官：串起一层，再谈 Triton 还没解决的问题。** minillm 一层前向的端到端对比（eager / Inductor / 本书 kernel）。最后诚实收尾：编译时间、版本不稳、跨后端性能可移植、调试体验。

### 附录
- **A 环境搭建**：Triton、PyTorch、Helion 与 CUDA / ROCm / Intel XPU 的版本配套；CE 的用法；没有 GPU 时能做什么。
- **B 概念对照表**：CUDA ↔ Triton ↔ Gluon ↔ Helion（以及 CuTe DSL 一列）。
- **C 环境变量与调试开关速查**：全部以 3.9.0 源码为准。
- **D 源码导览地图**：想看某个机制，去 `triton/` 的哪个目录。

## 11. 取舍：不单独成章的话题及其归属

- **Triton 语言参考大全**：不写，读者去官方文档查；本书的价值在机制与判断。
- **反向传播 kernel**（FlashAttention bwd 等）：ch17 末尾一节讲思路，代码放 minillm 里作为延伸。
- **分布式 / 通信 kernel**（Triton-distributed 等）：ch30 一段指路。
- **TileIR 后端**（`Triton-to-tile-IR`）：ch23 末尾和 ch26 Helion 多后端处各一段。
- **ThunderKittens、cuTile 等**：ch27 的决策表里列出来，不写代码。
- **自定义 Triton 后端**：ch23 一段讲接口在哪里，附录 D 指路。

## 12. 写作顺序与批次

PR 节奏沿用 MLIR 书：**全书写完再开一个 PR**（若有不同意见见 §14）。

| 批次 | 内容 | 前置 |
| --- | --- | --- |
| 0 | `build.js` 加 `triton-ir` / `gluon` / `helion` 围栏，实测 CE 的 Triton 接口；`verify/` 脚本骨架；附录 A；`book.json`；minillm 包骨架 | 定下 §14 的硬件与版本 |
| 1 | ch01–04 + 附录 B | 批次 0 |
| 2 | ch05–10 | 批次 1 |
| 3 | ch11–17 | 批次 2；ch15、ch16 要先定下 §14 第 1 项 |
| 4 | ch18–23 | 批次 3（ch22 要用到 ch17 的 kernel） |
| 5 | ch24–27 | 批次 4；ch25 后半同 §14 第 1 项 |
| 6 | ch28–30 + 附录 C、D + 全书交叉引用校对 | 批次 5 |

每个批次结束都要跑 `node build.js inside-triton`，并跑对应的 verify 脚本。

## 13. 质量红线

- **代码必须在真机上跑过**：每个 `triton` / `gluon` / `helion` 完整块都要在标注的架构上跑通；「运行结果」贴**真实输出**。
- **性能数字必须可复现**：注明 GPU 型号、驱动、Triton 版本、输入形状；只用于说明趋势和量级，不做营销式的「快 N 倍」。
- **IR 必须是真 dump 出来的**：不手写「想象中的 TTGIR」；可以节选，但要标明是节选。
- **API、环境变量、pass 名不得编造**，一律以 §3 的基线源码为准。
- 可运行的 `python` 模拟器：完整、可独立运行、只用标准库；「运行结果」与真实输出逐字节一致。
- **只用公开信息**：不含任何公司内部代码、内部项目名或未公开信息。Intel 相关内容只引用 `intel/intel-xpu-backend-for-triton` 公开仓库与公开文档。
- 课后题答案必须先自己跑一遍。

## 14. 待定决定

1. **Hopper 那几章怎么验证。** 硬件已定（§3.1），A10 和 Max 1100 合起来大约能真机覆盖 85%。剩下的 ch15、ch16 大部分、ch25 后半有两种做法：(a) 写到批次 3 时短期租一台云上的 H100 / B200，集中跑完；(b) 用 CE 看 IR，代码块写成 `-norun`，并明确标注「需 Hopper，本书未真机验证」。倾向 (a)，到批次 3 再定。
2. **CE 版本落后。** CE 目前最高只有 Triton 3.7.0，基线是 3.9.0。建议：正文按 3.9.0 写，`triton-ir` 块用 CE 上最新的版本，两者 IR 有差异的地方用 `warn` 框提醒。等 CE 更新以后，把默认编译器 id 改一处即可。
3. **PyTorch 版本**：批次 0 里选与 Triton 3.9.0 配套的那一版。
4. **书名**：暂定《深入 Triton》（Inside Triton），与《深入 MLIR》成系列；另一个备选是《Triton：从写对到写快》。
5. **分支与 PR**：在新分支 `inside-triton` 上做；PR 节奏默认全书一个。
