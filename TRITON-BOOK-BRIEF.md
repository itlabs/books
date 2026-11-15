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

两台机器。2026-11-15 起，**写作也挪到 Max 1100 那台机器上**，Intel 这条线可以边写边测；A10 在另一台机器上，脚本写好后由作者本人去跑，再把输出贴回来：

| 机器 | 架构 | 角色 |
| --- | --- | --- |
| **NVIDIA A10** | Ampere，sm_86，24 GB GDDR6 | **主力平台**：正文的性能数字以它为准；真机 PTX；Gluon 的 Ampere 部分（`gluon.language.nvidia.ampere` 下的 `async_copy`、`mbarrier`，官方教程 01–03） |
| **Intel Data Center GPU Max 1100** | PVC（Xe-HPC） | Intel 后端（ch23）、SPIR-V；示例代码的可移植性验证；`gluon.language.intel` 的 `IntelDPASLayout` |

两台都**跑不了**的内容：TMA、wgmma / tcgen05、warp specialization、cluster / multi-CTA（需要 Hopper 及以上），fp8（需要 sm_89 及以上）和 mxfp，以及 TLX。受影响的是 **ch15、ch16 的大部分、ch25 的后半**。处理方式见 §14 第 1 项。

由此带来的写作约束：
- 示例代码**与设备无关**：统一写 `DEVICE = triton.runtime.driver.active.get_active_torch_device()`（以 3.9.0 源码核对过的写法为准），不写死 `"cuda"`。
- 每个代码块都要标注它所需的最低架构（§7.3），正文里的性能数字要写明是 A10 还是 Max 1100 测的。
- Python 环境一律用 **uv**（`uv venv`、`uv pip install`），不用 `python -m venv` 加 pip。给读者看的安装命令是否也统一成 uv，到写附录 A 时再定（§14）。
- 批次 0 要在两台机器上各搭一遍环境，核对驱动版本能否跑 Triton 3.9.0。GPU 书当时那台 NVIDIA 机器上是 nvcc 11.5；Triton 的 wheel 自带 ptxas，nvcc 版本不要紧，关键是驱动。
- **A10 的驱动约束**（2026-11-15 实测 `probe-env.py`）：驱动 550.163.01，最高支持 CUDA 12.4，**暂不升级**。PyTorch 与 Triton 的配套关系是：torch 2.15.0 配 triton ~=3.9.0，但只有 CUDA 13.2 版，要求驱动 ≥ 580；torch 2.14.1 配 triton ~=3.8.0，有 CUDA 12.6 版。**已实测**（2026-11-15）：torch 2.14.1+cu126，再用 `--no-deps` 强装 triton 3.9.0，在 550 驱动上通过了 vector add、`tl.dot`（`#ttg.nvidia_mma`）、`do_bench`（x*2 约 473 GB/s，标称 600 GB/s）、解释器、Gluon（含 `nvidia/ampere`）和 proton 的检查。**基线保持 Triton 3.9.0。** 复测时 `torch.compile`（Inductor）和 Helion 1.5.2 的固定 Config kernel 也通过了，`probe-env.py` 14/14 全过。**A10 环境就此定下。**
  - Max 1100 不需要绕：torch 2.15.0+xpu 自带依赖 `triton-xpu~=3.9.0`，正好与基线一致。**已实测**（2026-11-15）：`probe-env.py --helion` 14/14 全过，`tl.dot` 的 layout 是 `#ttig.dpas`（`threadsPerWarp = 16`），`do_bench` 的 x*2 约 531 GB/s。环境变量见 `verify/env-xpu.sh.example`，踩过的坑见 §15.2。**Max 1100 环境就此定下。**
  - A10 上的装法：`uv pip install --no-config torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126`，再装 `helion packaging`，最后 `uv pip install --no-config triton==3.9.0 --no-deps`。
  - **坑 1**：Helion 1.5.2 运行时要导入 `packaging`，但它的依赖声明里漏写了这个包，要手动装。写进附录 A。
  - **坑 2**：Triton 3.9 在未设置 `TRITON_DEFAULT_FP_FUSION` 时会发警告：3.9 默认**开启** FP fusion（乘加融合成 FMA），3.10 将改为默认关闭。这会影响数值结果的逐位一致性，写进 ch07（精度）、ch09（数值对比）和附录 C。

易变点，写到哪里就要提醒读者去核对：

- Gluon 的模块路径、layout 类名、架构子包（`nvidia/{ampere,hopper,blackwell,rubin}`、`amd/{cdna3,cdna4,cdna5,rdna3,rdna4,gfx1250}`）都可能改。
- 访存 API 正在迁移：指针 → block pointer（`tl.make_block_ptr`）→ tensor descriptor。**已核实（2026-11-15）：3.9.0 里 `tl.make_block_ptr` 和 `tl.advance` 已经移除**，只剩一句 `raise NotImplementedError("Block pointers have been removed in favor of the tensor descriptor API")`（`python/triton/language/core.py`）。推荐写法是 tensor descriptor。
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
| ` ```mlir-norun ` | 已有，带 MLIR 高亮。本书贴 TTIR / TTGIR 改用下面的 `ttir-norun` / `ttgir-norun` |

### 7.2 批次 0 加的（2026-11-15 已加进 `build.js`，并用浏览器实测过）

| 围栏 | 用途 | 按钮 / 底部命令 |
| --- | --- | --- |
| ` ```triton ` | **本书里**：完整、能在真机上跑的 Triton 程序 | 只高亮；页脚是 `python main.py` 加上块首 `# arch:` 标明的平台（GPU 书的 `triton` 块不受影响，仍是原来的页脚） |
| ` ```triton-ir ` | 完整的 Triton **或 Gluon** 程序，点开在 CE 里看各级 IR（**不需要 GPU**） | 「🔬 在 Compiler Explorer 看 IR」：打开「源码 / 编译器 / Device Viewer」三栏，Device Viewer 直接停在要看的那一级 |
| ` ```gluon ` / ` ```gluon-norun ` | Gluon kernel | 只高亮 + 本地命令 + 平台 |
| ` ```helion ` / ` ```helion-norun ` | Helion kernel | 只高亮 + `pip install helion torch` + 平台 |
| ` ```ttir-norun ` / ` ```ttgir-norun ` | 贴真 dump 出来的 TTIR / TTGIR | 无按钮。高亮在 MLIR 之上另加了三条规则（`mlir-syntax.js` 里的 `ttgir` 语言，`mlir` 语言本身没动）：layout 别名与 encoding（`#blocked1`、`#ttg.nvidia_mma`、`#ttig.dpas`）加粗着色，方言 op（`tt.load`、`ttg.convert_layout`）着色，layout 参数（`sizePerThread`、`kWidth`、`order`）着色 |

**块首注释**（普通 Python 注释，留在代码里；`build.js` 和将来的 verify 脚本读同一行）：

- `# arch: sm80 xpu`：这个块要在哪些平台上跑，空格分隔，满足其一即可。取值 `any` / `sm80` / `sm86` / `sm90` / `sm100` / `cdna3` / `xpu`。**`triton`、`gluon`、`helion` 块不写这一行，构建直接报错。**
- `# CE: triton_amd_370`，或 `# CE: triton_nvidia_370 --arch 90`：只对 `triton-ir` 块，用来换编译器、传 wrapper 参数。不写时 NVIDIA 默认 `--arch 86`（A10，与真机一致），AMD 用 CE 的默认值 `gfx942`。
- `# IR: ttir`：只对 `triton-ir` 块，指定 Device Viewer 一打开停在哪一级，默认 `ttgir`。取值 `source` / `ttir` / `ttgir` / `llir` / `ptx` / `sass` / `amdgcn`。

**CE 的 Triton 接口实测结论**（读的是 compiler-explorer 仓库的 `etc/scripts/triton_wrapper.py`，并用 API 和无头浏览器实测）：

1. **源码必须真的启动一次 kernel**。CE 把整个脚本 import 一遍，只编译被启动的 kernel；只定义不启动，CE 返回 `<No output file>`。不需要 `triton.compile` 一类的驱动代码。
2. **张量是 `FakeTensor`，driver 是 `MagicMock`**：只编译、不运行。所以 **`triton-ir` 块写 `device="cuda"`**（ROCm 版 torch 也叫 `"cuda"`）。§3.1 要求的 `DEVICE = triton.runtime.driver.active.get_active_torch_device()` 在 CE 上返回的是 MagicMock，`torch.randn(device=DEVICE)` 会直接报错。**§3.1 的「不写死 `"cuda"`」只适用于真机上跑的 `triton` / `gluon` / `helion` 块，`triton-ir` 块例外**。
3. **各级 IR 在 Device Viewer 面板里**，名字是 `<kernel> [stage N].<ir>`；主面板只显示 PTX / AMDGCN。stage 的数目和编号因组合而异：NVIDIA 上的 Triton 是 source、ttir、ttgir、llir、ptx、sass 六级；AMD 没有 sass，第 5 级是 amdgcn；**Gluon 没有 ttir**，所以 ttgir 是第 2 级。`build.js` 会现算编号。
4. CE 的分享链接有两种格式。clientstate 格式打不开 Device Viewer，所以 `triton-ir` 用的是另一种：`https://godbolt.org/#` 后面跟完整的 GoldenLayout JSON。
5. wrapper 参数只有 `--backend`、`--arch`、`--warp_size` 三个。用户参数追加在默认参数后面，argparse 取最后一次出现的值，所以 `--arch 86` 能覆盖默认的 90。实测 `--arch 86` 出的是 `#ttg.nvidia_mma` version 2，PTX 里是 `mma.sync`，和 A10 真机一致；默认的 sm_90 出的是 version 3，PTX 里是 `wgmma`。AMD 不用传 `--warp_size 64`，HIP 后端会自己把 `threads-per-warp` 定成 64。
6. CE 偶尔超时（SIGKILL，code 143），重试一次就好。
7. **CE 会按源码行给 Python 源码和 Device Viewer 里的 IR 上同样的底色**。这正好对应 §4「每一段 IR 都要能指回它来自哪一行 `tl.*`」，正文可以引导读者去看。
8. **3.7.0 和 3.9.0 的 IR 差异**：拿 64×64×64 的 `tl.dot` 比过，sm_86 的 TTGIR 两边都是 46 行，layout 和 op 完全相同，只差 3.9.0 的 module 属性多了一个 `"ttng.two-ctas" = false`。3.9.0 那份是在 Max 1100 那台机器上，用 triton-xpu 自带的 NVIDIA 后端加 `triton.compile(ASTSource(...), target=GPUTarget("cuda", 86, 32))` 编出来的，**不需要 NVIDIA GPU**。正文贴的 IR 一律用这种办法从 3.9.0 dump；CE 只给读者自己动手看。流水线、persistent 这些复杂情况到 ch13 以后再逐个比较，有差异就按 §14 第 2 项加 `warn` 框。

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
- **ch08 访存的三种写法。** 裸指针、block pointer（**3.9.0 已移除**，只讲历史和怎么迁移）、tensor descriptor（kernel 里的 `tl.make_tensor_descriptor` 和 host 端的写法）各有什么取舍，以及它们在 IR 里长得有什么不同（预告第四部分）。**Max 1100 实测：同一个 GEMM，tensor descriptor 版本 122–138 TFLOP/s，指针版本只有 13–18 TFLOP/s**，这一章是 Intel 后端「写法决定性能」的主场。
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
5. **分支与 PR**：在分支 `inside-triton` 上做；PR 节奏默认全书一个。
6. ~~**读者的安装命令用不用 uv**~~ **已定（2026-11-15）：以 uv 为主，附 pip 对照**。正文和附录里的命令一律写 `uv venv` / `uv pip install`。附录 A 第 1 节给了一张 uv ↔ pip 的逐条对照表，正文不再逐条重复 pip 的写法。

## 15. 进度与交接（换机器、换会话时从这里接着做）

> 仓库是**公开**的。下面只写通用描述，不写任何机器的主机名、用户名、内部路径或内部驱动构建名。

### 15.1 已完成（批次 0 进行中）
- 本 BRIEF 草案。
- `books/inside-triton/verify/probe-env.py`：只读的环境探测脚本，共 14 项，每项独立 try，最后汇总。会打印 venv 状态、`LD_LIBRARY_PATH`、`CXX`、`CPATH`、PATH 里能找到的编译器、驱动与设备信息，然后跑 vector add、`tl.dot`、`do_bench`、解释器、`torch.compile`、Gluon、proton、Helion。用法：`<venv>/bin/python probe-env.py --helion`。
- **A10：14/14 通过**（torch 2.14.1+cu126，再 `--no-deps` 强装 triton 3.9.0；装法见 §3.1）。

### 15.2 Max 1100：14/14 通过（2026-11-15）
装法：`uv venv` 建在**本地盘**（XPU 版 torch 加 oneAPI 运行时有好几 GB），然后
`uv pip install --no-config --python <venv>/bin/python torch==2.15.0 --index-url https://download.pytorch.org/whl/xpu`，再装 `helion packaging`。
torch 2.15.0+xpu 自带依赖 `triton-xpu 3.9.0`。

设备：448 EU，sub-group 16/32，约 48 GB；Gluon 有 `gluon.language.intel`（导出 `IntelDPASLayout`）。环境变量的完整组合固化在 `verify/env-xpu.sh.example`（只放占位路径），从 `env -i` 的干净环境里只 source 这一份文件，就能跑到 14/14。一路踩到的坑（每一条都要写进附录 A）：

1. **自己编译的 DPC++（intel/llvm）出现在 `LD_LIBRARY_PATH` 里**：pip 装的 `libsycl.so.9` 会链到那份更旧的 `libur_loader.so.0`，报 `undefined symbol: urKernelSetArgLocal, version LIBUR_LOADER_0.12`。原因是 `LD_LIBRARY_PATH` 的优先级高于 wheel 自带的 RUNPATH。解法：从 `LD_LIBRARY_PATH` 里去掉那个构建目录。
2. **反过来，把 `LD_LIBRARY_PATH` 整个清空也不行**：如果 Level Zero loader（`libze_loader`）和 GPU 用户态驱动不在系统目录里、只靠 `LD_LIBRARY_PATH` 才找得到，清空之后 `sycl-ls` 里就没有 `level_zero:gpu` 这一行，`torch.xpu` 看到 0 个设备，Triton 报 `0 active drivers`。解法：`LD_LIBRARY_PATH` 里只保留 Level Zero loader 和 GPU 驱动所在的目录。
3. **Triton 自己挑编译器**：Intel driver 首次运行时要现场编一段 C++ 胶水代码。`python/triton/runtime/build.py` 的挑法是 `knobs.intel.sycl_compiler`，然后 PATH 里的 `icpx`，然后 `dpclang++`。它挑中了自己编译的 `dpclang++`，而那份编译器的 SYCL 头文件与 pip 装的运行时版本不一致。解法：设 `CXX=g++`，这样它就不加 `-fsycl`，SYCL 头文件改从 pip 包 `intel-sycl-rt` 里取（逻辑见 `backends/intel/driver.py` 的 `find_sycl_icpx`）。
4. **缺 Level Zero 头文件**：胶水代码要 `#include <level_zero/ze_api.h>`，pip 包不提供（PyPI 上没有 Level Zero 的开发包）。解法：系统装 `level-zero-dev`，或者用 `CPATH` 指向一份现成的 Level Zero `include/`（Level Zero loader 的发行包里通常带 `include/level_zero/`，最好用和 loader 同版本的那份；intel/llvm 的构建目录里也有）。
5. **链接时找不到 `-lze_loader`**：`LD_LIBRARY_PATH` 只管运行时，链接器看的是 `LIBRARY_PATH`。解法：`LIBRARY_PATH` 指向 `libze_loader.so` 所在的目录。
6. **`ocloc` 不在 PATH 里**，有两种后果：
   - 明的：`torch.compile`（Inductor）和 Helion 报 `FileNotFoundError: 'ocloc'`，因为它们要生成原生二进制（`backends/intel/compiler.py` 的 `make_zebin`）。
   - **暗的（重要）**：Intel driver 通过查询 OpenCL 扩展来判断设备能力（`cl_intel_subgroup_matrix_multiply_accumulate` 等）。机器上没有 OpenCL 平台时（`clinfo -l` 是空的，`sycl-ls` 只有 `level_zero:gpu`），会退回走 Level Zero 的 `ext_oneapi_supports_cl_extension`，而这条路需要 `ocloc`（见 `backends/intel/extension_utils.c`）。找不到 `ocloc` 时异常被吞掉、返回 False，结果是 target 里 `has_subgroup_matrix_multiply_accumulate: False`，**`tl.dot` 算得对，但 TTGIR 里没有 `#ttig.dpas`，不走矩阵单元**，也没有任何报错或警告。这一条除了写进附录 A，也是 ch23「后端怎么探测硬件能力」和 ch09「结果对不代表编译对」的现成例子。`probe-env.py` 现在把这种情况判为 FAIL。
   - 解法：把 GPU 用户态驱动里带的 `ocloc` 所在目录加进 PATH。

### 15.2b 批次进度（2026-11-15）
- **批次 0**：完成。还剩 `verify-outputs.py`，等第一个需要逐字节比对的批次再写，规则照 MLIR 书那份：性能数字只核量级，末尾空白不计。
- **批次 1**：完成。ch01–04 和附录 B 都已写完，`verify-triton.sh` 在 Max 1100 上 7/7 通过；浏览器模拟器的输出都逐字节核对过；每章的 SVG 都截图检查过。A10 上要补测的东西见 §15.5。
- **批次 2**：完成（2026-11-15）。ch05–10 都已写完；minillm 有 `pointwise`、`softmax`、`norm`、`matmul`（指针版 / descriptor 版 / `mm`）、`ops.py`（`triton_op` 加反向），共 126 个测试；`verify-triton.sh` 全书通过。
  - 写作中订正过的事实：block pointer 在 3.9.0 已移除；`tl.dot` 的形状下限由各后端的 `min_dot_size` 决定（NVIDIA 只限 K，Intel 退回 FMA）。
  - 写作中新发现、写进正文的坑：Intel 后端的 `tf32x3` / `bf16x3` 不走矩阵单元；解释器不查越界；`triton_op` 加 host 端 `TensorDescriptor` 在 `torch.compile` 后会被删掉。
- **下一步**：批次 3（ch11–17）。ch15、ch16 的 Hopper 部分仍按 §14 第 1 项待定。
- **写作流程**（每章都照这个来）：
  1. 先在 Max 1100 上把要用的数字测出来，或者用 `triton_ir_local.py` 取 3.9.0 的 IR；
  2. 写正文，运行结果先留占位符；
  3. 用 `verify/blocks.py` 从章节里原样抽出代码块重跑，把真实输出填回去，再按输出修正正文里的数字和结论；
  4. 浏览器模拟器的输出逐字节比对；
  5. 构建，用 Playwright 截图检查 SVG；
  6. 查公开信息，然后提交。

### 15.3 下一步
1. ~~Max 1100 跑到 14/14~~（已完成）；~~固化 `verify/env-xpu.sh.example`~~（已完成）。据此写附录 A 的「Intel GPU」一节。
2. 改过的 `probe-env.py`（`tl.dot` 没有矩阵 layout 判 FAIL；打印 `LIBRARY_PATH`、`ocloc`）请作者在 A10 上再跑一次，确认仍是 14/14。
3. ~~`build.js` 加 `triton-ir` / `gluon` / `helion` / `ttgir-norun` 围栏，实测 CE 的 Triton 接口~~（已完成，见 §7.2）；~~`book.json`~~（已完成）。
4. ~~minillm 包骨架~~（已完成：`code/minillm/`，`pip install -e '.[test]'`；`tests/test_env.py` 是环境冒烟测试）。~~`verify-triton.sh`、`verify-minillm.sh`~~（已完成，用法见 `verify/README.md`）。
   - `verify-triton` 默认**离线**编译 `triton-ir` 块，用的是 `verify/triton_ir_local.py`：做法和 CE 一样，FakeTensor 加 mock driver，在本机用 3.9.0 编 NVIDIA / AMD 目标，**不需要对应的 GPU**。正文里的 `ttgir-norun` 块也用它来取 3.9.0 的真 IR。`CE=1` 时改走 CE 的 API。
   - 两种模式都在 Max 1100 上测过：本机能跑的块通过，`sm90` 的块跳过，故意写错的块报失败。`triton-ir` 块覆盖了 NVIDIA、AMD、Gluon 三种情况；`# IR:` 写了不存在的一级时也会报错。
   - 公司网络要走代理时，`CE=1` 需要 `https_proxy`。连不上时脚本会立刻报错退出，不会每个块都干等到超时。
5. ~~附录 A~~（已完成：`chapters/appendix-a.md`）。写的时候顺带核实或订正了几件事：
   - PyPI 上的 triton 3.9.0 **没有 cp310 和 cp313 的 wheel**，本书定用 Python 3.12。
   - torch 2.15.0 只有 `cu132`、`xpu`、`cpu` 三个版本；ROCm 索引上最新只到 2.14.1+rocm7.2。
   - `triton` 和 `triton-xpu` 装进的是同一个 `triton/` 目录，会互相覆盖。
   - **FP fusion 的警告只在 NVIDIA 和 AMD 上出现**：这两个后端走 `knobs.language.fp_fusion_enabled()`，Intel 后端直接读 `default_fp_fusion`，开关同样有效，但不发警告。
   - Helion 缺 `packaging` 时，**`import helion` 就会失败**（`helion/_compat.py` 的顶层 import），不是等到运行 kernel。
   - Intel 后端挑 C++ 编译器时 **`CXX` 优先**；没设 `CXX` 才依次找 `TRITON_INTEL_SYCL_COMPILER`、`icpx`、`dpclang++`、`clang++`、`g++`。
   - 纯 CPU 环境（`torch==2.15.0+cpu` 加 `triton==3.9.0 numpy`）实测过：`TRITON_INTERPRET=1` 和 `triton_ir_local.py` 都能用。
   - `probe-env.py` 的说明和结尾提示改成了面向读者的写法。
6. 批次 0 剩下的（§12）：`verify-outputs.py`（等第一章有「运行结果」块时再写，规则照 MLIR 书那份）；本书的 `README.md`（首页）。

### 15.5 待 A10 实测（写作在 Max 1100 上进行，正文先用 Max 1100 的数字并标明；A10 的输出贴回来后补进正文）
每一项都写明「跑什么」和「正文给出的预测」。预测没对上，也是值得写进正文的素材。
- **ch01**：第一个 `triton` 块（softmax 四种实现），分别不带参数和带 `1024` 各跑一次；第二个 `triton` 块（attention）跑一次。
  - **预测**：A10 的 L2 只有 6 MB，所以 1024 列和 16384 列两档，eager 都比 `torch.softmax` 慢 4 倍左右；在 Max 1100 上，1024 列只慢 2.7 倍。
  - 正文「动手」第 1 题就是请读者验证这个预测。A10 的结果出来以后，补进「先预测，再测量」那一节，和 Max 1100 的数字放在一起。
- **ch02**：`hw_probe.py`（第一个 `triton` 块）跑一次。
  - **预测**：fp32 峰值约 31.2 TFLOP/s（72 × 128 × 2 × 1.695 GHz）；拐点约 52 FLOP/字节，所以 ITERS=64（16 FLOP/字节）仍受带宽限制，ITERS=256（64 FLOP/字节）应当已经过了拐点。
  - L2 只有 6 MB，所以实验 1 的「L2 台阶」只会出现在 1–4 MiB 附近，而且会和启动开销搅在一起，未必看得出来。
  - fp16 矩阵乘不超过标称的 125 TFLOP/s；显存拷贝应在 600 GB/s 以下（probe 的 x*2 测到约 473 GB/s）。
  - 结果出来后，在第 2 章的 roofline 一节补一张 A10 的对照。
- **ch03**：
  - 把 CUDA 的 `add_kernel` 完整程序存成 `add.cu`，用 `nvcc -O3 -arch=sm_86 add.cu -o add && ./add` 跑一次。CE 上只验证过它能编译，正文还没有贴它的运行结果，请把输出贴回来。
  - 跑一次第二个 `triton` 块（跨步读取）。**预测**：NVIDIA 取数据的粒度是 32 字节的 sector，平台从 `STRIDE=8` 就开始，而不像 Max 1100 那样从 16 开始。
- **ch04**：
  - 用空的缓存目录把 `life.py`（第一个 `triton` 块）跑两次，正文「动手」第 1 题需要 NVIDIA 上的数字。**预测**：「初始化 driver」比 Max 1100 小得多；第一次调用仍然有一笔几百毫秒的 `triton_key()`（PyPI 的 `libtriton.so` 是 477 MB，Intel 的是 1.2 GB）。
  - 跑 `verify-minillm.sh`，以及 `bench/bench_pointwise.py`。**预测**：`silu_mul` 快 1.6 倍左右；A10 的 L2 小，可能更接近 1.7。
- **ch05、ch06**：
  - 跑 ch05 的 `copy2d` 块和 ch06 的溢出实验。
  - 跑 `bench_softmax.py`、`bench_rmsnorm.py`。
  - **预测**：NVIDIA 没有自动换大寄存器模式这一步，所以 32768 列、4 个 warp 的溢出会更严重；online 和三遍的比值在长行上仍然约为 4/3。
- **ch07**：
  - 跑精度实验（第二个 `triton` 块）。**预测**：`tf32x3`、`bf16x3` 走矩阵单元，速度介于 TF32 和 `ieee` 之间，误差接近 `ieee`。
  - 跑 `bench_matmul.py`。**预测**：指针版在 A10 上和 `torch.matmul` 的差距比 Max 1100 上的 12 倍小得多。官方教程的指针写法在 NVIDIA 上是能接近 cuBLAS 的。
- **ch08**（这一项最重要，`minillm.mm` 在 A10 上默认走哪条路就看它）：
  - 跑第一个 `triton` 块，对比指针版和 kernel 里创建 descriptor 的版本。
  - 跑 `bench_matmul.py`，对比指针版和 host 端创建 descriptor 的 `matmul_desc`。
  - **预测**：sm_86 上 kernel 里创建的 descriptor 编成 `cp.async`，应当和指针版接近；host 端创建的编成 `ld.global` 加 `st.shared`，没有异步拷贝，可能更慢。
  - 如果实测发现 descriptor 在 A10 上更快，要改 `prefer_descriptor()`，并修改 ch08「该用哪种写法」那张表。
- **ch09**：
  - 跑 `verify-minillm.sh`，看 `tests/test_codegen.py` 在 NVIDIA 上能不能通过（Intel 专属的那一项应当跳过）。
  - 跑本章的 `device_print` 块，看 NVIDIA 上的打印格式（Max 1100 上是 `pid (1, 0, 0) idx (0) x: 0.000000`）。
  - 试一试 fp16 的 8×8 `tl.dot`，应当报 `Input shapes should have M >= 1, N >= 1 and K >= 16`（离线编译已证实）。
- **ch10**：
  - 跑 `verify-minillm.sh`（`test_ops.py` 在 NVIDIA 上走的是指针版 mm）。
  - 跑 MLP 的四种对比。
  - 复现「`triton_op` + host 端 `TensorDescriptor` + `torch.compile`」，看 NVIDIA 上是不是同样会被删掉。
- 跑法：`books/inside-triton/verify/verify-triton.sh ch01` 只校验代码能不能跑，不打印输出；要贴的数字请用 `VERBOSE=1` 跑，或者把代码块单独存成文件运行。

### 15.4 写作规则备忘（来自作者的反馈，散落在各节，这里汇总）
- 每本书自成一体，与本仓库其它书重复没关系（§1）。
- 篇幅不设上限，判据是内容实不实在（§1、§13）。
- 只用公开信息（§13）。仓库是公开的，交接记录也一样。
- Python 环境用 uv（§3.1）。
