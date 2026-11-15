# 附录 A 环境搭建

## 这份附录怎么用

<div class="goal-box">
<p>正文讲 kernel 怎么写、为什么快，这份附录负责把环境装好。Triton 自己装起来不难，难在<strong>它和 PyTorch、GPU 驱动三者的版本要对得上</strong>：装错一个，报错常常出现在离真正原因很远的地方，比如 <code>undefined symbol</code>、<code>0 active drivers</code>，甚至什么错都不报，只是 <code>tl.dot</code> 悄悄没用上矩阵单元。</p>
<p>下面按 GPU 厂商分节，每一节都只写本书验证机上真走通过的路；没走通过的路会明说。装完以后用第 6 节的体检脚本确认 14 项全过，再开始读第 4 章。<strong>先看第 0 节的版本基线。</strong></p>
</div>

## 0. 版本基线（全书按这个写）

| 组件 | 本书取值 | 备注 |
| --- | --- | --- |
| Triton | **3.9.0**（2026-10-27 发布） | 源码锚点以 `triton-lang/triton` 的 `v3.9.0` 标签为准 |
| PyTorch | **2.15.0**（NVIDIA 驱动太老时用 2.14.1，见 2.3 节） | torch 2.15.0 声明依赖 `triton~=3.9.0` |
| Intel 后端 | **triton-xpu 3.9.0** | 随 `torch==2.15.0+xpu` 一起装上，来自公开仓库 `intel/intel-xpu-backend-for-triton` |
| Helion | **1.5.2** | 第 26 章用 |
| Python | **3.12** | 见下面的提醒 |
| 网页上的 CE 按钮 | Compiler Explorer 的 **Triton 3.7.0** | 比基线落后两个小版本，见第 7 节 |

本书的两台验证机，正文里的性能数字都会注明是哪一台测的：

| 机器 | 架构 | 驱动 | 装法 |
| --- | --- | --- | --- |
| NVIDIA A10 | Ampere（sm_86），24 GB | 550.163.01（最高支持 CUDA 12.4） | 2.3 节 |
| Intel Data Center GPU Max 1100 | Xe-HPC（PVC），48 GB | Level Zero 用户态驱动 | 第 3 节 |

<div class="warn">
<strong>Python 选 3.12</strong>
<p>写书时（2026-11-15）PyPI 上的 <code>triton==3.9.0</code> 只有 cp311、cp312、cp314、cp315 的 wheel，<strong>没有 3.10 和 3.13</strong>。用 3.13 的话，<code>pip</code> 会退回去装源码包，或者干脆报找不到版本。本书全程用 3.12。</p>
</div>

## 1. 用 uv 管 Python 环境

本书的命令一律用 **uv**：建 venv 快，`uv venv --python 3.12` 在机器上没有 3.12 时会自己下载一份，装几个 GB 的 PyTorch wheel 也比 pip 快得多。装 uv（官方脚本，装到 `~/.local/bin`）：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

习惯用 pip 也没关系，下表是逐条对照，本书后面的命令照着换就行：

| uv | pip |
| --- | --- |
| `uv venv --python 3.12 .venv` | `python3.12 -m venv .venv` |
| `source .venv/bin/activate` | 同左 |
| `uv pip install 包名` | `pip install 包名` |
| `uv pip install 包名 --index-url URL` | `pip install 包名 --index-url URL` |
| `uv pip install --no-deps 包名` | `pip install --no-deps 包名` |
| `uv pip check` | `pip check` |

<div class="note">
<strong>别混用</strong>
<p><code>uv venv</code> 建出来的环境里<strong>默认没有 pip</strong>。在它里面敲 <code>pip install</code>，多半会落到系统的 pip 上，把包装到别处去。选定一种就一直用它。</p>
</div>

venv 要建在**本地盘**上。CUDA 版、XPU 版的 PyTorch 连同依赖都有好几 GB，放在网络盘上，安装和 `import torch` 都会慢得多。

## 2. NVIDIA GPU

### 2.1 先看驱动支持到哪个 CUDA

```bash
nvidia-smi
```

输出右上角的 `CUDA Version` 是**这个驱动最高能支持的 CUDA 版本**，不是你装了哪个 CUDA toolkit。Triton 的 wheel 自带 `ptxas`，机器上的 `nvcc` 是什么版本都无所谓；**真正卡人的是驱动**。

PyTorch 2.15.0 只出了 CUDA 13.2 版（`cu132`），要求驱动支持 CUDA 13，也就是 580 及以上。驱动够新走 2.2 节，不够走 2.3 节。

### 2.2 驱动 ≥ 580：直接装 torch 2.15.0

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132   # 会一起装上 triton 3.9.0
```

<div class="note">
<strong>这条路本书没有在真机上走过</strong>
<p>本书的 A10 驱动是 550，用的是 2.3 节的装法。上面的版本号和索引地址是 2026-11-15 在 PyTorch 官方索引上核对过的：<code>cu132</code> 下有 <code>torch-2.15.0+cu132</code> 和 <code>triton-3.9.0</code>。如果你走这条路遇到问题，以第 6 节体检脚本的结果为准。</p>
</div>

### 2.3 驱动更老（本书 A10 走的路）：torch 2.14.1 + triton 3.9.0

不想升级驱动的话，可以装 CUDA 12.6 版的 torch 2.14.1，再**强行**把 Triton 换成 3.9.0：

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126   # 会带上 triton 3.8.x
uv pip install triton==3.9.0 --no-deps                                           # 覆盖成 3.9.0
```

torch 2.14.1 声明的依赖是 `triton~=3.8.0`，所以第二步必须加 `--no-deps`，之后 `uv pip check` 也会报这一条版本冲突。**这是预期的，不用管它。** 本书在 A10（驱动 550.163.01）上实测过这个组合：vector add、`tl.dot`（TTGIR 里是 `#ttg.nvidia_mma`）、`do_bench`、解释器、`torch.compile`、Gluon（含 `nvidia/ampere` 子包）、proton、Helion 全部通过。

这毕竟是 PyTorch 官方配套表之外的组合。如果哪天 `torch.compile` 生成的 kernel 编不过，先怀疑这一条，试试换回 torch 2.14.1 自带的 triton 3.8。

## 3. Intel GPU

### 3.1 驱动

Intel GPU 要装两样东西：内核驱动，以及用户态的 Level Zero 运行时和计算运行时。按 Intel 官方文档（`dgpu-docs.intel.com`）给你的 GPU 和发行版装。在 Ubuntu 24.04 上，先按文档加好 Intel 的软件源，Triton 用得到的用户态包是这几个：

```bash
sudo apt install libze1 libze-intel-gpu1 intel-opencl-icd intel-ocloc libze-dev
```

装完用 oneAPI 的 `sycl-ls` 看一眼（也可以等第 6 节体检脚本告诉你）：输出里要有一行 `[level_zero:gpu]`。没有这一行，后面什么都跑不起来。再用 `clinfo -l` 看一眼，应当列出你的 GPU。这一条 Triton 用不到，但 PyTorch 的部分算子要用，见 3.3 节表格的最后一行。

另外还要一个 C++ 编译器，`g++` 就行，原因见 3.3 节。

### 3.2 PyTorch 与 Triton

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/xpu   # 会一起装上 triton-xpu 3.9.0
```

`import triton` 拿到的就是 Intel 后端：`triton-xpu` 这个发行包装进的正是 `triton/` 这个目录。

<div class="warn">
<strong>不要再装 PyPI 上的 <code>triton</code></strong>
<p><code>triton</code> 和 <code>triton-xpu</code> 装进的是<strong>同一个 <code>triton/</code> 目录</strong>，后装的会覆盖先装的。一旦哪条命令顺手带上了 <code>triton</code>（比如没加 <code>--index-url</code> 就装了别的依赖 torch 的包），Intel 后端就没了。出了怪问题，先 <code>uv pip list | grep -i triton</code> 看看，应当只有一个 <code>triton-xpu</code>。</p>
</div>

### 3.3 第一次运行时的坑

Intel 后端第一次运行时要**现场编一段 C++ 胶水代码**（`spirv_utils`），编完放进缓存，以后复用。所以这一步要用到 C++ 编译器、Level Zero 的头文件和库，以及 `ocloc`。如果它们都按 3.1 节装在系统目录里，正常情况下什么都不用设。

本书的 Max 1100 验证机走的是另一条路：Level Zero 和 GPU 驱动不在系统目录里，靠环境变量指过去。一路踩到的坑如下，每一条都是真实报错：

| 症状 | 原因 | 解法 |
| --- | --- | --- |
| 加载 SYCL 运行时时报 `undefined symbol: urKernelSetArgLocal, version LIBUR_LOADER_0.12` | `LD_LIBRARY_PATH` 里混进了另一份（自己编译的、或者更旧的）DPC++ / oneAPI 运行时。`LD_LIBRARY_PATH` 的优先级高于 wheel 自带的 RUNPATH，pip 装的 `libsycl` 于是链到了旧的 `libur_loader` | 把那个目录从 `LD_LIBRARY_PATH` 里拿掉 |
| `torch.xpu` 看到 0 个设备；Triton 报 `0 active drivers` | 反过来，`LD_LIBRARY_PATH` 被清得太干净。Level Zero loader 和 GPU 驱动不在系统目录时，只能靠它找到 | `LD_LIBRARY_PATH` 里**只**放 Level Zero loader 和 GPU 用户态驱动所在的目录 |
| 胶水代码编译失败，报错里的编译命令是 PATH 里另一份 `icpx` / `dpclang++`，还带着 `-fsycl` | Triton 优先挑 SYCL 编译器，挑中的这一份和 pip 装的 SYCL 运行时不是同一个版本，头文件对不上 | 设 `CXX=g++`（见下面的说明） |
| `fatal error: 'level_zero/ze_api.h' file not found` | 胶水代码要 `#include <level_zero/ze_api.h>`，pip 包里不带这个头文件 | 装 `libze-dev`，或者用 `CPATH` 指向一份 Level Zero 的 `include/`，最好和 loader 同版本 |
| `ld: cannot find -lze_loader` | 链接器不看 `LD_LIBRARY_PATH`，它看的是 `LIBRARY_PATH` | `LIBRARY_PATH` 指向 `libze_loader.so` 所在的目录 |
| `torch.compile` 或 Helion 报 `FileNotFoundError: 'ocloc'` | 要生成原生二进制时得调用 `ocloc` | 把 `ocloc` 所在的目录加进 `PATH` |
| `F.scaled_dot_product_attention` 等算子报 `RuntimeError: could not create a primitive`；设 `ONEDNN_VERBOSE=1` 能看到 `OpenCL device not found` | 这些算子走 oneDNN，而 oneDNN 要通过 OpenCL ICD loader 找到 GPU。`/etc/OpenCL/vendors/` 下没有 GPU 的 `.icd` 文件（`clinfo -l` 是空的，或者只有 CPU） | 装 `intel-opencl-icd`；或者自己建一个目录，放一个 `.icd` 文件，内容是 GPU 驱动里 `libigdrcl.so` 的绝对路径，再用 `OCL_ICD_VENDORS` 指向这个目录。Triton 本身不受影响 |

Triton 挑 C++ 编译器的顺序是：先看 **`CXX`**；没设 `CXX`，再依次找 `TRITON_INTEL_SYCL_COMPILER` 指定的编译器、`icpx`、`dpclang++`，最后退到 `clang++`、`g++`。挑中 SYCL 编译器时它会加 `-fsycl`，此时用的是那个编译器自带的 SYCL 头文件；挑中普通 C++ 编译器时，头文件改从 pip 包 `intel-sycl-rt` 里取，正好和运行时同版本。逻辑在 Intel 后端的 `triton/runtime/build.py`。所以只要设了 `CXX=g++`，机器上装了几份 oneAPI 都不会互相干扰。

`ocloc` 那一条还有一个**不报错**的后果，单独说：

<div class="warn">
<strong>没有 <code>ocloc</code>，<code>tl.dot</code> 照样算对，但不走矩阵单元</strong>
<p>Intel 后端靠查询设备扩展（例如 <code>cl_intel_subgroup_matrix_multiply_accumulate</code>）来判断能不能用 DPAS 矩阵指令。机器上没有 OpenCL 平台时（<code>clinfo -l</code> 是空的，<code>sycl-ls</code> 里只有 <code>level_zero:gpu</code>），这个查询要靠 <code>ocloc</code> 完成，见 Intel 后端的 <code>backends/intel/extension_utils.c</code>。找不到 <code>ocloc</code> 时，查询结果是「不支持」，<strong>没有任何报错或警告</strong>。于是 <code>tl.dot</code> 的结果完全正确，但 TTGIR 里没有 <code>#ttig.dpas</code>，性能差一个量级。第 6 节的体检脚本会把这种情况判为 FAIL；第 9 章会回到这个例子，讲「结果对，不代表编译对」。</p>
</div>

本书仓库里的 `books/inside-triton/verify/env-xpu.sh.example` 把上面这些设置都写好了，每个变量为什么要设也写了注释。复制成 `env-xpu.sh`，填上三个路径，每次开新终端先 `source` 它：

```bash
cp books/inside-triton/verify/env-xpu.sh.example env-xpu.sh   # 改里面的 ZE_LOADER_DIR / GPU_DRIVER_DIR / VENV
source env-xpu.sh
```

## 4. AMD GPU

本书**没有 AMD 验证机**，AMD 相关的内容（第 23 章等）用的是 Compiler Explorer 和第 8 节的离线编译。写书时（2026-11-15），PyTorch 官方 ROCm 索引上最新的是 `torch 2.14.1+rocm7.2`，还没有 2.15.0。要在 AMD GPU 上跑本书的代码，请按 PyTorch 官网（`pytorch.org` 的 Get Started 页）的安装选择器装 ROCm 版，再用第 6 节的体检脚本确认。

## 5. Gluon、proton 与 Helion

**Gluon** 和 **proton**（Triton 自带的 profiler）都随 Triton 一起装好了，不用另装：

- Gluon 在 `triton.experimental.gluon`。名字里的 `experimental` 是真的，API 每个版本都可能变（第 24 章）；
- proton 在 `triton.profiler`（第 11 章）。

**Helion**（第 26 章）要单独装：

```bash
uv pip install helion packaging
```

最后那个 `packaging` 不是写错：Helion 1.5.2 一导入就要用 `packaging`（`helion/_compat.py` 的第一批 import 里就有），但它的依赖声明里漏了这个包。不手动装，`import helion` 就会报 `ModuleNotFoundError: No module named 'packaging'`。

## 6. 体检：probe-env.py

装完以后先跑本书的体检脚本。它是只读的，不装任何东西，共查 14 项，每一项单独 try，某一项失败不影响后面：

```bash
.venv/bin/python books/inside-triton/verify/probe-env.py --helion
```

它会先打印 venv 状态、相关的环境变量、PATH 里能找到的编译器、驱动和设备信息，然后依次真跑 vector add、`tl.dot`、`do_bench`、解释器、`torch.compile`、Gluon、proton、Helion，最后给出汇总。下面是 Max 1100 上的汇总：

运行结果（Max 1100，torch 2.15.0+xpu，triton-xpu 3.9.0）：

```
===== 汇总 =====
OK   | import torch
OK   | torch 设备
OK   | import triton
OK   | active driver / target / DEVICE
OK   | vector add（逐点 + mask）
OK   | 编译产物（asm 各级 IR）
OK   | tl.dot（fp16 输入，fp32 累加，64x64x64）
OK   | triton.testing.do_bench
OK   | TRITON_INTERPRET 可用（CPU 解释器）
OK   | torch.compile（Inductor 生成 Triton kernel）
OK   | import triton.experimental.gluon
OK   | import triton.profiler (proton)
OK   | import helion
OK   | Helion 固定 Config 的 add kernel（--helion）

共 14 项，失败 0 项。
```

A10 上也是 14 项全过。汇总上面的明细里有两行值得留意：

- **`tl.dot` 那一行会打印 TTGIR 里的矩阵 layout。** NVIDIA 上应当是 `#ttg.nvidia_mma`，Intel 上应当是 `#ttig.dpas`。哪个都没有，就说明 `tl.dot` 没用上矩阵单元，脚本会把这一项判为 FAIL。
- **`do_bench` 那一行是一个粗略的带宽数字**：把 64 MiB 的张量乘以 2，算读写合计的带宽。A10 上大约 473 GB/s（标称 600 GB/s），Max 1100 上大约 531 GB/s。它只用来确认量级对不对；怎么测才准，是第 11 章的事。

## 7. 先设好一个开关：`TRITON_DEFAULT_FP_FUSION`

在 NVIDIA 和 AMD 上，Triton 3.9 第一次编译 kernel 时会打印一条警告（`knobs.py` 前面的路径因机器而异）：

```
…/triton/knobs.py:504: UserWarning: TRITON_DEFAULT_FP_FUSION is unset: defaulting to FP fusion ENABLED for backward compatibility in Triton 3.9. The default will change to DISABLED in Triton 3.10; set TRITON_DEFAULT_FP_FUSION=1 to keep fusion enabled or =0 to adopt the new default early and silence this warning.
```

**FP fusion** 指的是把 `a * b + c` 融合成一条 FMA（乘加）指令：只舍入一次，所以和分两步算的结果在最后一位上可能不同。3.9 默认开启，3.10 将改成默认关闭。这会影响和 PyTorch 结果的逐位比较（第 7 章、第 9 章）。

Intel 后端看不到这条警告：它直接读同一个开关的默认值，没有经过发警告的那条路径。但开关同样有效，默认同样是开启的。

本书正文统一按 **3.9 的默认行为（开启）** 写。把它显式设上，警告就不会再出现；以后升级到 3.10，行为也不会悄悄变：

```bash
export TRITON_DEFAULT_FP_FUSION=1
```

## 8. 没有 GPU 时能做什么

这本书有相当一部分内容不需要 GPU：

1. **网页上 ` triton-ir ` 代码块的「🔬 在 Compiler Explorer 看 IR」按钮。** 点开是 Compiler Explorer 上的三栏：源码、编译器、Device Viewer。Device Viewer 里能切换 source / TTIR / TTGIR / LLVM IR / PTX（或 AMDGCN），CE 还会给源码行和对应的 IR 行上同样的底色。注意 CE 上是 **Triton 3.7.0**，比本书基线旧两个小版本。正文贴的 IR 都是 3.9.0 dump 出来的，两者有差别的地方，正文会用提示框说明。
2. **本地离线编译**，看 3.9.0 的 IR。装一个 CPU 版的 PyTorch 和 Triton：

   ```bash
   uv venv --python 3.12 .venv && source .venv/bin/activate
   uv pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
   uv pip install triton==3.9.0 numpy
   ```

   然后用本书仓库里的 `triton_ir_local.py`。它和 CE 用的是同一个办法：张量换成 FakeTensor，GPU driver 换成一个只报告目标架构的 mock，kernel 只编译、不运行。所以 NVIDIA 和 AMD 目标的 IR 都能编出来：

   ```bash
   python books/inside-triton/verify/triton_ir_local.py my_kernel.py --show ttgir                  # NVIDIA sm_86
   python books/inside-triton/verify/triton_ir_local.py my_kernel.py --arch 90 --show ttgir        # Hopper
   python books/inside-triton/verify/triton_ir_local.py my_kernel.py --backend hip --arch gfx942 --show ttgir
   ```

   要求和 CE 一样：脚本里必须真的启动一次 kernel（`kernel[grid](...)`），张量写 `device="cuda"`。
3. **Triton 解释器。** 设 `TRITON_INTERPRET=1`，kernel 就改在 CPU 上用 NumPy 逐个 program 模拟执行，张量用普通的 CPU 张量。它慢，也不模拟任何性能特征，但用来检查**结果对不对**、在 kernel 里打断点，都很好用（第 9 章）。同样是上面那个 CPU 环境：

   ```python-norun
   import torch
   import triton
   import triton.language as tl

   @triton.jit
   def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
       offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
       mask = offs < n
       tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)

   x = torch.arange(10, dtype=torch.float32); y = torch.ones(10); out = torch.empty(10)
   add_kernel[(triton.cdiv(10, 4),)](x, y, out, 10, BLOCK=4)
   print(out.tolist())
   ```

   运行结果（`TRITON_INTERPRET=1 python add.py`，CPU 版 torch 2.15.0 + triton 3.9.0）：

   ```
   [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
   ```

   忘了设 `TRITON_INTERPRET=1`，在没有 GPU 的机器上会得到下面这个报错。它说的是「一个可用的 GPU 后端都没找到」：

   ```
   RuntimeError: 0 active drivers ([]). There should only be one.
   ```

## 9. 跑本书的代码与校验脚本

贯穿全书的 minillm 项目（第 4 章起）是一个能 `pip install -e` 的小包。它故意没有把 torch 和 triton 写进依赖，因为不同 GPU 要装不同的 wheel，写进去反而可能装错。所以先按上面装好环境，再：

```bash
cd books/inside-triton/code/minillm
uv pip install -e '.[test]'
pytest
```

在仓库根目录还有两个校验脚本（详见 `books/inside-triton/verify/README.md`）：

```bash
books/inside-triton/verify/verify-triton.sh         # 正文里的完整代码块：能跑的真跑，triton-ir 块离线编译
books/inside-triton/verify/verify-minillm.sh        # minillm 的全部测试
```

`verify-triton.sh` 会读每个代码块开头的 `# arch:` 注释，判断它能不能在你这台机器上跑。不能跑的块记为「跳过」，最后列出来，**跳过不算通过**。

---

**这份附录的要点**

- 本书基线是 **Triton 3.9.0 + PyTorch 2.15.0 + Python 3.12**。三者版本要对得上，**真正卡人的是 GPU 驱动**。
- NVIDIA：驱动 ≥ 580 直接装 `torch==2.15.0`（cu132）。驱动更老的话，torch 2.14.1+cu126 加 `triton==3.9.0 --no-deps` 也能用，本书的 A10 就是这么装的。
- Intel：`torch==2.15.0` 从 `xpu` 索引装，会带上 triton-xpu 3.9.0；**别再装 PyPI 的 `triton`**。第一次运行时要编胶水代码，所以要 `CXX=g++`、Level Zero 的头文件和库，以及 PATH 里的 `ocloc`。少了 `ocloc`，`tl.dot` 会悄悄不走 DPAS。
- Helion 要多装一个 `packaging`。显式设 `TRITON_DEFAULT_FP_FUSION=1`：消掉 3.9 的警告，行为与正文一致，升级到 3.10 也不会悄悄变。
- 装完先跑 `probe-env.py --helion`，14 项全过再往下读。没有 GPU 也能看 IR（CE、`triton_ir_local.py`），也能用 `TRITON_INTERPRET=1` 检查结果对不对。
