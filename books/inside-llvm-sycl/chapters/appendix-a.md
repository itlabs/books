# 附录 A 搭建与构建 intel/llvm：config、compile 与常用命令

## 这份附录怎么用

<div class="goal-box">
<p>正文教你"为什么"，这份附录帮你"上手做"。全书的每一个源码锚点（<code>SemaSYCL.cpp:1174</code>、<code>feature_test.hpp.in</code>、<code>getDeviceLibNames()</code>……）都指向 <strong>intel/llvm 的 <code>sycl</code> 分支</strong>。要真正把它们打开、改一改、编出一个自己的 <code>clang++</code>，你需要能<strong>配置 → 构建 → 运行</strong>这套工具链。这份附录把这条路走一遍：拉代码、<code>configure.py</code> 配置、<code>compile.py</code> 构建、用编出来的编译器跑一个 SYCL 程序，以及日常最常用的命令速查。</p>
<p>它<strong>不是</strong> intel/llvm 官方文档的替代（官方 <code>sycl/doc/GetStartedGuide.md</code> 永远是权威），而是一份"够用就走"的实操清单，配合本书的例子。命令里的绝对路径和版本号取自本书写作时的一台开发机，<strong>请以你自己的环境为准</strong>。</p>
</div>

## 0. 前置：拿到源码

intel/llvm 是一个巨型 monorepo（第 1 章讲过 LLVM 的单仓结构）。SYCL/DPC++ 活在它的 `sycl` 分支：

```bash
git clone https://github.com/intel/llvm.git
cd llvm
git checkout sycl        # DPC++/SYCL 开发分支
```

它顶层就有 `clang/`、`llvm/`、`sycl/`、`libdevice/`、`libclc/`、`sycl-jit/`、`unified-runtime/` 等（附录 C 有完整地图）。构建脚本都在 `llvm/buildbot/` 下——`configure.py` 和 `compile.py` 是官方推荐的入口，比手写 CMake 命令省心。

## 1. 配置（configure.py）

DPC++ 的构建不直接 `cmake`，而是走 `llvm/buildbot/configure.py`——它把 SYCL 需要的一大堆 CMake 变量（启用哪些子项目、SPIR-V 翻译器、UR 后端……）打包好。最小配置：

```bash
# 从 workspace 根目录（llvm/ 的父目录）
CXX=/usr/bin/clang++ CC=/usr/bin/clang \
python llvm/buildbot/configure.py \
    --build-type=Release \
    --llvm-external-projects="clang-tools-extra;compiler-rt"
```

几个要点：

- **`--build-type=Release`**：日常开发用 Release（快、省内存）。要调试编译器本身、下断点，用 `--build-type=Debug`（慢很多、二进制巨大）或 `RelWithDebInfo`（折中，推荐调试用）。
- **`--llvm-external-projects="clang-tools-extra;compiler-rt"`**：把 `clang-tools-extra`（含 `clangd`、`clang-tidy`）和 `compiler-rt`（sanitizer 运行时）一起编。想用 clangd 跳转源码、或用第 19 章的设备端 sanitizer，就需要它们。
- **用系统 clang 自举**：`CXX=/usr/bin/clang++` 让你用现成的 clang 编译 LLVM 自身，比 gcc 快且兼容性好。
- 配置只需跑一次；**改了配置标志才需要重跑**。它会在 `llvm/build/` 下生成 CMake 缓存。

<div class="note">
<strong>常用的额外配置标志</strong>
<code>configure.py</code> 支持一批开关，按需加：<code>--cuda</code> / <code>--hip</code> 启用 NVIDIA/AMD 后端（第 11、19 章那两条 libclc/devicelib 路）；<code>--shared-libs</code> 编成动态库（增量链接快）；<code>--cmake-opt=-DXXX=YYY</code> 透传任意 CMake 变量。<code>python llvm/buildbot/configure.py --help</code> 看全部。不确定就用最小配置——SPIR-V（Intel GPU/CPU 经 Level Zero/OpenCL）默认就开。
</div>

## 2. 构建（compile.py）

配置好后用 `llvm/buildbot/compile.py` 构建。默认目标是 `deploy-sycl-toolchain`——编出一套完整可用的 DPC++ 工具链：

```bash
python llvm/buildbot/compile.py -j64                 # 主工具链（deploy-sycl-toolchain）
python llvm/buildbot/compile.py -j64 -t clang-format # clang-format（不在主目标里）
python llvm/buildbot/compile.py -j64 -t clangd       # clangd（不在主目标里）
```

- **`-j64`**：并行度，设成你机器的核数（或略高）。LLVM 首次全量构建很重，几十分钟到几小时不等，取决于机器。
- **`-t <target>`**：单独构建某个目标。`clang-format` 和 `clangd` **不属于** `deploy-sycl-toolchain`，要单独 `-t` 编。
- **增量构建**：改了几个 `.cpp` 再跑一次 `compile.py`，只重编受影响的部分，快得多。改了头文件可能触发大范围重编。

构建产物在 **`llvm/build/bin/`**（可执行：`clang++`、`llvm-spirv`、`sycl-ls`……）和 **`llvm/build/lib/`**（运行时库）。

## 3. 运行环境（PATH / LD_LIBRARY_PATH）

编出来的 `clang++` 要能找到运行时库（OpenCL/Level Zero loader、TBB、UMF……）才能编译和运行 SYCL 程序。开发机上这些运行时的路径通常集中在一个 `env.sh` 里 `source`。核心是把构建产物的 `bin`/`lib` 放进 `PATH`/`LD_LIBRARY_PATH`：

```bash
# 让 shell 找到你刚编出来的工具链
export PATH=/path/to/llvm/build/bin:$PATH
export LD_LIBRARY_PATH=/path/to/llvm/build/lib:$LD_LIBRARY_PATH
# 再 source 一份提供 OpenCL/L0/TBB 等运行时路径的 env.sh
source env.sh
```

<div class="warn">
<strong>构建目录和 <code>env.sh</code> 里的路径可能不一致——以实际构建位置为准</strong>
一个常见的坑：<code>env.sh</code> 里写死的 <code>PATH</code>/<code>LD_LIBRARY_PATH</code> 指向的构建目录，可能<strong>不是</strong>你这次真正构建的目录（比如 <code>env.sh</code> 指向 <code>/iusers/.../llvm/build</code>，而你实际编在 <code>/localdisk2/.../llvm/build</code>）。<strong>先 <code>source env.sh</code>，再用 <code>export</code> 把这两个变量覆盖成你实际的构建路径</strong>——顺序反了就会跑到旧二进制。运行前 <code>which clang++</code> 确认一下指向的是你刚编的那个。
</div>

验证工具链能看到设备：

```bash
sycl-ls        # 列出所有可用的 SYCL 设备（GPU/CPU/...）；空了说明运行时没配好
```

## 4. 编译并运行一个 SYCL 程序

拿全书最小的例子验证整条链通了：

```bash
cat > hello.cpp <<'EOF'
#include <sycl/sycl.hpp>
#include <cstdio>
int main() {
  sycl::queue q;
  std::printf("Running on: %s\n",
      q.get_device().get_info<sycl::info::device::name>().c_str());
  int n = 8; sycl::buffer<int,1> buf(n);
  q.submit([&](sycl::handler& h){
    sycl::accessor a(buf, h, sycl::write_only);
    h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i){ a[i] = i[0]*i[0]; });
  });
  sycl::host_accessor a(buf, sycl::read_only);
  for (int i=0;i<n;i++) std::printf("%d ", a[i]); std::printf("\n");
}
EOF

clang++ -fsycl hello.cpp -o hello    # -fsycl 触发第 12 章那套两趟编译
./hello
```

跑通了，说明你的工具链**编译期（`-fsycl`）+ 运行期（找到设备并执行）**两头都对了。想看中间产物，把正文各章的诊断标志加上（下节）。

## 5. 常用命令速查

把全书用到的"看内部"的命令汇总成一张表——它们是你验证正文每个论断的手术刀：

| 目的 | 命令 | 正文对应 |
|---|---|---|
| 只编设备端、出 LLVM IR | `clang++ -fsycl -fsycl-device-only -S -emit-llvm x.cpp -o -` | 第 14、15、21、22 章 |
| 看集成头文件 | `clang++ -fsycl -fsycl-device-only ... -Xclang -fsycl-int-header=x.h` | 第 14 章 |
| 列出设备 | `sycl-ls`（`sycl-ls --verbose` 更详细） | 第 16 章 |
| SPIR-V ↔ LLVM 互转 | `llvm-spirv x.bc -o x.spv` / `llvm-spirv -r x.spv` | 第 9 章 |
| 反汇编 bitcode | `llvm-dis x.bc -o -` | 第 4 章 |
| 单跑一个 LLVM Pass | `opt -passes=<pass> x.ll -S -o -` | 第 6、7 章 |
| 追设备库链接决策 | 在编译命令加 `-###`（打印实际子命令） | 第 19 章 |
| 运行时 UR 调用跟踪 | `SYCL_UR_TRACE=1 ./prog` | 第 18 章 |
| 打开设备代码 sanitizer | `-fsanitize=address`（配 compiler-rt） | 第 19 章 |
| 跑 LIT 测试 | `llvm-lit sycl/test/path/to/test.cpp` | 第 24 章 |

<div class="keypoint">
<strong>三步走：configure.py → compile.py → 覆盖 PATH/LD_LIBRARY_PATH</strong>
<strong>构建 DPC++ 不手写 CMake，走 <code>llvm/buildbot/</code> 的两个脚本：<code>configure.py</code>（<code>--build-type=Release --llvm-external-projects="clang-tools-extra;compiler-rt"</code>，只跑一次/改标志才重跑）→ <code>compile.py -j64</code>（默认 <code>deploy-sycl-toolchain</code>；<code>clang-format</code>/<code>clangd</code> 要单独 <code>-t</code>）。</strong>运行前 <code>source env.sh</code> 后用 <code>export</code> 把 <code>PATH</code>/<code>LD_LIBRARY_PATH</code> 覆盖成<strong>实际构建目录</strong>（<code>which clang++</code> 确认），再 <code>sycl-ls</code> 看到设备就绪。验证正文论断的手术刀是 <code>-fsycl-device-only -S -emit-llvm</code>（看 IR）、<code>llvm-spirv</code>（看 SPIR-V）、<code>SYCL_UR_TRACE=1</code>（看运行时调用）。
</div>

## 常见问题排查

- **`sycl-ls` 什么都不列**：运行时库没配好。检查 `LD_LIBRARY_PATH` 有没有 OpenCL/Level Zero loader；`OCL_ICD_FILENAMES` 指向的 `libigdrcl.so` 存不存在。
- **`clang++` 找不到或跑的是系统的**：`PATH` 里构建目录没排在前面，或被 `env.sh` 覆盖回旧路径。`which clang++` 确认。
- **链接期 `undefined reference` 到 SYCL 运行时**：`LD_LIBRARY_PATH` 里的 `libsycl.so` 和你编 `clang++` 的构建不是同一个。保持 `bin` 和 `lib` 来自同一次构建。
- **构建 OOM/太慢**：降 `-j`（链接阶段很吃内存），或配置时加 `--shared-libs` 让链接更轻。
- **改了源码没生效**：确认重跑了 `compile.py`，且跑的是新 `bin/clang++`（时间戳/`which`）。

## 小结

- intel/llvm 的 `sycl` 分支是全书源码锚点的家；构建走 **`llvm/buildbot/configure.py` + `compile.py`**，不手写 CMake。
- 配置：`--build-type=Release --llvm-external-projects="clang-tools-extra;compiler-rt"`，一次即可；构建：`compile.py -j64`（`clang-format`/`clangd` 另 `-t`）。
- 运行：`source env.sh` 后 **`export` 覆盖 `PATH`/`LD_LIBRARY_PATH` 到实际构建目录**，`sycl-ls` 验证设备，`clang++ -fsycl` 编 SYCL 程序。
- 看内部的常用工具：`-fsycl-device-only -S -emit-llvm`、`llvm-spirv`、`opt`、`sycl-ls`、`SYCL_UR_TRACE=1`、`llvm-lit`——正文每章论断都能用它们复现。
- 权威文档：`sycl/doc/GetStartedGuide.md`（构建/使用）、各子项目的 `CLAUDE.md`/README。这份附录只是"够用就走"的清单。
