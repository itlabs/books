# 附录 A 搭建：构建 LLVM/MLIR、out-of-tree 模板与常用命令

## 这份附录怎么用

<div class="goal-box">
<p>正文讲"为什么这么设计"，这份附录帮你"把环境弄好"。全书每个源码锚点（<code>mlir/include/mlir/IR/Operation.h</code>、<code>DialectConversion.cpp</code> 里的 <code>OperationLegalizer</code>……）都指向上游 <strong><code>llvm/llvm-project</code> 的 <code>mlir/</code> 子树</strong>；第 7 章起的 pix 方言又需要一份<strong>装好的</strong> MLIR 才能编出 <code>pix-opt</code>。这份附录把这条路走一遍：拿到 MLIR 的三条路、源码构建的 CMake 参数、编出来的工具各管什么、out-of-tree 项目模板，以及日常命令速查。</p>
<p>它不是官方文档的替代（<a href="https://mlir.llvm.org/getting_started/">mlir.llvm.org/getting_started</a> 永远是权威），而是一份"够用就走"的清单。<strong>本书的版本基线写在下面第 0 节，请先看一眼。</strong></p>
</div>

## 0. 版本基线（全书按这个写）

MLIR 是 LLVM 里演进最快的部分：pass 名会改、选项会加、工具会更名。本书正文统一按这个基线写：

| 项 | 本书取值 |
| --- | --- |
| LLVM / MLIR 版本 | **23.1.x**（23.1.1 发布于 2026-09-08） |
| 源码仓库 | `https://github.com/llvm/llvm-project` |
| 源码锚点前缀 | `mlir/include/mlir/...`、`mlir/lib/...` |
| 网页代码块跑的工具 | Compiler Explorer 的 **trunk** 版 `mlir-opt` / `mlir-translate` |

两点提醒，后面不再重复：

- **正文引用的是类名和函数名，不是行号。** 行号一个 commit 就过期；`grep -rn "class OperationLegalizer" mlir/` 永远有效。
- **网页按钮用的是 trunk，你手上可能是发布版。** 极少数情况下（某个 pass 刚改名）两者会不一致——正文会在这类地方用 <code>warn</code> 框提醒你去 `mlir-opt --help` 核对。拿不准某个选项在你的版本里还存不存在时，最快的办法就是直接问工具：

```bash
mlir-opt --help | grep -i bufferize     # 这个 pass 在你的版本里叫什么、有哪些选项
mlir-opt --show-dialects                # 这份二进制注册了哪些方言
```

<div class="warn">
<strong>几个已知的"版本地雷"</strong>
<p><code>mlir-cpu-runner</code> 在较新的 LLVM 里已更名为 <strong><code>mlir-runner</code></strong>（第 24 章 JIT 那一步会用到）。<code>--one-shot-bufferize</code> 的选项名（第 22 章）与 <code>transform</code> 方言的 op 名（第 18 章）在历史上都动过。<code>--convert-*-to-*</code> 这一族 pass 时有合并拆分。遇到"命令报 Unknown command line argument"，先别怀疑自己，去 <code>--help</code> 里搜一下。</p>
</div>

## 1. 拿到 MLIR 的三条路

<div class="concept">
<svg viewBox="0 0 720 300" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">
  <defs>
    <marker id="a-arr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 z" fill="#6b7280"/>
    </marker>
  </defs>
  <text x="14" y="24" font-size="13" font-weight="bold" fill="#24292f">三条路，按你要做什么选</text>
  <rect x="14" y="44" width="200" height="66" rx="8" fill="#eef4fb" stroke="#2b6cb0"/>
  <text x="28" y="66" font-size="12" font-weight="bold" fill="#2b6cb0">① 发行版包</text>
  <text x="28" y="84" font-size="11" fill="#24292f">十分钟装好，能跑 mlir-opt</text>
  <text x="28" y="100" font-size="11" fill="#24292f">读第一、三、四部分够用</text>
  <rect x="14" y="124" width="200" height="66" rx="8" fill="#eef4fb" stroke="#2b6cb0"/>
  <text x="28" y="146" font-size="12" font-weight="bold" fill="#2b6cb0">② 源码构建</text>
  <text x="28" y="164" font-size="11" fill="#24292f">几十分钟～几小时</text>
  <text x="28" y="180" font-size="11" fill="#24292f">要造方言、要读源码就走这条</text>
  <rect x="14" y="204" width="200" height="66" rx="8" fill="#f5f5f5" stroke="#6b7280"/>
  <text x="28" y="226" font-size="12" font-weight="bold" fill="#6b7280">③ 什么都不装</text>
  <text x="28" y="244" font-size="11" fill="#24292f">点代码块的 CE 按钮</text>
  <text x="28" y="260" font-size="11" fill="#24292f">通勤路上读也能动手</text>
  <line x1="214" y1="77" x2="300" y2="140" stroke="#6b7280" marker-end="url(#a-arr)"/>
  <line x1="214" y1="157" x2="300" y2="150" stroke="#6b7280" marker-end="url(#a-arr)"/>
  <line x1="214" y1="237" x2="300" y2="160" stroke="#6b7280" marker-end="url(#a-arr)"/>
  <rect x="308" y="112" width="150" height="76" rx="8" fill="#fff" stroke="#2b6cb0" stroke-width="2"/>
  <text x="330" y="140" font-size="13" font-weight="bold" fill="#2b6cb0">mlir-opt</text>
  <text x="322" y="160" font-size="11" fill="#24292f">读 IR → 跑 pass</text>
  <text x="322" y="176" font-size="11" fill="#24292f">→ 打印 IR</text>
  <line x1="458" y1="150" x2="530" y2="150" stroke="#6b7280" marker-end="url(#a-arr)"/>
  <rect x="538" y="60" width="168" height="54" rx="8" fill="#f0f8f2" stroke="#3f9b5c"/>
  <text x="552" y="82" font-size="12" font-weight="bold" fill="#3f9b5c">第 1–6、13–23 章</text>
  <text x="552" y="100" font-size="11" fill="#24292f">读 IR、跑 pass 就够</text>
  <rect x="538" y="124" width="168" height="54" rx="8" fill="#fdf1ec" stroke="#e06a3b"/>
  <text x="552" y="146" font-size="12" font-weight="bold" fill="#e06a3b">第 7–12 章（pix）</text>
  <text x="552" y="164" font-size="11" fill="#24292f">必须要 ② 的安装树</text>
  <rect x="538" y="188" width="168" height="54" rx="8" fill="#fdf1ec" stroke="#e06a3b"/>
  <text x="552" y="210" font-size="12" font-weight="bold" fill="#e06a3b">第 24–25 章</text>
  <text x="552" y="228" font-size="11" fill="#24292f">JIT / GPU 出口要 ②</text>
</svg>
<div class="caption">图 A-1　三条路与它们分别够读哪些章</div>
</div>

### ① 发行版包（最快）

Debian/Ubuntu 用 LLVM 官方 apt 源，装工具 + 开发头文件：

```bash
# 官方脚本按版本号装（把 23 换成你要的版本）
bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)" -- 23
sudo apt install mlir-23-tools libmlir-23-dev llvm-23-dev
# 工具带版本后缀，做个软链或用全名
mlir-opt-23 --version
```

macOS 上 `brew install llvm` 自带 MLIR 工具（`$(brew --prefix llvm)/bin`）。包名与版本后缀各发行版不同，`apt search mlir` / `brew info llvm` 确认一下。

这条路能跑全书绝大多数代码块，也够 `verify-mlir.sh` 用。但要编 out-of-tree 方言（第 7 章起），需要 `libmlir-*-dev` 带的 `MLIRConfig.cmake`——包装得全不全各发行版有差异，编不过就走第 ② 条。

### ② 源码构建（要造方言、要读源码就走这条）

```bash
git clone --depth=1 https://github.com/llvm/llvm-project.git
cd llvm-project
cmake -G Ninja -S llvm -B build \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_USE_LINKER=lld \
  -DLLVM_CCACHE_BUILD=ON
ninja -C build -j"$(nproc)"
```

每个参数都有理由，别照抄了不看：

- **`LLVM_ENABLE_PROJECTS="mlir"`**：MLIR 是 monorepo 里的一个 project，不开这个开关它不参与构建。（想同时玩 ClangIR 就加 `clang`，见第 28 章。）
- **`LLVM_ENABLE_ASSERTIONS=ON`**：**强烈建议开**，哪怕 Release 构建。两个理由：一是 MLIR 的很多不变式（第 10 章讲的 verifier、pattern 的前置条件）靠断言兜住，关了断言，写错的变换不会当场炸，而是给你一段莫名其妙的 IR；二是 **`--debug-only=` 这个选项只在开了断言的构建里存在**——关掉断言的二进制上它直接报 `Unknown command line argument`，而第 14、17 章调试 pattern 和 conversion 全靠它。这不是调试选项，是 MLIR 开发的必需品。
- **`LLVM_TARGETS_TO_BUILD="host;NVPTX"`**：`host` 够跑第 24 章的 JIT；`NVPTX` 是第 25 章 GPU 出口之一（AMD 的话加 `AMDGPU`）。全都编（`all`）会显著拖慢构建。
- **`LLVM_INSTALL_UTILS=ON`**：把 `lit`、`FileCheck` 一起装出来——第 12 章给 pix 写测试要用。
- **`LLVM_USE_LINKER=lld` / `LLVM_CCACHE_BUILD=ON`**：链接和重编快很多，装了 `lld`/`ccache` 就加上。
- 想**跟着源码下断点**看 pass 怎么跑，用 `-DCMAKE_BUILD_TYPE=RelWithDebInfo`（Debug 慢且二进制巨大）。
- 想用 Python 探索 IR（第 3 章会提一句），加 `-DMLIR_ENABLE_BINDINGS_PYTHON=ON`。

构建产物在 `build/bin/`。不装到系统也能用，本书所有命令都支持直接指到这里：

```bash
export PATH="$PWD/build/bin:$PATH"          # 或者
export MLIR_BIN="$PWD/build/bin"            # verify-mlir.sh 认这个变量
```

跑一下 MLIR 自己的测试，确认构建是好的：

```bash
ninja -C build check-mlir
```

### ③ 什么都不装

本书的 `mlir` 代码块右上角有 **🔗 在 Compiler Explorer 打开（mlir-opt）**，点开就是 trunk 版 `mlir-opt` 在跑，改一行立刻看结果。第一、三部分的概念章这样读完全够；另有浏览器里直接运行的 Python 模拟器讲机制。只有第 7–12 章（造方言）和第 24–25 章（落地）真的需要本地工具链。

## 2. 编出来的工具各管什么

| 工具 | 干什么 | 本书哪里用 |
| --- | --- | --- |
| `mlir-opt` | 读 `.mlir` → 跑 pass/管线 → 打印 `.mlir`。**全书主力** | 第 3 章起处处 |
| `mlir-translate` | IR ↔ 外部格式互转（如 `--mlir-to-llvmir`） | 第 24 章 |
| `mlir-tblgen` | 把 `.td`（ODS）生成 C++ 头/实现 | 第 7–9 章 |
| `mlir-runner` | 把 IR 交给 JIT 直接跑（旧名 `mlir-cpu-runner`） | 第 24 章 |
| `mlir-lsp-server` | 编辑器里的 MLIR 补全/跳转（VS Code 装 "MLIR" 插件） | 写 pix 时的生产力工具 |
| `mlir-reduce` | 自动缩小一个触发崩溃的 IR 输入 | 第 17 章调试 |
| `lit` / `FileCheck` | 测试驱动与输出匹配 | 第 12 章 |

## 3. out-of-tree 项目模板

第 7 章会把这套骨架逐行拆开讲，这里先给出最小形态，方便你现在就确认环境是通的。关键只有一句 `find_package(MLIR)`：

```cmake
# CMakeLists.txt 的开头
cmake_minimum_required(VERSION 3.20)
project(pix LANGUAGES CXX C)

find_package(MLIR REQUIRED CONFIG)          # ← 由 MLIR_DIR 指过来
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)
include(TableGen)
```

配置时把 `MLIR_DIR` 指到安装树或构建树里的 `lib/cmake/mlir`：

```bash
cmake -G Ninja -S books/inside-mlir/code/pix -B /tmp/pix-build \
  -DMLIR_DIR=$HOME/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=$HOME/llvm-project/build/lib/cmake/llvm \
  -DCMAKE_BUILD_TYPE=Release
ninja -C /tmp/pix-build
```

本书带了脚本替你做这件事（含参数探测与错误提示）：

```bash
books/inside-mlir/verify/verify-pix.sh          # 构建 code/pix 并跑 lit 测试
```

## 4. 常用命令速查

```bash
# —— 看 IR 与 pass ——
mlir-opt input.mlir                                  # 只解析 + verify + 打印回来（round-trip）
mlir-opt input.mlir --verify-roundtrip               # 再确认打印出来的还能重新解析成同一个 IR
mlir-opt input.mlir --mlir-print-op-generic          # 用通用格式打印（第 3 章：通用格式才是真相）
mlir-opt input.mlir --canonicalize --cse             # 按顺序跑几个 pass
mlir-opt input.mlir --pass-pipeline='builtin.module(func.func(canonicalize,cse))'
                                                     # 嵌套管线：第 13 章讲它为什么长这样
mlir-opt input.mlir --canonicalize --mlir-print-ir-after-all   # 每个 pass 后都打印一遍
mlir-opt input.mlir --canonicalize --mlir-print-ir-after-change # 只在 IR 真变了时打印
mlir-opt input.mlir --allow-unregistered-dialect     # 允许没注册的方言（手写实验 IR 时很方便）

# —— 出问题时 ——
# 注意：--debug-only 需要开了断言的构建（见上文），且 DEBUG_TYPE 名以源码里的
# `#define DEBUG_TYPE` 为准（grep 一下就知道）。网页上的 CE 按钮跑的是关断言的构建，没有这个选项。
mlir-opt input.mlir --debug-only=greedy-rewriter     # 看 pattern 驱动到底在试什么（第 14 章）
mlir-opt input.mlir --debug-only=dialect-conversion  # 看合法化卡在哪个 op（第 17 章）
mlir-opt input.mlir --mlir-print-ir-tree-dir=/tmp/ir # 每步 IR 落盘成目录树，便于 diff
mlir-opt input.mlir --mlir-pass-pipeline-crash-reproducer=/tmp/repro.mlir

# —— 查清单 ——
mlir-opt --show-dialects                             # 注册了哪些方言
mlir-opt --help | less                               # 所有 pass 与选项（最终真相）

# —— 落地 ——
mlir-opt input.mlir --convert-to-llvm                # 下降到 llvm 方言（第 24 章）
mlir-translate llvm.mlir --mlir-to-llvmir            # llvm 方言 → 真正的 .ll
mlir-runner input.mlir -e main --entry-point-result=void \
  --shared-libs=$MLIR_BIN/../lib/libmlir_c_runner_utils.so     # JIT 直接跑

# —— 生成 ODS 代码（第 8 章会看生成出来的东西）——
mlir-tblgen -gen-op-decls -I $(llvm-config --includedir) PixOps.td
mlir-tblgen -gen-op-defs  -I $(llvm-config --includedir) PixOps.td
mlir-tblgen -gen-dialect-doc -I $(llvm-config --includedir) PixOps.td   # 自动生成方言文档
```

## 5. 校验本书的代码

书里的 MLIR 代码块不是抄来的伪 IR，都过了工具。你也可以自己验一遍：

```bash
books/inside-mlir/verify/verify-mlir.sh                 # 需要本地 mlir-opt
CE=1 books/inside-mlir/verify/verify-mlir.sh            # 没装本地工具，走 Compiler Explorer
books/inside-mlir/verify/verify-mlir.sh ch16 ch17       # 只验指定章
```

细节（`// RUN:` 行的约定、各语言默认参数）见 `books/inside-mlir/verify/README.md`。

## 6. 顺手配好的两件小事

- **编辑器**：VS Code 装 MLIR 官方插件，指向你的 `mlir-lsp-server`；`.mlir` 里能补全 op 名、跳转到方言定义、悬停看 op 文档。写 pix 时它比 grep 快得多。
- **一份可 grep 的源码**：全书的源码锚点都假设你手边有 `llvm-project`。哪怕你走的是发行版包那条路，也建议 `git clone --depth=1` 一份源码放着——读到 `RewritePattern` 就去 `grep -rn "class RewritePattern" mlir/include/`，亲眼确认比记住结论重要。

---

**这份附录的要点**

- 本书基线是 **LLVM/MLIR 23.1.x**，源码锚点用类名与函数名而非行号；工具的 `--help` 永远是最终真相。
- 拿到 MLIR 有三条路：发行版包（快）、源码构建（要造方言必走）、什么都不装（点网页上的 Compiler Explorer 按钮）。
- 源码构建**务必开 `LLVM_ENABLE_ASSERTIONS=ON`**——MLIR 的不变式靠断言兜住，关了就从"当场报错"退化成"结果诡异"。
- out-of-tree 项目的全部秘密就是 `find_package(MLIR REQUIRED CONFIG)` + `MLIR_DIR`，第 7 章详解。
- `mlir-opt` 是全书主力；调试三件套是 `--mlir-print-ir-after-change`、`--debug-only=…`、`--mlir-print-ir-tree-dir`。
