# 如何构建和阅读

本仓库是一个**多书**工程：每本书的正文用 Markdown 写在 `books/<书名>/chapters/` 里，
通过统一的构建脚本转成交互式网页，输出到 `docs/`，可直接用浏览器打开，也能发布到 GitHub Pages。

## 目录结构

```
books/
├── build.js            构建脚本：遍历 books/ 下每本书，Markdown → 交互式 HTML
├── pack.js             打包脚本：把某一本压成离线 zip
├── STYLE.md            写作规范（改书/加章时看它）
├── shared/             所有书共用的资源
│   ├── style.css       样式
│   ├── runner.js       浏览器内运行 Python 的脚本
│   └── vendor/         Prism.js 高亮（本地，离线可用）
├── books/              每本书一个子目录
│   └── python-teens/
│       ├── book.json   书的元数据：标题、章节顺序与导航标题
│       ├── README.md   这本书的封面 + 目录页
│       └── chapters/   章节 Markdown 源（ch01.md … appendix-d.md）
└── docs/               构建产物（发布到 GitHub Pages）
    ├── index.html      书架首页（自动列出所有书）
    └── python-teens/   每本书一个子目录
        ├── index.html  本书目录页
        ├── ch01.html … 各章
        ├── assets/     从 shared/ 复制来的 style.css / runner.js
        └── vendor/     从 shared/vendor/ 复制来的高亮文件
```

> GitHub Pages 指向 `docs/`；打开站点先看到书架首页，点进任意一本即可阅读。

## 怎么构建

装好 Node.js 后，在本目录运行：

```bash
npm install          # 只需第一次
node build.js        # 构建全部书 + 书架首页
node build.js python-teens   # 只构建某一本（书架首页仍会刷新）
```

## 怎么打包成离线 zip 分享

```bash
node pack.js python-teens    # 生成 python-teens-offline.zip
```

只有一本书时可省略书名参数。zip 解压后双击 `index.html` 即可离线阅读。

## 怎么新增一本书

1. 在 `books/` 下新建一个目录，如 `books/<新书名>/`。
2. 放入三样东西：
   - `book.json` —— 抄 `python-teens/book.json` 改：`slug`、`title`、`subtitle`、
     `description`，以及 `chapters` 数组（每项 `["文件名不带后缀", "导航显示的标题"]`，按顺序）。
   - `README.md` —— 本书封面 + 目录（指向 `chapters/xxx.md` 的链接会自动改写成 `.html`）。
   - `chapters/` —— 章节 Markdown 源文件。
3. 运行 `node build.js`，新书会自动出现在书架首页上。

样式、代码运行、语法高亮、离线打包等能力所有书自动共用，无需重复配置。

## 怎么给一本书增改章节

1. 在该书的 `chapters/` 编辑或新增 `.md` 文件（写作请遵循 `STYLE.md`）。
2. 如果是新章节，在该书的 `book.json` 的 `chapters` 数组里按顺序加一行。
3. 重新 `node build.js`（或 `node build.js <书名>`）。

### 代码块约定（重要）

**Python 书（python-teens）：**
- ` ```python ` —— 会带"运行"按钮，能在浏览器里跑（Pyodide）。**必须是完整、可独立运行**的代码。
- ` ```python-norun ` —— 只高亮、不给运行按钮。用于：故意报错的示范、需要联网/密钥的 AI 代码、turtle/pygame 代码。

**C++ 书（modern-cpp）：**
- ` ```cpp ` —— 高亮 + "🔗 在 Compiler Explorer 打开"按钮（点开即在线编译运行、可改代码重跑）+ 附本地编译命令。**应是完整可编译的程序。**
- ` ```cpp-asm ` —— 同上，但按钮是"🔬 看汇编 / 编译器处理"，godbolt 默认 `-O2` 并展开汇编面板。用于讲优化、内联、模板实例化、RVO 等"编译器处理"的地方。
- ` ```cpp-norun ` —— 只高亮、不给按钮。用于片段、故意报错示范、伪代码。

> godbolt 链接用 `clientstate`（base64 编码的会话 JSON）生成，携带源码、编译器 `g142`(gcc 14.2)、`-std=c++23 -O2 -Wall` 参数和运行面板。点开即联网编译；离线时代码块与本地编译命令仍完整可读。默认编译器/参数可在 `build.js` 的 `godboltUrl()` 里改。

**GPU 书（gpu-programming）：**
- ` ```cuda ` —— CUDA C++，"🔗 在 Compiler Explorer 打开（看 PTX）" + `nvcc` 本地命令。**应是完整可编译的 .cu 程序。**
- ` ```cuda-ptx ` —— 同上，按钮为"🔬 看 PTX / 编译器处理"。用于只展示 kernel（无需 host/main，编到 PTX 即可）。
- ` ```sycl ` —— SYCL/DPC++，Compiler Explorer（`icx -fsycl`）+ `icpx` 本地命令。完整可编译程序。
- ` ```triton ` / ` ```cutedsl ` —— Triton / CuTe DSL（Python），只高亮 + `pip` 本地命令（需 NVIDIA GPU）。
- ` ```cuda-norun ` / ` ```sycl-norun ` / ` ```triton-norun ` / ` ```cutedsl-norun ` —— 只高亮、不给按钮。用于片段、伪代码。
- ` ```python ` —— 会在浏览器里真运行（Pyodide），本书用来做**概念模拟器**（SIMT 调度、warp 分歧、访存、归约树等）。必须完整、可独立运行、仅标准库。

> **GPU 代码校验（随书入库，`books/gpu-programming/verify/`）**：GPU 代码需真实工具链/硬件、无法在浏览器里跑，故校验脚本按技术分开，各自只依赖自己那套（nvcc / oneAPI / Triton 常在不同机器上）：
> ```bash
> books/gpu-programming/verify/verify-cuda.sh    # 需 nvcc；有 GPU 则编译+运行，NORUN=1 只编译
> books/gpu-programming/verify/verify-sycl.sh    # 需 icpx（或 CXX=acpp）；默认可在 CPU 设备上跑
> books/gpu-programming/verify/verify-triton.sh  # 需 triton+torch+GPU；无 GPU 自动降级为语法检查（SYNTAX=1）
> ```
> 三者共用 `verify/extract-blocks.js` 从章节抽取完整代码块。工具链缺失时脚本干净退出（127）并给出安装提示（见附录 A）。不带参数校验全书，也可只传章名如 `verify-cuda.sh ch09 ch11`。详见 `verify/README.md`。这些脚本随书入库、便于在任何装好工具链的机器上验证，但不参与 `node build.js` 网页构建。

**通用：**
- ` ```bash ` —— 终端命令，只高亮。
