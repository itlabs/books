# 附录 A 搭好环境：编译器、构建与在线工具

本书的代码块都配了 **🔗 在 Compiler Explorer 打开** 按钮，点开即可在线编译运行，不装任何东西也能学完全书。但要认真写 C++，你迟早需要一套本地环境。这份附录帮你快速搭好。

## 最快上手：只用浏览器

如果你现在只想跟着书跑代码，**什么都不用装**：

- 点每段代码的 **🔗 在 Compiler Explorer 打开**，在 [godbolt.org](https://godbolt.org) 里直接编译、运行、改代码重跑。
- 想看编译器生成的汇编（"编译器到底做了什么"），点 **🔬 看汇编 / 编译器处理**。

这足以支撑你读完全书、做完所有练习。下面的本地环境，等你想写自己的项目时再装。

## 本地编译器

C++ 有三大主流编译器，任选其一：

| 编译器 | 平台 | 安装 |
|---|---|---|
| **GCC**（`g++`） | Linux / macOS / Windows(MSYS2) | 见下 |
| **Clang**（`clang++`） | Linux / macOS / Windows | 见下 |
| **MSVC**（`cl`） | Windows | 装 Visual Studio |

<div class="keypoint">
<strong>确保编译器够新（支持 C++20/23）</strong>
本书用到 concepts、ranges、<code>std::expected</code> 等现代特性，需要较新的编译器：<strong>GCC ≥ 13、Clang ≥ 16、MSVC ≥ VS2022</strong>。版本太老会报"不认识的特性"。装完用 <code>g++ --version</code> 确认。
</div>

**Linux（Ubuntu/Debian）**：

```bash
sudo apt update
sudo apt install g++ cmake
g++ --version        # 确认版本 >= 13
```

**macOS**：

```bash
xcode-select --install    # 装 Apple Clang（够用）
# 或用 Homebrew 装新版 GCC / LLVM：
brew install gcc llvm cmake
```

**Windows**：推荐两条路——装 [Visual Studio](https://visualstudio.microsoft.com/)（勾选"使用 C++ 的桌面开发"，自带 MSVC + 调试器），或装 [MSYS2](https://www.msys2.org/) 用其中的 GCC。若用 WSL，则按上面 Linux 的方式装即可。

## 编译与运行一个程序

把代码存成 `main.cpp`，然后：

```bash
g++ -std=c++23 -O2 -Wall -Wextra main.cpp -o app
./app                 # Windows 上是 app.exe
```

各选项的意思（本书所有代码块下方的"本地编译运行"命令就是这套）：

<div class="keypoint">
<strong>常用编译选项</strong>
<ul>
<li><code>-std=c++23</code>：用 C++23 标准（编译器较老可退到 <code>-std=c++20</code>）。<strong>不加会默认用老标准，现代特性会报错。</strong></li>
<li><code>-O2</code>：开启优化（第 19 章）。<strong>测性能必须加</strong>；日常调试可用 <code>-O0</code> 或 <code>-Og</code>。</li>
<li><code>-Wall -Wextra</code>：打开常用警告。<strong>强烈建议一直开着</strong>——它能帮你抓住悬垂引用、未初始化变量等一大类 bug（第 4 章）。</li>
<li><code>-pthread</code>：用到 <code>&lt;thread&gt;</code>（第 20 章）时加上。</li>
</ul>
</div>

调试相关的两个附加选项，排查内存问题时极有用：

```bash
# 开调试信息 + 地址消毒器：能在运行时抓出越界、悬垂、泄漏
g++ -std=c++23 -g -fsanitize=address,undefined main.cpp -o app
./app
```

`-fsanitize=address,undefined`（ASan/UBSan）会在程序真正踩到未定义行为（越界访问、使用已释放内存、有符号溢出等）时**立刻报告并指出位置**——这是现代 C++ 最有用的排错工具之一，建议养成用它跑测试的习惯。

## 用 CMake 管理多文件项目

单文件用 `g++ main.cpp` 就够。项目一旦有多个 `.cpp`（回忆第 2 章的编译/链接模型），就该用构建系统。**CMake** 是事实标准。一个最小的 `CMakeLists.txt`：

```cpp-norun
cmake_minimum_required(VERSION 3.20)
project(MyApp)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(app main.cpp helper.cpp)   # 列出所有 .cpp
```

构建它：

```bash
cmake -B build              # 在 build/ 目录生成构建文件
cmake --build build         # 编译
./build/app                 # 运行
```

CMake 会自动处理"每个 `.cpp` 独立编译成 `.o`、再链接到一起"（第 2 章）的全过程，还能管理依赖库、增量编译。IDE（VS、CLion、VS Code）都原生支持它。

## 编辑器与 IDE

任选：

- **VS Code** + C/C++ 扩展 + CMake Tools：轻量、跨平台，最流行的免费组合。
- **CLion**：JetBrains 出品（用过 IntelliJ / PyCharm 会很熟悉），开箱即用，付费。
- **Visual Studio**（Windows）：最强的 C++ 调试体验，社区版免费。

<div class="note">
<strong>务必配上 clangd 或 IntelliSense</strong>
现代 C++ 编辑器的核心是<strong>语言服务器</strong>（clangd，或 VS 的 IntelliSense）——它提供补全、跳转、实时错误提示、以及对模板/concepts 的理解。配好它，写 C++ 的体验会天差地别。VS Code 装 clangd 扩展、或微软的 C/C++ 扩展即可。
</div>

## 几个值得知道的工具

真正做项目时，这些工具会成为你的日常：

- **clang-format**：自动格式化代码（团队统一风格）。
- **clang-tidy**：静态检查，能揪出很多"能编译但有问题"的写法。
- **AddressSanitizer / UBSan**：上面讲过的运行时消毒器，抓内存/未定义行为。
- **Valgrind**（Linux）：另一种内存检测工具，较慢但无需重新编译。
- **包管理**：[vcpkg](https://vcpkg.io) 或 [Conan](https://conan.io)，用来引入第三方库（C++ 没有 Maven/pip 那样的官方中央仓库，这两个是社区主流方案）。

<div class="keypoint">
<strong>一句话建议</strong>
入门阶段：<strong>用 Compiler Explorer 学书 → 装 GCC/Clang + VS Code + clangd 写小程序 → 项目变大再上 CMake</strong>。别一开始就被工具链劝退——先把语言学会，工具随需而学。
</div>
