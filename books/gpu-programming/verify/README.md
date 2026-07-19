# GPU 代码校验脚本

本书的 `cuda` / `sycl` / `triton` / `cutedsl` 代码块需要真实工具链甚至 GPU 才能编译运行，
无法在浏览器里跑。这里的脚本把各章的**完整代码块**抽取出来，用对应工具链实际编译/运行，
用于在有相应环境的机器上验证"示例可编译可执行"。

三套脚本按技术分开，各自只依赖自己那套工具链——因为 nvcc、oneAPI、Triton 通常装在不同机器上。

## 用法

在**仓库根目录**运行（不带参数=校验全书；也可只传章名）：

```bash
# 需要 NVIDIA CUDA Toolkit（nvcc）。有 GPU 则编译+运行；无 GPU 用 NORUN=1 只编译。
books/gpu-programming/verify/verify-cuda.sh
books/gpu-programming/verify/verify-cuda.sh ch09 ch11      # 只校验指定章
NORUN=1 books/gpu-programming/verify/verify-cuda.sh        # 只编译不运行（无 GPU）
ARCH=sm_75 books/gpu-programming/verify/verify-cuda.sh     # 指定 GPU 架构（默认 sm_70）

# 不指定 CXX 时自动探测编译器：icpx → clang++ → acpp，取第一个支持 -fsycl 的。
#   · Intel oneAPI 的 icpx——先 source 环境：source /opt/intel/oneapi/setvars.sh
#   · intel/llvm 源码构建的 clang++（开源 DPC++）——把 <build>/bin 加进 PATH，或用 CXX 指全路径
#   · AdaptiveCpp 的 acpp
books/gpu-programming/verify/verify-sycl.sh                          # 自动探测
CXX=$HOME/sycl_workspace/llvm/build/bin/clang++ books/gpu-programming/verify/verify-sycl.sh   # intel/llvm clang++
SYCL_TARGETS=spir64 books/gpu-programming/verify/verify-sycl.sh      # 显式指定 -fsycl-targets（intel/llvm 常需）
ONEAPI_DEVICE_SELECTOR=opencl:cpu books/gpu-programming/verify/verify-sycl.sh   # 强制 CPU 设备（无 GPU）
CXX=acpp books/gpu-programming/verify/verify-sycl.sh                 # 改用 AdaptiveCpp

# 需要 pip install triton torch + NVIDIA GPU。无 GPU 时自动降级为语法检查。
books/gpu-programming/verify/verify-triton.sh
SYNTAX=1 books/gpu-programming/verify/verify-triton.sh     # 只做语法检查，不真跑
```

工具链缺失时脚本会干净退出（退出码 127）并提示安装方式——安装步骤见**附录 A**。
校验通过时退出码为 0，任一块失败为 1，可直接接入 CI。

## 文件说明

- `extract-blocks.js` —— 共享的代码块提取器（Node）。按语言从章节 md 抽取完整块（不含 `-norun` 片段）到临时目录，并标注哪些是可独立运行的（有 `main` 的 C/C++，或有顶层可执行语句的 Python driver）。
- `verify-cuda.sh` —— `cuda` 块用 `nvcc` 编译（有 `main` 则运行）；`cuda-ptx` 块只生成 PTX（不需要 GPU）。
- `verify-sycl.sh` —— `sycl` 块用 `icpx -fsycl`（或 `acpp`）编译并运行。
- `verify-triton.sh` —— `triton` driver 块真跑 `python`；kernel 片段做 AST 语法检查；无 GPU 自动降级。

> 这些脚本随书入库，方便你在任何一台装好对应工具链的机器上直接验证；它们不参与 `node build.js` 的网页构建。
