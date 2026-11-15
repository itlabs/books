# 《深入 Triton》校验脚本

本书的代码要真的在 GPU 上跑过、IR 要真的 dump 出来才算数（BRIEF §13）。Triton 的 API、layout 名、pass 名版本一变就错，
光靠肉眼审不住。这里的脚本把各章的**完整代码块**抽出来，逐个跑一遍或编译一遍。

## 先准备环境

所有脚本都需要一个装好 `torch` 和 `triton` 的 Python（装法见附录 A）。脚本按这个顺序找 Python：
`$PYTHON` → 当前激活的 venv → PATH 里的 `python3`。

Intel GPU 上，Level Zero、`ocloc` 往往不在系统目录里，要先设几个环境变量：
把 `env-xpu.sh.example` 复制成 `env-xpu.sh`，填上本机路径，然后 `source` 它（每个变量为什么要设，文件里都有注释）。

`probe-env.py` 是只读的环境体检，共 14 项。换机器、换版本以后先跑它：

```bash
<venv>/bin/python books/inside-triton/verify/probe-env.py --helion
```

## 代码块

在**仓库根目录**运行（不带参数就是全书，也可以只传章名）：

```bash
books/inside-triton/verify/verify-triton.sh               # 全书
books/inside-triton/verify/verify-triton.sh ch12 ch13     # 只校验指定章
CE=1 books/inside-triton/verify/verify-triton.sh          # triton-ir 块改走 Compiler Explorer（会上传代码块）
VERBOSE=1 books/inside-triton/verify/verify-triton.sh     # 每块的输出都打印
```

| 围栏 | 怎么校验 |
| --- | --- |
| ` ```triton ` / ` ```gluon ` / ` ```helion ` | 看块首的 `# arch:`。本机满足就在单独的进程里真跑，退出码为 0 算通过；不满足就记为**跳过**。 |
| ` ```triton-ir ` | 默认用本机的 Triton（基线 3.9.0）**离线编译**，不需要对应的 GPU（见下一节），并检查 `# IR:` 指定的那一级确实编出来了。`CE=1` 时改走 CE 的 API（CE 上是 3.7.0），检查的是读者点按钮时实际看到的内容，包括 Device Viewer 能不能找到那一级。 |
| ` ```*-norun ` | 不校验：片段、伪代码，或者本书没在真机上验证过的写法。 |

`# arch:` 的取值：`any`、`sm80`、`sm86`、`sm90`、`sm100`（NVIDIA，都表示「该架构及以上」）、`cdna3`、`xpu`，
空格分隔，满足其中一个就跑。`build.js` 读的也是这一行。

**跳过不算通过。** 脚本最后会列出跳过的块，这些块要到满足 `# arch:` 的那台机器上再跑一次。
本书有两台验证机：A10 满足 `any sm80 sm86`，Max 1100 满足 `any xpu`。标了 `sm90`、`sm100` 的块，这两台都跑不了（BRIEF §14 第 1 项）。

## 不用 GPU 看 3.9.0 的 IR

`triton_ir_local.py` 的做法和 CE 的 Triton 编译器一样：张量换成 PyTorch 的 FakeTensor，GPU driver 换成一个只会报告目标架构的 mock，
kernel 只编译、不启动。所以在任何一台装了 Triton 的机器上（Intel 那台也行），都能拿到 NVIDIA 和 AMD 目标的真实 IR：

```bash
# 读块首 # CE: 行；默认 NVIDIA sm_86（A10）
python books/inside-triton/verify/triton_ir_local.py block.py --show ttgir
python books/inside-triton/verify/triton_ir_local.py block.py --arch 90 --show ttgir            # Hopper
python books/inside-triton/verify/triton_ir_local.py block.py --backend hip --arch gfx942 --out ir/
```

正文里的 ` ```ttgir-norun ` 块就用它来取：CE 上只有 3.7.0，而正文以 3.9.0 为准。

## minillm

```bash
books/inside-triton/verify/verify-minillm.sh              # 全部 pytest
books/inside-triton/verify/verify-minillm.sh -k rmsnorm   # 其余参数原样传给 pytest
```

不需要先 `pip install -e`：脚本用 `PYTHONPATH` 直接指向 `code/minillm/`，改完代码不用重装。

缺工具时脚本会干净地退出（退出码 127），并提示该怎么装。
