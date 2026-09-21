#!/usr/bin/env python3
"""校验正文里「贴出来的运行结果」是不是真的。

`verify-mlir.sh` 只保证代码块**能跑**；这个脚本进一步保证**贴出来的输出和实跑
一致**——抄漏一行、改完代码忘了更新输出，都会被它抓到（第 13 章就是这么抓到
一处漏抄的）。

用法（在仓库根目录）：
    MLIR_BIN=<llvm-build>/bin PIX_OPT=<pix-build>/bin/pix-opt \
        python3 books/inside-mlir/verify/verify-outputs.py
    …… 末尾可以带章名，只查指定章：ch11 ch13

规则：
  · 只比对**完整**输出（以 `module {` 开头的那种）。正文里常有"只看那两行"
    式的节选，那种没法机械比对，会被列出来提示人工核。
  · 尾随换行不计：mlir-opt 的输出末尾带一个空行，那是格式不是内容。
  · 工具与参数取自块首的 `// RUN:` 行（和 build.js、verify-mlir.sh 同一套约定）；
    `|` 之后的部分（比如 FileCheck）由 lit 测试负责，这里只跑前半段。
"""
import difflib
import glob
import os
import re
import subprocess
import sys

MLIR_BIN = os.environ.get("MLIR_BIN", "")
PIX_OPT = os.environ.get("PIX_OPT", "")
CHAPTER_DIR = "books/inside-mlir/chapters"

# 代码块与输出块都不许跨过下一个 ``` 栅栏，否则会把中间的正文一起吞进来。
BLOCK_RE = re.compile(
    r"```mlir(?:-lower)?\n((?:(?!```)[\s\S])*)```\n\n运行结果[^\n]*\n\n```\n"
    r"((?:(?!```)[\s\S])*)```"
)


def tool_and_args(code):
    """从块首的 RUN 行取工具名与参数；没有 RUN 行就按约定给默认值。"""
    first = code.splitlines()[0] if code.splitlines() else ""
    m = re.search(r"//\s*RUN:\s*(\S+)\s*%s\s*(.*)", first)
    if m:
        name, args = m.group(1), m.group(2)
    else:
        name = "pix-opt" if "pix." in code else "mlir-opt"
        args = "--canonicalize"
    args = args.split("|")[0].strip()
    exe = PIX_OPT if name == "pix-opt" else os.path.join(MLIR_BIN, name)
    return name, exe, [a for a in args.split() if a]


def main():
    wanted = set(sys.argv[1:])
    paths = sorted(glob.glob(f"{CHAPTER_DIR}/*.md"))
    if wanted:
        paths = [p for p in paths if os.path.basename(p)[:-3] in wanted]

    total = exact = partial = fail = 0
    for path in paths:
        name = os.path.basename(path)
        for m in BLOCK_RE.finditer(open(path, encoding="utf-8").read()):
            code, expected = m.group(1), m.group(2)
            total += 1
            if not expected.lstrip().startswith("module {"):
                partial += 1
                print(f"  － {name}: 节选输出，跳过自动比对（请人工核）")
                continue

            tool, exe, args = tool_and_args(code)
            if not os.path.exists(exe):
                print(f"  ✗ 找不到 {tool}：{exe}")
                return 127
            with open("/tmp/_verify_outputs.mlir", "w", encoding="utf-8") as f:
                f.write(code)
            run = subprocess.run([exe, "/tmp/_verify_outputs.mlir", *args],
                                 capture_output=True, text=True)
            got, want = run.stdout.rstrip("\n"), expected.rstrip("\n")
            if got == want:
                exact += 1
                print(f"  ✓ {name}: 逐字节一致  [{tool} {' '.join(args)}]")
                continue

            fail += 1
            print(f"  ✗ {name}: 与实跑不一致  [{tool} {' '.join(args)}]")
            for line in list(difflib.unified_diff(
                    want.splitlines(), got.splitlines(), "正文", "实跑",
                    lineterm=""))[:16]:
                print("     ", line)
            if run.stderr.strip():
                print("      stderr:", run.stderr.strip().splitlines()[0])

    print(f"\n输出校验：{total} 块 —— {exact} 逐字节一致，"
          f"{partial} 节选跳过，{fail} 不一致")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
