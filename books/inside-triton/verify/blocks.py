"""从章节 md 里抽代码块，解析块首注释。verify-triton.py 和 verify-outputs.py 共用。

块首注释的约定与 build.js 的 tritonHeader() 完全一致（只看开头连续的注释行）：
    # arch: sm80 xpu          这个块要在哪些平台上跑，满足其一即可
    # CE: triton_amd_370 ...  triton-ir 块用哪个 CE 编译器 / wrapper 参数
    # IR: ttgir               triton-ir 块的 Device Viewer 停在哪一级
两边改一处就要改另一处。
"""
import os
import re

# 要校验的围栏；-norun 变体是片段 / 伪代码 / 本书没验证过的写法，不抽
RUNNABLE = ("triton", "gluon", "helion")
LANGS = RUNNABLE + ("triton-ir",)
ARCH_TAGS = ("any", "sm80", "sm86", "sm90", "sm100", "cdna3", "xpu")

FENCE_RE = re.compile(r"^```([\w-]+)[ \t]*\n(.*?)^```[ \t]*$", re.M | re.S)
HEADER_RE = re.compile(r"^\s*#\s*(arch|CE|IR)\s*:\s*(.*?)\s*$")


class Block:
    def __init__(self, chapter, index, line, lang, code):
        self.chapter, self.index, self.line, self.lang, self.code = chapter, index, line, lang, code
        self.header = header(code)
        self.arch = self.header.get("arch", "").replace(",", " ").split()

    @property
    def name(self):
        return f"{self.chapter}#{self.index}"

    def where(self):
        return f"{self.chapter}.md:{self.line}"


def header(code):
    h = {}
    for line in code.splitlines():
        m = HEADER_RE.match(line)
        if m:
            h[m.group(1)] = m.group(2)
        elif line.strip() and not line.lstrip().startswith("#"):
            break
    return h


def chapter_files(book_dir, only=()):
    d = os.path.join(book_dir, "chapters")
    if not os.path.isdir(d):
        return []
    names = sorted(f[:-3] for f in os.listdir(d) if f.endswith(".md"))
    if only:
        missing = [c for c in only if c not in names]
        if missing:
            raise SystemExit(f"没有这些章：{', '.join(missing)}（chapters/ 下现有：{', '.join(names) or '无'}）")
        names = [n for n in names if n in only]
    return [(n, os.path.join(d, n + ".md")) for n in names]


def extract(book_dir, only=(), langs=LANGS):
    """按章、按出现顺序返回 Block 列表。index 是该章内同类可校验块的序号（从 1 起）。"""
    out = []
    for ch, path in chapter_files(book_dir, only):
        text = open(path, encoding="utf-8").read()
        n = 0
        for m in FENCE_RE.finditer(text):
            lang = m.group(1)
            if lang not in langs:
                continue
            n += 1
            line = text.count("\n", 0, m.start()) + 1
            out.append(Block(ch, n, line, lang, m.group(2)))
    return out
