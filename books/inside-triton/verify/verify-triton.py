#!/usr/bin/env python3
"""校验《深入 Triton》各章的完整代码块真的能跑 / 能编译。用法见同目录的 README.md。

  ```triton / ```gluon / ```helion
      按块首 `# arch:` 判断这台机器能不能跑：能跑就在单独的进程里真跑一遍，退出码 0 才算通过；
      不能跑记为「跳过」——**跳过不算通过**，要到对应的机器上再跑一次。
  ```triton-ir
      默认：用本机装的 Triton（基线 3.9.0）离线编译（triton_ir_local.py，不需要对应的 GPU），
            并检查块首 `# IR:` 指定的那一级确实编出来了。
      CE=1：改走 Compiler Explorer 的 API（CE 上是 3.7.0），检查的是读者点按钮时看到的东西。
            会把代码块上传到 godbolt.org。

退出码：有失败 1；全部通过 0（有跳过时也是 0，但会醒目地报出来）。
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BOOK = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import blocks  # noqa: E402

CE = os.environ.get("CE", "0") == "1"
VERBOSE = os.environ.get("VERBOSE", "0") == "1"
TIMEOUT = int(os.environ.get("TIMEOUT", "900"))


# ---------------------------------------------------------------- 平台 ----
def detect_platform():
    """这台机器满足哪些 # arch: 标签。sm 标签是「该架构及以上」。"""
    tags, desc = set(), []
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        return tags, f"（import torch 失败：{e}）"
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        if getattr(torch.version, "hip", None):
            arch = getattr(p, "gcnArchName", "")
            desc.append(f"AMD {p.name} {arch}")
            if arch.startswith("gfx94"):
                tags.add("cdna3")
        else:
            cc = torch.cuda.get_device_capability(0)
            desc.append(f"NVIDIA {p.name} sm_{cc[0]}{cc[1]}")
            for tag, need in (("sm80", (8, 0)), ("sm86", (8, 6)), ("sm90", (9, 0)), ("sm100", (10, 0))):
                if cc >= need:
                    tags.add(tag)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        desc.append(f"Intel {torch.xpu.get_device_name(0)}")
        tags.add("xpu")
    if tags or desc:
        tags.add("any")
    return tags, "、".join(desc) or "没有可用的 GPU"


# ---------------------------------------------------------------- 运行 ----
def run_python(code, args=()):
    with tempfile.TemporaryDirectory(prefix="verify-triton-") as d:
        path = os.path.join(d, "main.py")
        open(path, "w", encoding="utf-8").write(code)
        try:
            p = subprocess.run([sys.executable, path, *args], cwd=d, capture_output=True, text=True, timeout=TIMEOUT)
            return p.returncode, p.stdout + p.stderr
        except subprocess.TimeoutExpired:
            return -1, f"超时（{TIMEOUT} 秒）"


def check_runnable(b, tags):
    if not b.arch:
        return "FAIL", "缺少块首 # arch: 行（build.js 也会因此报错）"
    bad = [a for a in b.arch if a not in blocks.ARCH_TAGS]
    if bad:
        return "FAIL", f"未知的 # arch: {' '.join(bad)}"
    if not tags.intersection(b.arch):
        return "SKIP", f"需要 {' 或 '.join(b.arch)}"
    rc, out = run_python(b.code)
    return ("PASS" if rc == 0 else "FAIL"), out


def stage_list(gluon, amd):
    # 与 build.js 的 tritonIrUrl() 一致
    return ["source", *([] if gluon else ["ttir"]), "ttgir", "llir", *(["amdgcn"] if amd else ["ptx", "sass"])]


def check_ir_local(b):
    want = b.header.get("IR", "ttgir")
    with tempfile.TemporaryDirectory(prefix="verify-triton-ir-") as d:
        path = os.path.join(d, "block.py")
        open(path, "w", encoding="utf-8").write(b.code)
        p = subprocess.run([sys.executable, os.path.join(HERE, "triton_ir_local.py"), path, "--json"],
                           cwd=d, capture_output=True, text=True, timeout=TIMEOUT)
    if p.returncode != 0:
        return "FAIL", p.stdout + p.stderr
    info = json.loads(p.stdout.strip().splitlines()[-1])
    if want == "sass":  # 本地这条路不跑反汇编；sass 只能由 CE=1 校验
        return "PASS", f"{info['target']}：{info['kernels']}（sass 只在 CE=1 下检查）"
    missing = [k for k, st in info["kernels"].items() if want not in st]
    if len(missing) == len(info["kernels"]):
        return "FAIL", f"# IR: {want} 没编出来：{info['kernels']}"
    return "PASS", f"{info['target']}：{info['kernels']}"


def ce_compile(code, compiler, options):
    body = json.dumps({"source": code, "lang": "triton",
                       "options": {"userArguments": options, "filters": {}}}).encode()
    req = urllib.request.Request(f"https://godbolt.org/api/compiler/{compiler}/compile", data=body,
                                 headers={"Content-Type": "application/json", "Accept": "application/json"})
    for attempt in range(3):  # CE 偶尔超时被 SIGKILL（code 143），重试就好
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.load(r)
        if d.get("code") != 143:
            return d
        time.sleep(5)
    return d


def check_ir_ce(b):
    ce = (b.header.get("CE") or "triton_nvidia_370").split()
    compiler, args = ce[0], ce[1:]
    amd = "amd" in compiler
    options = " ".join(args) if args else ("" if amd else "--arch 86")  # 与 build.js 默认一致
    want = b.header.get("IR", "ttgir")
    gluon = "@gluon.jit" in b.code
    stages = stage_list(gluon, amd)
    if want not in stages:
        return "FAIL", f"# IR: {want} 在 {compiler} 上没有这一级；可选 {' / '.join(stages)}"
    d = ce_compile(b.code, compiler, options)
    err = "\n".join(l["text"] for l in d.get("stderr", []) if "nvdisasm warning" not in l["text"])
    if d.get("code") != 0:
        return "FAIL", f"CE code {d.get('code')}\n{err}"
    devices = d.get("devices") or {}
    key = f"[stage {stages.index(want) + 1}].{want}"
    hits = [k for k in devices if k.endswith(key)]
    if not hits:
        return "FAIL", f"CE 结果里没有 {key}（按钮打开会显示 Device not found）；有的是：{list(devices)}"
    return "PASS", f"{compiler} {options}：{hits}"


# ---------------------------------------------------------------- 主流程 ----
def ce_reachable():
    """CE=1 时先确认连得上 godbolt，免得每个块都干等到超时。"""
    try:
        req = urllib.request.Request("https://godbolt.org/api/compilers/triton", headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return [c["id"] for c in json.load(r)], None
    except Exception as e:  # noqa: BLE001
        return None, e


def main():
    sys.stdout.reconfigure(line_buffering=True)  # 接管道时也逐行输出，看得到进度
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    bs = blocks.extract(BOOK, only)
    tags, desc = detect_platform()
    import triton
    print(f"平台：{desc}；满足的 # arch: 标签：{' '.join(sorted(tags)) or '无'}")
    print(f"Triton {triton.__version__}（{sys.executable}）；triton-ir 块走{'Compiler Explorer（3.7.0）' if CE else '本机离线编译'}")
    if not bs:
        print("没有可校验的代码块。")
        return 0
    if CE and any(b.lang == "triton-ir" for b in bs):
        ids, err = ce_reachable()
        if ids is None:
            print(f"✗ CE=1 但连不上 godbolt.org（{type(err).__name__}: {err}）。"
                  "需要走代理的话先设 https_proxy；或者去掉 CE=1，用本机离线编译。")
            return 127
        print(f"CE 上的 Triton 编译器：{' '.join(i for i in ids if i.endswith(('_370', '_380', '_390')))} 等 {len(ids)} 个")
    res = []
    for b in bs:
        t0 = time.time()
        try:
            st, detail = check_runnable(b, tags) if b.lang in blocks.RUNNABLE else (check_ir_ce(b) if CE else check_ir_local(b))
        except Exception as e:  # noqa: BLE001
            st, detail = "FAIL", f"{type(e).__name__}: {e}"
        res.append((b, st))
        print(f"[{st}] {b.name:10} {b.lang:9} {b.where():18} {time.time() - t0:5.1f}s"
              + (f"  {detail}" if st == "SKIP" else ""))
        if st == "FAIL" or VERBOSE:
            print("       " + (detail or "").strip().replace("\n", "\n       ")[-4000:])
    n = {s: sum(1 for _, x in res if x == s) for s in ("PASS", "FAIL", "SKIP")}
    print(f"\n共 {len(res)} 块：通过 {n['PASS']}，失败 {n['FAIL']}，跳过 {n['SKIP']}。")
    if n["SKIP"]:
        print(f"⚠ 跳过的 {n['SKIP']} 块不算通过，要到满足 # arch: 的机器上再跑：")
        for b, s in res:
            if s == "SKIP":
                print(f"    {b.name}（{b.where()}，# arch: {' '.join(b.arch)}）")
    return 1 if n["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
