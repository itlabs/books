#!/usr/bin/env python3
"""不用 GPU，在本机用当前安装的 Triton 编译一个 triton-ir 块，dump 各级 IR。

用途：
  1. verify-triton.py 用它校验 ```triton-ir 块在本书基线（Triton 3.9.0）上也能编译——
     CE 上只有 3.7.0；
  2. 写正文时给 ```ttir-norun / ```ttgir-norun 块取**真 dump 出来的** 3.9.0 IR（BRIEF §13）。

用法：
    python triton_ir_local.py block.py                         # 读块首 # CE: 行；默认 NVIDIA sm_86
    python triton_ir_local.py block.py --backend hip --arch gfx942
    python triton_ir_local.py block.py --out ir/ --show ttgir  # IR 写进 ir/，并把 TTGIR 打到 stdout

原理和 Compiler Explorer 的 Triton 编译器一样：把脚本当模块执行一遍，里面的张量换成
PyTorch 的 FakeTensor，GPU driver 换成一个只会报告目标架构的 mock，kernel 只编译、不启动。
所以 triton-ir 块里写 device="cuda" 就行，本机有没有 NVIDIA / AMD GPU 都无所谓——
只要装的 Triton 里带了对应的后端（PyPI 的 triton、triton-xpu 都带 nvidia 和 amd）。

注意：mock 掉 driver 以后，同一个进程里就不能再真跑 kernel 了；所以它总在单独的进程里用。
"""
import argparse
import glob
import json
import os
import re
import runpy
import shutil
import sys
import tempfile
from unittest import mock

# 与 build.js 的 tritonIrUrl() 同一套约定：块首 `# CE: <编译器 id> [wrapper 参数]`
CE_DEFAULT = {"triton_nvidia": ("cuda", "86"), "triton_amd": ("hip", "gfx942")}
# 各级 IR 在缓存目录里的扩展名，按管线顺序
STAGES = ["source", "ttir", "ttgir", "llir", "ptx", "amdgcn", "spv"]


def header_target(code):
    """从块首 # CE: 行推出 (backend, arch)；没有就是 NVIDIA sm_86（与 build.js 默认一致）。"""
    backend, arch = CE_DEFAULT["triton_nvidia"]
    for line in code.splitlines():
        m = re.match(r"\s*#\s*CE\s*:\s*(\S+)(.*)$", line)
        if m:
            comp, args = m.group(1), m.group(2).split()
            backend, arch = CE_DEFAULT["triton_amd" if "amd" in comp else "triton_nvidia"]
            if "--arch" in args:
                arch = args[args.index("--arch") + 1]
            break
        if line.strip() and not line.lstrip().startswith("#"):
            break
    return backend, arch


def compile_block(path, backend, arch, out_dir):
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    import triton
    import triton.compiler.compiler as tcc
    from triton.backends.compiler import GPUTarget

    target = GPUTarget(backend, int(arch) if backend == "cuda" else arch, 64 if backend == "hip" else 32)
    os.environ["TRITON_CACHE_DIR"] = out_dir      # 各级 IR 就落在缓存目录里，编完去捡
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"     # 不复用旧缓存，保证这次真的编了

    fake_driver = mock.MagicMock()
    fake_driver.get_current_target = lambda: target
    fake_driver.get_benchmarker = lambda: (lambda *a, **k: [0.0])
    triton.runtime.driver.set_active(fake_driver)
    # 编译完以后 JIT 会构造 CompiledKernel 去加载二进制、启动；换成 mock，到编译为止
    tcc.CompiledKernel = mock.MagicMock()

    with FakeTensorMode():
        # FakeTensor 的 data_ptr() 默认会报错；给 0，顺便让 16 字节对齐检查通过
        torch._subclasses.FakeTensor.data_ptr = lambda self: 0
        runpy.run_path(path, run_name="__main__")


def collect(out_dir):
    """把缓存目录里的各级 IR 按 kernel 归好：{kernel: {stage: 文本}}。"""
    result = {}
    for meta in glob.glob(os.path.join(out_dir, "*", "*.json")):
        if os.path.basename(meta).startswith("__grp__"):
            continue
        name = json.load(open(meta)).get("name") or os.path.splitext(os.path.basename(meta))[0]
        d = os.path.dirname(meta)
        stages = {"json": open(meta, encoding="utf-8").read()}   # 编译元数据：shared memory 用量、num_warps 等
        for st in STAGES:
            files = glob.glob(os.path.join(d, f"*.{st}"))
            if files:
                stages[st] = open(files[0], encoding="utf-8", errors="replace").read()
        result[name] = stages
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("file")
    ap.add_argument("--backend", choices=["cuda", "hip"])
    ap.add_argument("--arch")
    ap.add_argument("--out", help="把各级 IR 写成 <kernel>.<stage> 放进这个目录")
    ap.add_argument("--show", help="把某一级 IR 打到 stdout，例如 ttgir；json 是编译元数据")
    ap.add_argument("--json", action="store_true", help="以 JSON 打印 {kernel: [有哪些 stage]}（给 verify 脚本用）")
    a = ap.parse_args()

    code = open(a.file, encoding="utf-8").read()
    backend, arch = header_target(code)
    backend, arch = a.backend or backend, a.arch or arch

    tmp = tempfile.mkdtemp(prefix="triton-ir-")
    compile_block(os.path.abspath(a.file), backend, arch, tmp)
    kernels = collect(tmp)
    if not kernels:
        shutil.rmtree(tmp, ignore_errors=True)
        sys.exit("没有编出任何 kernel：triton-ir 块必须真的启动一次 kernel（kernel[grid](...)），只定义不启动不会编译。")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        for k, stages in kernels.items():
            for st, text in stages.items():
                open(os.path.join(a.out, f"{k}.{st}"), "w", encoding="utf-8").write(text)
    if a.json:
        print(json.dumps({"target": f"{backend}:{arch}", "kernels": {k: list(v) for k, v in kernels.items()}}))
    if a.show:
        for k, stages in kernels.items():
            if a.show not in stages:
                sys.exit(f"{k} 没有 {a.show} 这一级；有的是：{', '.join(stages)}")
            print(stages[a.show], end="")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
