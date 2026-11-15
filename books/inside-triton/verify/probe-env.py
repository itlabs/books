#!/usr/bin/env python3
"""《深入 Triton》批次 0：验证机环境探测。

在 A10 和 Max 1100 两台机器上各跑一次，把完整输出贴回来：

    python3 probe-env.py            # 基本探测 + 三个小 kernel
    python3 probe-env.py --helion   # 额外试一个固定 Config 的 Helion kernel

脚本只读、不安装任何东西。每一项单独 try，某一项失败不影响后面的项；
最后打印一份汇总，失败项附带异常信息。
"""
import importlib
import importlib.metadata as md
import os
import platform
import shutil
import subprocess
import sys
import traceback

RESULTS = []  # (名称, 状态, 详情)


def record(name, ok, detail=""):
    status = {True: "OK", False: "FAIL", None: "SKIP"}[ok]
    RESULTS.append((name, status, detail))
    print(f"[{status:4}] {name}" + (f": {detail}" if detail else ""))


def check(name):
    """装饰器：运行一项检查，捕获异常并记录。"""
    def deco(fn):
        def run(*a, **kw):
            try:
                detail = fn(*a, **kw)
                record(name, True, detail or "")
                return True
            except SkipCheck as e:
                record(name, None, str(e))
            except Exception as e:  # noqa: BLE001 —— 探测脚本，什么错都要记下
                record(name, False, f"{type(e).__name__}: {e}")
                traceback.print_exc(limit=3)
            return False
        return run
    return deco


class SkipCheck(Exception):
    pass


def section(title):
    print(f"\n===== {title} =====")


def run_cmd(cmd):
    exe = shutil.which(cmd[0])
    if not exe:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (out.stdout + out.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return f"<{cmd[0]} 运行失败: {e}>"


# ---------------------------------------------------------------- 系统 ----
section("系统")
print("platform :", platform.platform())
print("python   :", sys.version.replace("\n", " "))
print("executable:", sys.executable)
if sys.prefix == sys.base_prefix:
    print("!! 警告：当前 python 不在任何 venv 里。如果你以为已经激活了 venv，"
          "多半是 PATH、alias 或 shell 函数把 python 指到了别处；"
          "可以直接用 <venv>/bin/python probe-env.py 来跑。")
for var in ("CUDA_VISIBLE_DEVICES", "ZE_AFFINITY_MASK", "ONEAPI_ROOT",
            "LD_LIBRARY_PATH", "LD_PRELOAD", "CC", "CXX", "CPATH",
            "TRITON_HOME", "TRITON_CACHE_DIR"):
    if os.environ.get(var):
        print(f"{var}={os.environ[var]}")
# Triton 的驱动在首次运行时会现场编一段 C/C++ 胶水代码；它挑哪个编译器，取决于 PATH 里有什么
for tool in ("icpx", "dpclang++", "clang++", "g++", "gcc"):
    if shutil.which(tool):
        print(f"which {tool}: {shutil.which(tool)}")

section("驱动与设备工具")
for cmd in (["nvidia-smi", "--query-gpu=name,driver_version,compute_cap,memory.total",
             "--format=csv"],
            ["nvidia-smi"],
            ["xpu-smi", "discovery"],
            ["sycl-ls"],
            ["clinfo", "-l"]):
    out = run_cmd(cmd)
    if out is not None:
        print(f"$ {' '.join(cmd)}\n{out}\n")
        if cmd[0] == "nvidia-smi":
            # 驱动支持的最高 CUDA 版本只在完整输出的表头里
            full = run_cmd(["nvidia-smi"]) or ""
            cuda_line = [ln for ln in full.splitlines() if "CUDA Version" in ln]
            if cuda_line:
                print("驱动表头:", cuda_line[0].strip(" |"), "\n")
            break  # 不必再跑完整版
if shutil.which("nvcc"):
    print("$ nvcc --version\n" + (run_cmd(["nvcc", "--version"]) or ""))

section("已安装的相关包")
for dist in ("torch", "triton", "pytorch-triton", "pytorch-triton-xpu",
             "triton-xpu", "intel-xpu-backend-for-triton", "helion",
             "nvidia-cutlass-dsl", "tilelang", "numpy"):
    try:
        print(f"{dist:32} {md.version(dist)}")
    except md.PackageNotFoundError:
        pass

# --------------------------------------------------------------- torch ----
section("PyTorch")
torch = None
DEVICE = None


@check("import torch")
def _torch():
    global torch
    torch = importlib.import_module("torch")
    info = [f"torch {torch.__version__}"]
    if getattr(torch.version, "cuda", None):
        info.append(f"cuda {torch.version.cuda}")
    if getattr(torch.version, "xpu", None):
        info.append(f"xpu {torch.version.xpu}")
    return ", ".join(info)


_torch()


@check("torch 设备")
def _torch_dev():
    if torch is None:
        raise SkipCheck("torch 不可用")
    lines = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            lines.append(f"cuda:{i} {p.name} cc={p.major}.{p.minor} "
                         f"mem={p.total_memory / 2**30:.1f}GiB SMs={p.multi_processor_count}")
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        for i in range(xpu.device_count()):
            p = xpu.get_device_properties(i)
            extra = {k: getattr(p, k) for k in ("driver_version", "gpu_eu_count",
                                                 "max_compute_units", "sub_group_sizes",
                                                 "has_fp64", "total_memory")
                     if hasattr(p, k)}
            lines.append(f"xpu:{i} {p.name} {extra}")
    if not lines:
        raise RuntimeError("torch 既没有看到 CUDA 设备，也没有看到 XPU 设备")
    return " | ".join(lines)


_torch_dev()

# -------------------------------------------------------------- triton ----
section("Triton")
triton = tl = None


@check("import triton")
def _triton():
    global triton, tl
    triton = importlib.import_module("triton")
    tl = importlib.import_module("triton.language")
    return f"triton {triton.__version__} @ {os.path.dirname(triton.__file__)}"


_triton()


@check("active driver / target / DEVICE")
def _target():
    global DEVICE
    if triton is None:
        raise SkipCheck("triton 不可用")
    drv = triton.runtime.driver.active
    target = drv.get_current_target()
    note = ""
    try:
        DEVICE = drv.get_active_torch_device()
    except AttributeError:
        # 老版本 Triton（如 3.1）没有这个 API：按后端名退回到 torch 设备
        DEVICE = torch.device("xpu" if target.backend == "xpu" else "cuda")
        note = "（老版本 Triton 无 get_active_torch_device，已按 backend 退回）"
    return f"driver={type(drv).__name__} target={target} DEVICE={DEVICE}{note}"


_target()


def need_device():
    if triton is None or torch is None or DEVICE is None:
        raise SkipCheck("triton / torch / DEVICE 不可用")


ADD_KERNEL = None


@check("vector add（逐点 + mask）")
def _vadd():
    global ADD_KERNEL
    need_device()

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    n = 98_432  # 故意不是 BLOCK 的整数倍，测 mask
    x = torch.rand(n, device=DEVICE)
    y = torch.rand(n, device=DEVICE)
    out = torch.empty_like(x)
    ADD_KERNEL = add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
    err = (out - (x + y)).abs().max().item()
    if err != 0.0:
        raise AssertionError(f"max abs err = {err}")
    return f"n={n}, max abs err = 0"


_vadd()


@check("编译产物（asm 各级 IR）")
def _asm():
    if ADD_KERNEL is None or not hasattr(ADD_KERNEL, "asm"):
        raise SkipCheck("拿不到 CompiledKernel.asm")
    keys = list(ADD_KERNEL.asm.keys())
    ttgir = ADD_KERNEL.asm.get("ttgir", "")
    # 只要 layout 别名定义（如 #blocked = #ttg.blocked<...>），不要 #loc
    layout_lines = [ln.strip() for ln in str(ttgir).splitlines()
                    if ln.lstrip().startswith("#") and "= #ttg." in ln][:3]
    print("  asm keys:", keys)
    for ln in layout_lines:
        print("  ttgir layout:", ln[:160])
    return f"keys={keys}"


_asm()


@check("tl.dot（fp16 输入，fp32 累加，64x64x64）")
def _dot():
    need_device()

    @triton.jit
    def dot_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        rk = tl.arange(0, K)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
        c = tl.dot(a, b)  # 累加器默认 fp32
        tl.store(c_ptr + rm[:, None] * N + rn[None, :], c)

    M = N = K = 64
    a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
    b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
    c = torch.empty(M, N, device=DEVICE, dtype=torch.float32)
    k = dot_kernel[(1,)](a, b, c, M, N, K)
    ref = a.float() @ b.float()
    err = (c - ref).abs().max().item()
    if err > 1e-2:
        raise AssertionError(f"max abs err = {err}")
    ttgir = str(getattr(k, "asm", {}).get("ttgir", ""))
    import re
    mma = sorted(set(re.findall(r"#[\w.]*(?:mma|dpas)[\w.]*", ttgir)))
    return f"max abs err = {err:.2e}; ttgir 里的矩阵 layout: {mma or '未找到'}"


_dot()


@check("triton.testing.do_bench")
def _bench():
    need_device()
    from triton.testing import do_bench
    x = torch.rand(1 << 24, device=DEVICE)
    ms = do_bench(lambda: x * 2.0)
    gbps = 2 * x.numel() * x.element_size() / (ms * 1e-3) / 1e9
    return f"x*2 on 64 MiB: {ms:.3f} ms ≈ {gbps:.0f} GB/s（粗略带宽）"


_bench()


@check("TRITON_INTERPRET 可用（CPU 解释器）")
def _interp():
    if triton is None:
        raise SkipCheck("triton 不可用")
    code = (
        "import torch, triton, triton.language as tl\n"
        "@triton.jit\n"
        "def k(x, BLOCK: tl.constexpr):\n"
        "    o = tl.arange(0, BLOCK)\n"
        "    tl.store(x + o, tl.load(x + o) * 2)\n"
        "x = torch.arange(8, dtype=torch.float32)\n"
        "k[(1,)](x, BLOCK=8)\n"
        "print(x.tolist())\n"
    )
    env = dict(os.environ, TRITON_INTERPRET="1")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=120, env=env)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip().splitlines()[-1] if out.stderr else "非零退出")
    return out.stdout.strip()


_interp()

@check("torch.compile（Inductor 生成 Triton kernel）")
def _compile():
    need_device()

    def f(x):
        return torch.softmax(x * 2.0, dim=-1) + 1.0

    x = torch.randn(256, 1024, device=DEVICE)
    cf = torch.compile(f)
    err = (cf(x) - f(x)).abs().max().item()
    if err > 1e-5:
        raise AssertionError(f"max abs err = {err}")
    return f"max abs err = {err:.1e}（torch 与 triton 版本不配套时，这一项最容易先坏）"


_compile()

# --------------------------------------------------------- 生态组件 ----
section("Gluon / proton / Helion")


@check("import triton.experimental.gluon")
def _gluon():
    if triton is None:
        raise SkipCheck("triton 不可用")
    importlib.import_module("triton.experimental.gluon")
    lang = importlib.import_module("triton.experimental.gluon.language")
    base = os.path.dirname(lang.__file__)
    archs = []
    for vendor in ("nvidia", "amd", "intel"):
        d = os.path.join(base, vendor)
        if os.path.isdir(d):
            subs = sorted(s for s in os.listdir(d)
                          if not s.startswith("_") and s != "__pycache__")
            names = [s.removesuffix('.py') for s in subs]
            if not names:
                # 没有架构子包（如 intel 只有 _layouts.py）：改报模块导出的名字
                try:
                    mod = importlib.import_module(
                        f"triton.experimental.gluon.language.{vendor}")
                    names = ["exports:"] + list(getattr(mod, "__all__", []))
                except Exception as e:  # noqa: BLE001
                    names = [f"<import 失败: {type(e).__name__}: {e}>"]
            archs.append(f"{vendor}: {names}")
    return "; ".join(archs) or "没有厂商子包"


_gluon()


@check("import triton.profiler (proton)")
def _proton():
    if triton is None:
        raise SkipCheck("triton 不可用")
    importlib.import_module("triton.profiler")
    return "ok"


_proton()


@check("import helion")
def _helion_import():
    h = importlib.import_module("helion")
    ver = getattr(h, "__version__", None) or md.version("helion")
    return f"helion {ver}"


_helion_ok = _helion_import()


@check("Helion 固定 Config 的 add kernel（--helion）")
def _helion_run():
    if "--helion" not in sys.argv:
        raise SkipCheck("未加 --helion")
    if not _helion_ok:
        raise SkipCheck("helion 不可用")
    need_device()
    # hl 必须是模块级全局名：在函数里局部 import 的话，kernel 引用它会被
    # Helion 当成闭包，报 ClosuresNotSupported
    global helion, hl
    import helion
    import helion.language as hl

    @helion.kernel(config=helion.Config(block_sizes=[1024]))
    def hadd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for tile in hl.tile(x.size()):
            out[tile] = x[tile] + y[tile]
        return out

    x = torch.rand(98_432, device=DEVICE)
    y = torch.rand(98_432, device=DEVICE)
    err = (hadd(x, y) - (x + y)).abs().max().item()
    if err != 0.0:
        raise AssertionError(f"max abs err = {err}")
    return "max abs err = 0"


_helion_run()

# ----------------------------------------------------------------- 汇总 ----
section("汇总")
for name, status, detail in RESULTS:
    print(f"{status:4} | {name}" + (f" | {detail}" if status != "OK" and detail else ""))
fails = sum(1 for _, s, _ in RESULTS if s == "FAIL")
print(f"\n共 {len(RESULTS)} 项，失败 {fails} 项。请把从头到尾的完整输出贴回来。")
sys.exit(1 if fails else 0)
