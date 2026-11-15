"""逐点 kernel（第 4 章）：add、silu_mul。

两个 kernel 都只接受**连续**（contiguous）的张量：按一维的下标 0..n-1 去读写。
任意 stride 的写法在第 5 章。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


@triton.jit
def _silu_mul_kernel(gate_ptr, up_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """out = silu(gate) * up，SwiGLU MLP 中间那一步。"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    # 输入可能是 fp16 / bf16：在 fp32 里算 sigmoid，最后再转回输出的类型
    g = tl.load(gate_ptr + offs, mask=mask).to(tl.float32)
    u = tl.load(up_ptr + offs, mask=mask).to(tl.float32)
    out = g * tl.sigmoid(g) * u
    tl.store(out_ptr + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


def _check(a, b):
    assert a.shape == b.shape, f"形状不同：{a.shape} vs {b.shape}"
    assert a.dtype == b.dtype, f"dtype 不同：{a.dtype} vs {b.dtype}"
    assert a.device == b.device, f"设备不同：{a.device} vs {b.device}"
    assert a.is_contiguous() and b.is_contiguous(), "第 4 章的 kernel 只支持连续张量（任意 stride 见第 5 章）"


def add(x: torch.Tensor, y: torch.Tensor, BLOCK: int = 1024) -> torch.Tensor:
    _check(x, y)
    out = torch.empty_like(x)
    n = x.numel()
    if n:                                     # grid 不能是 0
        _add_kernel[(triton.cdiv(n, BLOCK),)](x, y, out, n, BLOCK=BLOCK)
    return out


def silu_mul(gate: torch.Tensor, up: torch.Tensor, BLOCK: int = 1024) -> torch.Tensor:
    _check(gate, up)
    out = torch.empty_like(gate)
    n = gate.numel()
    if n:
        _silu_mul_kernel[(triton.cdiv(n, BLOCK),)](gate, up, out, n, BLOCK=BLOCK)
    return out


@triton.jit
def _silu_mul_bwd_kernel(grad_ptr, gate_ptr, up_ptr, dgate_ptr, dup_ptr, n, BLOCK: tl.constexpr):
    """silu_mul 的反向（第 10 章）。设 s = sigmoid(g)：
         d out / d up   = silu(g) = g · s
         d out / d gate = up · s · (1 + g · (1 − s))
    读 3 个张量、写 2 个，一个 kernel 做完。"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    dy = tl.load(grad_ptr + offs, mask=mask).to(tl.float32)
    g = tl.load(gate_ptr + offs, mask=mask).to(tl.float32)
    u = tl.load(up_ptr + offs, mask=mask).to(tl.float32)
    s = tl.sigmoid(g)
    tl.store(dup_ptr + offs, (dy * g * s).to(dup_ptr.dtype.element_ty), mask=mask)
    tl.store(dgate_ptr + offs, (dy * u * s * (1 + g * (1 - s))).to(dgate_ptr.dtype.element_ty), mask=mask)
