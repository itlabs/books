"""把 minillm 的 kernel 注册成 PyTorch 算子（第 10 章）。

用的是 torch.library.triton_op：它注册的是一个真正的算子（有 schema、能注册反向、能 torch.export），
同时又让 torch.compile 看得见里面的 Triton kernel。函数体里只能调用 PyTorch 的算子和
用 wrap_triton(...) 包起来的 Triton kernel。

    import minillm.ops as ops
    y = ops.rmsnorm(x, w, 1e-6)          # 也可以写 torch.ops.minillm.rmsnorm(x, w, 1e-6)

反向：silu_mul 用 Triton kernel；add、mm 直接复用前向的算子；rmsnorm、softmax 先用 PyTorch 写
（写成 Triton 是留给读者的练习）。
"""
import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from .device import DEVICE
from .kernels import matmul as _mm
from .kernels import norm as _norm
from .kernels import pointwise as _pw
from .kernels import softmax as _sm

BLOCK = 1024


def _grid1d(n):
    return (triton.cdiv(n, BLOCK),)


# ---------------------------------------------------------------- add ----
@triton_op("minillm::add", mutates_args=())
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = x.contiguous(), y.contiguous()
    out = torch.empty_like(x)
    if x.numel():
        wrap_triton(_pw._add_kernel)[_grid1d(x.numel())](x, y, out, x.numel(), BLOCK=BLOCK)
    return out


def _add_backward(ctx, grad):
    return grad, grad


add.register_autograd(_add_backward)


# ----------------------------------------------------------- silu_mul ----
@triton_op("minillm::silu_mul", mutates_args=())
def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    gate, up = gate.contiguous(), up.contiguous()
    out = torch.empty_like(gate)
    if gate.numel():
        wrap_triton(_pw._silu_mul_kernel)[_grid1d(gate.numel())](gate, up, out, gate.numel(), BLOCK=BLOCK)
    return out


@triton_op("minillm::silu_mul_backward", mutates_args=())
def silu_mul_backward(grad: torch.Tensor, gate: torch.Tensor, up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    grad, gate, up = grad.contiguous(), gate.contiguous(), up.contiguous()
    dgate, dup = torch.empty_like(gate), torch.empty_like(up)
    if gate.numel():
        wrap_triton(_pw._silu_mul_bwd_kernel)[_grid1d(gate.numel())](grad, gate, up, dgate, dup, gate.numel(),
                                                                    BLOCK=BLOCK)
    return dgate, dup


def _silu_mul_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs)                 # 反向要用 gate 和 up，不保存输出


def _silu_mul_bwd(ctx, grad):
    gate, up = ctx.saved_tensors
    return silu_mul_backward(grad, gate, up)


silu_mul.register_autograd(_silu_mul_bwd, setup_context=_silu_mul_setup)


# ------------------------------------------------------------ rmsnorm ----
@triton_op("minillm::rmsnorm", mutates_args=())
def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    n_rows, n_cols = x2.shape
    block = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x2)
    wrap_triton(_norm._rmsnorm_kernel)[(n_rows,)](x2, weight, y, n_cols, x2.stride(0), y.stride(0), eps,
                                                  BLOCK=block, num_warps=_sm.num_warps_for(block))
    return y.view(x.shape)


def _rmsnorm_setup(ctx, inputs, output):
    x, weight, eps = inputs
    ctx.save_for_backward(x, weight)
    ctx.eps = eps


def _rmsnorm_bwd(ctx, grad):
    # y = x · r · w，r = 1 / sqrt(mean(x²) + eps)。在 fp32 里算：
    #   dx = r · (g·w) − x · r³ · mean(g·w·x)，dw = Σ_行 g · x · r
    x, weight = ctx.saved_tensors
    xf, gf, wf = x.float(), grad.float(), weight.float()
    r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + ctx.eps)
    gw = gf * wf
    dx = r * gw - xf * r.pow(3) * (gw * xf).mean(-1, keepdim=True)
    dw = (gf * xf * r).reshape(-1, x.shape[-1]).sum(0)
    return dx.to(x.dtype), dw.to(weight.dtype), None


rmsnorm.register_autograd(_rmsnorm_bwd, setup_context=_rmsnorm_setup)


# ------------------------------------------------------------ softmax ----
@triton_op("minillm::softmax", mutates_args=())
def softmax(x: torch.Tensor) -> torch.Tensor:
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    n_rows, n_cols = x2.shape
    y = torch.empty_like(x2)
    block = triton.next_power_of_2(n_cols)
    if block <= _sm.MAX_FUSED:
        wrap_triton(_sm._softmax_fused_kernel)[(n_rows,)](x2, y, n_cols, x2.stride(0), y.stride(0),
                                                          BLOCK=block, num_warps=_sm.num_warps_for(block))
    else:
        wrap_triton(_sm._softmax_online_kernel)[(n_rows,)](x2, y, n_cols, x2.stride(0), y.stride(0),
                                                           BLOCK=4096, num_warps=_sm.num_warps_for(4096))
    return y.view(x.shape)


def _softmax_setup(ctx, inputs, output):
    ctx.save_for_backward(output)                  # softmax 的反向只要输出 y


def _softmax_bwd(ctx, grad):
    (y,) = ctx.saved_tensors
    yf, gf = y.float(), grad.float()
    return (yf * (gf - (gf * yf).sum(-1, keepdim=True))).to(y.dtype)


softmax.register_autograd(_softmax_bwd, setup_context=_softmax_setup)


# ----------------------------------------------------------------- mm ----
def _desc_ok(t):
    # 和 kernels/matmul.py 的 descriptor_ok 一样，但不能调 data_ptr()：torch.compile 追踪时传进来的是
    # FakeTensor，取不到地址。PyTorch 分配的存储本身是对齐的，所以改查 storage_offset。
    size = t.element_size()
    return (t.stride(-1) == 1 and t.storage_offset() * size % 16 == 0 and
            all(s * size % 16 == 0 for s in t.stride()[:-1]))


def _use_descriptor(a, b):
    # 只在 Intel GPU 上走 descriptor（第 8 章）；descriptor 在 kernel 里创建，原因见 _matmul_desc_dev_kernel
    return DEVICE.type == "xpu" and b.shape[1] * a.element_size() % 16 == 0 and _desc_ok(a) and _desc_ok(b)


@triton_op("minillm::mm", mutates_args=())
def mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)
    if M == 0 or N == 0:
        return c
    if _use_descriptor(a, b):
        grid = (triton.cdiv(M, 256) * triton.cdiv(N, 256),)     # 第 12 章：一维 grid + L2 swizzle
        wrap_triton(_mm._matmul_grouped_kernel)[grid](a, b, c, M, N, K, a.stride(0), b.stride(0), c.stride(0),
                                                      BLOCK_M=256, BLOCK_N=256, BLOCK_K=32, GROUP_M=8, num_warps=32)
    else:
        grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))
        wrap_triton(_mm._matmul_kernel)[grid](a, b, c, M, N, K, *a.stride(), *b.stride(), *c.stride(),
                                              BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, ACC_DTYPE=tl.float32,
                                              num_warps=8)
    return c


def _mm_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _mm_bwd(ctx, grad):
    a, b = ctx.saved_tensors
    # dA = dC · Bᵀ，dB = Aᵀ · dC。转置视图的最后一维不连续，mm 会自动退回指针版
    return mm(grad, b.t()), mm(a.t(), grad)


mm.register_autograd(_mm_bwd, setup_context=_mm_setup)
