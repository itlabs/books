"""RMSNorm（第 6 章）：y = x / sqrt(mean(x²) + eps) * weight，按最后一维归一化。

一行（hidden 维）放进一个 program：读一遍 x、读一遍 weight、写一遍 y。
Transformer 的 hidden 维通常是 2048–16384，都放得下（第 6 章的溢出实验）。
"""
import torch
import triton
import triton.language as tl

from .softmax import MAX_FUSED, num_warps_for


@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, y_ptr, n_cols, stride_x, stride_y, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    # 在 fp32 里求平方和：bf16 的平方和很容易丢精度
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / n_cols     # other=0：被 mask 掉的位置不影响平方和
    rstd = tl.rsqrt(mean_sq + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * stride_y + cols, x * rstd * w, mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    assert x.shape[-1] == weight.shape[0] and weight.dim() == 1
    x2 = x.reshape(-1, x.shape[-1])                  # 前面的维度都当成「行」
    assert x2.stride(1) == 1, "最后一维必须连续"
    n_rows, n_cols = x2.shape
    block = triton.next_power_of_2(n_cols)
    assert block <= MAX_FUSED, f"hidden 维 {n_cols} 太大，一个 program 放不下（上限 {MAX_FUSED}）"
    y = torch.empty_like(x2)
    _rmsnorm_kernel[(n_rows,)](x2, weight, y, n_cols, x2.stride(0), y.stride(0), eps,
                               BLOCK=block, num_warps=num_warps_for(block))
    return y.view(x.shape)
