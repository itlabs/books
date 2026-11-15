"""矩阵乘 C = A @ B（第 7 章：朴素的 tile 版本）。

每个 program 算 C 的一个 BLOCK_M × BLOCK_N 的 tile：沿 K 方向循环，每次读 A 的一条 BLOCK_M × BLOCK_K、
B 的一条 BLOCK_K × BLOCK_N，用 tl.dot 乘起来，累加进一个 fp32 的累加器。
第 8 章换成 tensor descriptor，第 12–15 章逐步优化（执行顺序、流水、autotune……）。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                   stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                   ACC_DTYPE: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    # 指向 A 的第一条（BLOCK_M × BLOCK_K）和 B 的第一条（BLOCK_K × BLOCK_N）
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    for k in range(0, K, BLOCK_K):
        # 三个方向都可能不是 BLOCK 的整数倍：越界的位置填 0，乘出来也是 0，不影响累加
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=ACC_DTYPE)
        a_ptrs += BLOCK_K * stride_ak             # 沿 K 往前走一条
        b_ptrs += BLOCK_K * stride_bk
    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul(a: torch.Tensor, b: torch.Tensor, BLOCK_M: int = 128, BLOCK_N: int = 128, BLOCK_K: int = 32,
           num_warps: int = 8, acc_dtype=tl.float32) -> torch.Tensor:
    assert a.dim() == 2 and b.dim() == 2 and a.shape[1] == b.shape[0], f"形状不对：{a.shape} @ {b.shape}"
    assert a.dtype == b.dtype and a.device == b.device
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_kernel[grid](a, b, c, M, N, K, *a.stride(), *b.stride(), *c.stride(),
                         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, ACC_DTYPE=acc_dtype,
                         num_warps=num_warps)
    return c
