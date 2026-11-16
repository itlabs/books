"""低精度（第 16 章）：mxfp4 权重量化，以及「bf16 激活 × mxfp4 权重」的矩阵乘。

mxfp4（OCP Microscaling 规范）：每个元素是 4 位的 e2m1（可表示 0、0.5、1、1.5、2、3、4、6 和它们的负数），
K 方向每 32 个元素共用一个 8 位的 e8m0 scale（2 的幂，偏置 127）。权重按 (K, N) 存时：
    packed：(K/2, N) uint8，K 方向相邻两个元素一个字节，前一个在低 4 位（tl.dot_scaled 的约定）
    scale ：(N, K/32) uint8（tl.dot_scaled 的约定：rhs 的 scale 是 [N, K/32]，不要转置）
"""
import torch
import triton
import triton.language as tl

from ..device import DEVICE

E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)   # code 0–7 的绝对值；第 3 位是符号


def quantize_mxfp4(w: torch.Tensor):
    """w：(K, N)，K 是 32 的倍数。返回 (packed, scale)。"""
    K, N = w.shape
    assert K % 32 == 0, "K 要是 32 的倍数（每 32 个元素一个 scale）"
    g = w.float().t().reshape(N, K // 32, 32)
    amax = g.abs().amax(-1, keepdim=True).clamp_min(2.0 ** -126)
    # 规范的做法：scale = 2^(floor(log2(amax)) − 2)，2 是 e2m1 最大值 6 = 1.5·2² 的指数。
    # 于是 amax / scale 落在 [4, 8)，大于 6 的部分饱和到 6
    e = torch.floor(torch.log2(amax)) - 2
    q = g / torch.exp2(e)
    grid = torch.tensor(E2M1_VALUES, device=w.device)
    idx = (q.abs().unsqueeze(-1) - grid).abs().argmin(-1)          # 舍入到最近的可表示值
    code = (idx | ((q < 0).to(idx.dtype) << 3)).to(torch.uint8)
    code = code.reshape(N, K).t()                                   # (K, N)
    packed = (code[0::2] | (code[1::2] << 4)).contiguous()
    scale = (e.squeeze(-1) + 127).clamp(0, 254).to(torch.uint8).contiguous()
    return packed, scale


def dequantize_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """quantize_mxfp4 的逆：返回 (K, N) 的 fp32。"""
    code = torch.stack([packed & 0xF, packed >> 4], 1).reshape(-1, packed.shape[1])
    grid = torch.tensor(E2M1_VALUES, device=packed.device)
    v = grid[(code & 7).long()] * torch.where((code >> 3) == 1, -1.0, 1.0)
    return v * torch.exp2(scale.float() - 127).t().repeat_interleave(32, 0)


@triton.jit
def _matmul_mxfp4_kernel(a_ptr, w_ptr, s_ptr, c_ptr, M, N, K,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """C = A · dequant(W)。BLOCK_K 固定为 32：每一步只涉及一组 scale（每列一个），
    于是 scale 乘在这一步的部分和上（BLOCK_M × BLOCK_N 次乘法），而不是乘在每个权重上
    （32 × BLOCK_N 次乘法再加一次舍入）。在没有 fp4 矩阵单元的 GPU 上，这比把 scale 交给
    tl.dot_scaled 快（第 16 章的实测）。"""
    BLOCK_K: tl.constexpr = 32
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    a_desc = tl.make_tensor_descriptor(a_ptr, [M, K], [K, 1], [BLOCK_M, BLOCK_K])
    w_desc = tl.make_tensor_descriptor(w_ptr, [K // 2, N], [N, 1], [BLOCK_K // 2, BLOCK_N])
    c_desc = tl.make_tensor_descriptor(c_ptr, [M, N], [N, 1], [BLOCK_M, BLOCK_N])
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(0, K // BLOCK_K):
        a = a_desc.load([pid_m * BLOCK_M, i * BLOCK_K])
        w = w_desc.load([i * (BLOCK_K // 2), pid_n * BLOCK_N])
        s = tl.load(s_ptr + offs_n * (K // 32) + i, mask=offs_n < N, other=127)
        part = tl.dot_scaled(a, None, "bf16", w, None, "e2m1")      # 只解包，不乘 scale
        acc += part * tl.math.exp2(s.to(tl.float32) - 127.0)[None, :]
    c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], acc.to(c_desc.dtype))


def _config(m):
    """(BLOCK_M, BLOCK_N, num_warps)。Intel 的取值是第 16 章在 Max 1100 上扫出来的；NVIDIA 是随手挑的，未实测。"""
    if DEVICE.type == "xpu":
        if m <= 16:
            return 16, 128, 8
        if m <= 64:
            return 64, 128, 8
        return 256, 128, 32
    return (16, 128, 4) if m <= 16 else (64, 128, 4)


def matmul_mxfp4(a: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """a：(M, K) bf16；packed、scale：quantize_mxfp4 的输出。返回 (M, N) bf16。"""
    M, K = a.shape
    N = packed.shape[1]
    assert a.dtype == torch.bfloat16 and packed.shape[0] * 2 == K and scale.shape == (N, K // 32)
    assert K % 32 == 0 and N % 16 == 0, "K 要是 32 的倍数；N 要是 16 的倍数（packed 的行 stride 是 N 字节，descriptor 要求 16 字节对齐）"
    a = a.contiguous()
    c = torch.empty(M, N, device=a.device, dtype=torch.bfloat16)
    if M == 0:
        return c
    bm, bn, nw = _config(M)
    _matmul_mxfp4_kernel[(triton.cdiv(N, bn), triton.cdiv(M, bm))](a, packed, scale, c, M, N, K,
                                                                   BLOCK_M=bm, BLOCK_N=bn, num_warps=nw)
    return c
