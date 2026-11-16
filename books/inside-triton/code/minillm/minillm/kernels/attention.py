"""FlashAttention 前向（第 17 章）。

O = softmax(Q Kᵀ · scale) V，Q、K、V、O 的形状都是 (Z, H, N, D)、连续存放。
每个 program 算 O 的一块 BLOCK_M 行：沿 N 方向每次读 K、V 的 BLOCK_N 行，
用 online softmax（第 6 章）一边扫一边修正累加器，N × N 的注意力矩阵从来不写回显存。
"""
import torch
import triton
import triton.language as tl

from ..device import DEVICE

@triton.jit
def _attn_fwd_kernel(q_ptr, k_ptr, v_ptr, o_ptr, sm_scale, N_CTX, ROWS,
                     HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     CAUSAL: tl.constexpr, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr):
    pid_m = tl.program_id(0)
    hz = tl.program_id(1)                          # 第几个 (batch, head)
    # 把 (Z, H, N, D) 看成 (Z·H·N, D) 的二维矩阵：第 hz 个头从第 hz·N 行开始
    q_desc = tl.make_tensor_descriptor(q_ptr, [ROWS, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_M, HEAD_DIM])
    k_desc = tl.make_tensor_descriptor(k_ptr, [ROWS, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_N, HEAD_DIM])
    v_desc = tl.make_tensor_descriptor(v_ptr, [ROWS, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_N, HEAD_DIM])
    base = hz * N_CTX
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q = q_desc.load([base + pid_m * BLOCK_M, 0])
    qk_scale = sm_scale * 1.4426950408889634        # 用 exp2 代替 exp：exp(x) = exp2(x·log2(e))

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)   # 每一行到目前为止的最大值
    l_i = tl.zeros([BLOCK_M], tl.float32)                  # 每一行以 m_i 为基准的指数和
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)        # 还没除以 l_i 的输出
    # causal：第 i 行只看前 i 列，这一块之后的 K、V 整块跳过
    hi = tl.minimum((pid_m + 1) * BLOCK_M, N_CTX) if CAUSAL else N_CTX
    for start_n in range(0, hi, BLOCK_N):
        k = k_desc.load([base + start_n, 0])
        qk = tl.dot(q, tl.trans(k)) * qk_scale             # BLOCK_M × BLOCK_N，fp32
        offs_n = start_n + tl.arange(0, BLOCK_N)
        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, -float("inf"))
        elif not EVEN_N:
            # 最后一块越过了这个头的末尾：读到的是下一个头的 K（或越界填的 0），要屏蔽掉
            qk = tl.where(offs_n[None, :] < N_CTX, qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)                  # 最大值变大了：之前的累加要缩小这么多倍
        l_i = l_i * alpha + tl.sum(p, 1)
        v = v_desc.load([base + start_n, 0])
        acc = tl.dot(p.to(v.dtype), v, acc * alpha[:, None])
        m_i = m_new
    o = (acc / l_i[:, None]).to(o_ptr.dtype.element_ty)
    if EVEN_M:
        o_desc = tl.make_tensor_descriptor(o_ptr, [ROWS, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_M, HEAD_DIM])
        o_desc.store([base + pid_m * BLOCK_M, 0], o)
    else:
        # N 不是 BLOCK_M 的整数倍：最后一块的尾巴属于下一个头，descriptor 不知道头的边界，改用带 mask 的 store
        offs_d = tl.arange(0, HEAD_DIM)
        o_ptrs = o_ptr + (base + offs_m)[:, None].to(tl.int64) * HEAD_DIM + offs_d[None, :]
        tl.store(o_ptrs, o, mask=offs_m[:, None] < N_CTX)


def launch_config(n_ctx, head_dim, causal):
    """返回 (BLOCK_M, BLOCK_N, 其余启动参数)。Intel 的取值是第 17 章在 Max 1100 上扫出来的。"""
    if DEVICE.type == "xpu":
        bm, bn, opts = 256, 128, dict(num_warps=32, num_stages=2)
        if causal and head_dim > 64:
            # 因果时多了一个 BLOCK_M × BLOCK_N 的 mask，256×128 溢出得一塌糊涂（第 17 章的表），退到 BLOCK_N = 64；
            # 并且关掉 Intel 后端把 Q 的 load 挪进循环里的 ReduceVariableLiveness pass：这种情形下关掉快 24%
            bn = 64
            opts["reduce_variable_liveness"] = False
    else:
        bm, bn, opts = 128, 64, dict(num_warps=8 if head_dim > 64 else 4, num_stages=3)
    bm = min(bm, max(16, triton.next_power_of_2(n_ctx)))       # 很短的序列不必开那么大的块
    bn = min(bn, bm)
    opts["num_warps"] = max(1, min(opts["num_warps"], bm // 8))   # 每个 warp 至少分到 8 行
    return bm, bn, opts


def attention(q, k, v, causal=False, sm_scale=None):
    """q、k、v：(Z, H, N, D)，D ∈ {16, 32, 64, 128}。返回和 q 同形状、同 dtype 的输出。"""
    Z, H, N, D = q.shape
    assert k.shape == v.shape == q.shape, "本章只做 Q、K、V 同长度、同头数的情形"
    assert D in (16, 32, 64, 128), "HEAD_DIM 要是 2 的幂，且一块 Q 要放得进寄存器"
    assert q.dtype in (torch.float16, torch.bfloat16), "矩阵单元的输入：fp16 或 bf16"
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    o = torch.empty_like(q)
    if q.numel() == 0:
        return o
    sm_scale = D ** -0.5 if sm_scale is None else sm_scale
    bm, bn, opts = launch_config(N, D, causal)
    grid = (triton.cdiv(N, bm), Z * H)
    _attn_fwd_kernel[grid](q, k, v, o, sm_scale, N, Z * H * N, HEAD_DIM=D, BLOCK_M=bm, BLOCK_N=bn,
                           CAUSAL=causal, EVEN_M=N % bm == 0, EVEN_N=N % bn == 0, **opts)
    return o
