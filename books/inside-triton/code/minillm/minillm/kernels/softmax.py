"""按行 softmax（第 6 章）。

两种写法：
  - 一行放得进一个 program（BLOCK = next_pow2(n_cols) ≤ MAX_FUSED）：读一遍、写一遍；
  - 放不进：沿着行分段循环，用 online softmax 只读两遍（第一遍求最大值和累加和，第二遍写结果）。
另外留了一个三遍的版本 `softmax_3pass`，只用来和 online 版本对比（正文第 6 章）。
"""
import torch
import triton
import triton.language as tl

# 一个 program 最多一次装下多少个元素。超过它就改走循环版本。
# 取值依据见第 6 章的溢出实验：配合下面的 num_warps，每个线程最多拿 32 个元素。
MAX_FUSED = 16384


def num_warps_for(block: int) -> int:
    """每个线程大约拿 8–32 个元素：BLOCK 越大，warp 越多（上限 32 个 warp，也就是 1024 个线程）。"""
    return max(4, min(32, block // 512))


@triton.jit
def _softmax_fused_kernel(x_ptr, y_ptr, n_cols, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)          # 行号转成 int64：行数 × stride 可能超过 2^31（第 5 章）
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)                    # 减去最大值：exp 不会溢出
    e = tl.exp(x)
    y = e / tl.sum(e, axis=0)
    tl.store(y_ptr + row * stride_y + cols, y, mask=mask)   # store 会转回 y 的 dtype


@triton.jit
def _softmax_online_kernel(x_ptr, y_ptr, n_cols, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    x_row = x_ptr + row * stride_x
    y_row = y_ptr + row * stride_y
    # 第一遍：一边扫，一边维护「到目前为止的最大值 m」和「以 m 为基准的指数和 l」
    m = -float("inf")
    l = 0.0
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(x_row + cols, mask=cols < n_cols, other=-float("inf")).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, axis=0))
        # 最大值变大了：之前累加的和要按新的基准缩小 exp(m - m_new) 倍
        l = l * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new
    # 第二遍：有了整行的 m 和 l，再读一遍，写出结果
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=-float("inf")).to(tl.float32)
        tl.store(y_row + cols, tl.exp(x - m) / l, mask=mask)


@triton.jit
def _softmax_3pass_kernel(x_ptr, y_ptr, n_cols, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    x_row = x_ptr + row * stride_x
    y_row = y_ptr + row * stride_y
    m = -float("inf")
    for start in range(0, n_cols, BLOCK):        # 第一遍：最大值
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(x_row + cols, mask=cols < n_cols, other=-float("inf")).to(tl.float32)
        m = tl.maximum(m, tl.max(x, axis=0))
    l = 0.0
    for start in range(0, n_cols, BLOCK):        # 第二遍：指数和
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(x_row + cols, mask=cols < n_cols, other=-float("inf")).to(tl.float32)
        l += tl.sum(tl.exp(x - m), axis=0)
    for start in range(0, n_cols, BLOCK):        # 第三遍：写结果
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=-float("inf")).to(tl.float32)
        tl.store(y_row + cols, tl.exp(x - m) / l, mask=mask)


def _prepare(x):
    assert x.dim() == 2, "只支持二维输入（行数, 列数）；更高维先 reshape"
    assert x.stride(1) == 1, "最后一维必须连续"
    return torch.empty_like(x), x.shape[0], x.shape[1]


def softmax(x: torch.Tensor, loop_block: int = 4096) -> torch.Tensor:
    y, n_rows, n_cols = _prepare(x)
    block = triton.next_power_of_2(n_cols)
    if block <= MAX_FUSED:
        _softmax_fused_kernel[(n_rows,)](x, y, n_cols, x.stride(0), y.stride(0),
                                         BLOCK=block, num_warps=num_warps_for(block))
    else:
        _softmax_online_kernel[(n_rows,)](x, y, n_cols, x.stride(0), y.stride(0),
                                          BLOCK=loop_block, num_warps=num_warps_for(loop_block))
    return y


def softmax_online(x: torch.Tensor, loop_block: int = 4096) -> torch.Tensor:
    """强制走循环的 online 版本（测试和对比用）。"""
    y, n_rows, n_cols = _prepare(x)
    _softmax_online_kernel[(n_rows,)](x, y, n_cols, x.stride(0), y.stride(0),
                                      BLOCK=loop_block, num_warps=num_warps_for(loop_block))
    return y


def softmax_3pass(x: torch.Tensor, loop_block: int = 4096) -> torch.Tensor:
    """三遍的循环版本（对比用）。"""
    y, n_rows, n_cols = _prepare(x)
    _softmax_3pass_kernel[(n_rows,)](x, y, n_cols, x.stride(0), y.stride(0),
                                     BLOCK=loop_block, num_warps=num_warps_for(loop_block))
    return y
