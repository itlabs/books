"""第 6 章：长行 softmax 的三种写法。

    python bench/bench_softmax.py

每一档都是 512 MiB 的 fp32 输入（行数 × 列数 = 2^27），只有行长不同。
预测（按读写的趟数，假设都走显存）：一次装下是读 1 写 1；online 读 2 写 1；三遍读 3 写 1。
"""
import torch
import triton

from minillm import DEVICE
from minillm.kernels.softmax import _softmax_fused_kernel, num_warps_for, softmax_3pass, softmax_online


def fused_whole_row(x):
    """不管多长都一次装下：BLOCK = 整行，num_warps 尽量大。只用来对比。"""
    y = torch.empty_like(x)
    block = triton.next_power_of_2(x.shape[1])
    _softmax_fused_kernel[(x.shape[0],)](x, y, x.shape[1], x.stride(0), y.stride(0),
                                         BLOCK=block, num_warps=num_warps_for(block))
    return y


def main():
    print(f"{getattr(torch, DEVICE.type).get_device_name()}，torch {torch.__version__}，triton {triton.__version__}")
    for n_cols in (32768, 131072, 262144):
        x = torch.randn(2**27 // n_cols, n_cols, device=DEVICE)
        nbytes = 2 * x.numel() * x.element_size()
        print(f"  {x.shape[0]} 行 × {n_cols} 列")
        impls = [("一次装下", fused_whole_row), ("online（读 2 遍）", softmax_online),
                 ("三遍（读 3 遍）", softmax_3pass), ("torch.softmax", lambda t: torch.softmax(t, dim=-1))]
        for name, fn in impls:
            torch.testing.assert_close(fn(x), torch.softmax(x, dim=-1))
            ms = triton.testing.do_bench(lambda: fn(x))
            print(f"    {name:18} {ms:7.3f} ms   有效带宽 {nbytes / ms / 1e6:5.0f} GB/s")


if __name__ == "__main__":
    main()
