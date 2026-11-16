"""第 4 章：逐点 kernel 的性能。

    python bench/bench_pointwise.py

形状取一个 8B 级别模型 MLP 中间层的大小：4096 个 token × 14336，bf16。
预测（第 1 章的数趟数）：
    add       eager 和 Triton 都是读 2 趟、写 1 趟，应当差不多快；
    silu_mul  eager 是 F.silu 读 1 写 1、再乘法读 2 写 1，共 5 趟；融合后读 2 写 1，共 3 趟，
              所以应当快 5/3 ≈ 1.7 倍。
"""
import torch
import torch.nn.functional as F
import triton

from minillm import DEVICE
from minillm.kernels.pointwise import add, silu_mul


def main():
    shape, dtype = (4096, 14336), torch.bfloat16
    a = torch.randn(shape, device=DEVICE, dtype=dtype)
    b = torch.randn(shape, device=DEVICE, dtype=dtype)
    nbytes = a.numel() * a.element_size()
    print(f"{getattr(torch, DEVICE.type).get_device_name()}，torch {torch.__version__}，triton {triton.__version__}，"
          f"{shape[0]}×{shape[1]} {str(dtype).removeprefix('torch.')}（每个张量 {nbytes / 2**20:.0f} MiB）")
    cases = [
        ("add       eager", lambda: a + b, 3),
        ("add       minillm", lambda: add(a, b), 3),
        ("silu_mul  eager", lambda: F.silu(a) * b, 5),
        ("silu_mul  minillm", lambda: silu_mul(a, b), 3),
    ]
    for name, fn, passes in cases:
        ms = triton.testing.do_bench(fn)
        print(f"  {name:18} {ms * 1e3:8.1f} us   搬 {passes} 趟，按趟数算的带宽 {passes * nbytes / ms / 1e6:5.0f} GB/s")


if __name__ == "__main__":
    main()
