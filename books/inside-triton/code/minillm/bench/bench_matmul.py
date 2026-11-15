"""第 7 章：朴素 tile GEMM 的性能，和 torch.matmul 比。

    python bench/bench_matmul.py

第 7 章只看量级；第 8 章换访存写法，第 12–15 章逐步优化。
"""
import torch
import triton

from minillm import DEVICE
from minillm.kernels.matmul import matmul


def main():
    M = N = K = 4096
    print(f"{getattr(torch, DEVICE.type).get_device_name()}，torch {torch.__version__}，triton {triton.__version__}，"
          f"{M}×{N}×{K}")
    for dtype in (torch.float16, torch.bfloat16):
        a = torch.randn(M, K, device=DEVICE, dtype=dtype)
        b = torch.randn(K, N, device=DEVICE, dtype=dtype)
        print(f"  {str(dtype).removeprefix('torch.')}")
        for name, fn in (("minillm（128×128×32）", lambda: matmul(a, b)), ("torch.matmul", lambda: a @ b)):
            ms = triton.testing.do_bench(fn)
            print(f"    {name:22} {ms:6.3f} ms  {2 * M * N * K / ms / 1e9:6.1f} TFLOP/s")


if __name__ == "__main__":
    main()
