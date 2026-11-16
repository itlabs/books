"""第 7、8、12 章：GEMM 的几个版本，和 torch.matmul 比。

    python bench/bench_matmul.py

  指针版（第 7 章）、descriptor 版（第 8 章）、descriptor + L2 swizzle（第 12 章）。
  4096³ 的 A、B 合计 64 MiB，放得进 Max 1100 的 L2，swizzle 不起作用；
  16384×16384×4096 的 A、B 合计 256 MiB，放不下，swizzle 才有用（第 12 章）。
"""
import torch
import triton

from minillm import DEVICE
from minillm.kernels.matmul import matmul, matmul_desc, matmul_grouped


def main():
    print(f"{getattr(torch, DEVICE.type).get_device_name()}，torch {torch.__version__}，triton {triton.__version__}，bf16")
    for M, N, K in ((4096, 4096, 4096), (16384, 16384, 4096)):
        a = torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(K, N, device=DEVICE, dtype=torch.bfloat16)
        print(f"  {M}×{N}×{K}")
        cases = [("指针版 128×128×32", lambda: matmul(a, b)),
                 ("descriptor 256×256×32", lambda: matmul_desc(a, b)),
                 ("+ L2 swizzle（GROUP_M=8）", lambda: matmul_grouped(a, b)),
                 ("torch.matmul", lambda: a @ b)]
        for name, fn in cases:
            if name.startswith("指针版") and M > 4096:
                continue                                  # 太慢了，大形状不测
            ms = triton.testing.do_bench(fn)
            print(f"    {name:28} {ms:7.3f} ms  {2 * M * N * K / ms / 1e9:6.1f} TFLOP/s")


if __name__ == "__main__":
    main()
