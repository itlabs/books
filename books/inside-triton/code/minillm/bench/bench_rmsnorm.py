"""第 6 章：RMSNorm。

    python bench/bench_rmsnorm.py

预测：eager 写法是 5 个 kernel（转 fp32、平方、求均值、乘 rsqrt、乘 weight 再转回），
中间结果都落显存；融合以后读一遍 x、写一遍 y（weight 很小，可以忽略）。
"""
import torch
import torch.nn.functional as F
import triton

from minillm import DEVICE
from minillm.kernels.norm import rmsnorm


def rmsnorm_eager(x, w, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


def main():
    print(f"{getattr(torch, DEVICE.type).get_device_name()}，torch {torch.__version__}，triton {triton.__version__}")
    compiled = torch.compile(rmsnorm_eager)
    for hidden in (4096, 8192):
        x = torch.randn(8192, hidden, device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn(hidden, device=DEVICE, dtype=torch.bfloat16)
        nbytes = 2 * x.numel() * x.element_size()
        print(f"  8192 个 token × hidden {hidden}，bf16")
        impls = [("eager", lambda: rmsnorm_eager(x, w)), ("torch.compile", lambda: compiled(x, w)),
                 ("F.rms_norm", lambda: F.rms_norm(x, (hidden,), w, 1e-6)), ("minillm", lambda: rmsnorm(x, w))]
        for name, fn in impls:
            torch.testing.assert_close(fn(), rmsnorm_eager(x, w))
            ms = triton.testing.do_bench(fn)
            print(f"    {name:14} {ms * 1e3:7.1f} us   有效带宽 {nbytes / ms / 1e6:5.0f} GB/s")


if __name__ == "__main__":
    main()
