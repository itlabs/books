"""第 7 章：matmul 与 PyTorch 对比。

容差：输出是 fp16 / bf16 时，最后一次舍入就有 2^-11 / 2^-8 量级的相对误差；
累加顺序不同还会再差一点。所以和 PyTorch 比，用相对于 |C| 最大值的误差，阈值按 dtype 定。
"""
import pytest
import torch

from minillm.kernels.matmul import matmul

TOL = {torch.float16: 2e-3, torch.bfloat16: 1.6e-2, torch.float32: 2e-3}   # fp32 走 TF32（第 7 章）


def rel_err(c, ref):
    return ((c.double() - ref).abs().max() / ref.abs().max()).item()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mnk", [(128, 128, 32), (1000, 1000, 1000), (1, 4096, 512), (4096, 1, 512),
                                 (333, 77, 1029), (2048, 2048, 4096)])
def test_matmul(device, dtype, mnk):
    m, n, k = mnk
    a = torch.randn(m, k, device=device, dtype=dtype)
    b = torch.randn(k, n, device=device, dtype=dtype)
    ref = a.double() @ b.double()
    assert rel_err(matmul(a, b), ref) < TOL[dtype]


def test_matmul_transposed_inputs(device):
    # A、B 都是转置视图：kernel 只看 stride，不要求连续
    a = torch.randn(512, 768, device=device, dtype=torch.float16).t()
    b = torch.randn(1024, 512, device=device, dtype=torch.float16).t()
    assert rel_err(matmul(a, b), a.double() @ b.double()) < TOL[torch.float16]


def test_fp16_accumulator_is_worse(device):
    # 第 7 章的实验：fp16 累加器的误差随 K 增长，fp32 累加器不会
    import triton.language as tl
    a = torch.randn(64, 8192, device=device, dtype=torch.float16)
    b = torch.randn(8192, 64, device=device, dtype=torch.float16)
    ref = a.double() @ b.double()
    assert rel_err(matmul(a, b, 64, 64, 32, 4, tl.float16), ref) > 3 * rel_err(matmul(a, b, 64, 64, 32, 4), ref)
