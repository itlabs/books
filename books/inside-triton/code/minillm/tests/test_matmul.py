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


# ---------------------------------------------------------------- 第 8 章 ----
from minillm.kernels.matmul import descriptor_ok, matmul_desc, mm  # noqa: E402


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mnk", [(256, 256, 64), (1000, 776, 1032), (8, 4096, 512), (2048, 2048, 4096)])
def test_matmul_desc(device, dtype, mnk):
    # 1000×776×1032：都不是 tile 的整数倍，但行 stride 都是 16 字节的倍数，descriptor 能用；
    # 越界的部分由 descriptor 自己处理（读到 0、不写）
    m, n, k = mnk
    a = torch.randn(m, k, device=device, dtype=dtype)
    b = torch.randn(k, n, device=device, dtype=dtype)
    assert descriptor_ok(a, b)
    assert rel_err(matmul_desc(a, b), a.double() @ b.double()) < TOL[dtype]


def test_descriptor_ok_rejects(device):
    x = torch.randn(64, 777, device=device, dtype=torch.float16)      # 行 stride 1554 字节，不是 16 的倍数
    assert not descriptor_ok(x)
    y = torch.randn(64, 64, device=device, dtype=torch.float16).t()    # 最后一维不连续
    assert not descriptor_ok(y)


@pytest.mark.parametrize("mnk", [(1000, 776, 1032), (333, 777, 1029), (1, 4096, 512)])
def test_mm_dispatch(device, mnk):
    # mm 在能用 descriptor 时用 descriptor，否则退回指针版；两条路都要算对
    m, n, k = mnk
    a = torch.randn(m, k, device=device, dtype=torch.float16)
    b = torch.randn(k, n, device=device, dtype=torch.float16)
    assert rel_err(mm(a, b), a.double() @ b.double()) < TOL[torch.float16]


# ---------------------------------------------------------------- 第 12 章 ----
from minillm.kernels.matmul import matmul_grouped  # noqa: E402


@pytest.mark.parametrize("group_m", [1, 4, 8])
@pytest.mark.parametrize("mnk", [(1000, 776, 1032), (4096, 512, 256), (512, 4096, 1024), (2304, 2304, 512)])
def test_matmul_grouped(device, mnk, group_m):
    # 行数不是 GROUP_M × BLOCK_M 的整数倍时，最后一组不满：检查编号映射没有漏算、没有重复
    m, n, k = mnk
    a = torch.randn(m, k, device=device, dtype=torch.float16)
    b = torch.randn(k, n, device=device, dtype=torch.float16)
    assert rel_err(matmul_grouped(a, b, GROUP_M=group_m), a.double() @ b.double()) < TOL[torch.float16]
