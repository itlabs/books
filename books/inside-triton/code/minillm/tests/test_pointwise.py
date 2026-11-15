"""第 4 章：add、silu_mul 与 PyTorch 参考实现对比。"""
import pytest
import torch
import torch.nn.functional as F

from minillm.kernels.pointwise import add, silu_mul

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
# 1：只有一个元素；1000：不是 BLOCK 的整数倍；98432：第 3 章的例子；
# 2**20 + 7：不是 16 的倍数，会走没有向量化的那个特化版本（第 3、4 章）
SIZES = [1, 1000, 98432, 2**20 + 7]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", SIZES)
def test_add(device, dtype, n):
    x = torch.randn(n, device=device, dtype=dtype)
    y = torch.randn(n, device=device, dtype=dtype)
    # 逐点加法在同一个 dtype 里算，和 PyTorch 应当逐位相同
    torch.testing.assert_close(add(x, y), x + y, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", SIZES)
def test_silu_mul(device, dtype, n):
    gate = torch.randn(n, device=device, dtype=dtype)
    up = torch.randn(n, device=device, dtype=dtype)
    # 参考实现也在 fp32 里算，再转回输入类型；容差用 assert_close 对该 dtype 的默认值
    ref = (F.silu(gate.float()) * up.float()).to(dtype)
    torch.testing.assert_close(silu_mul(gate, up), ref)


def test_shape_2d(device):
    # 连续的多维张量按一维处理即可
    gate = torch.randn(64, 1000, device=device)
    up = torch.randn(64, 1000, device=device)
    torch.testing.assert_close(silu_mul(gate, up), F.silu(gate) * up)


def test_empty(device):
    x = torch.empty(0, device=device)
    assert add(x, x).numel() == 0


def test_rejects_non_contiguous(device):
    x = torch.randn(64, 64, device=device).t()
    with pytest.raises(AssertionError, match="连续"):
        add(x, x)
