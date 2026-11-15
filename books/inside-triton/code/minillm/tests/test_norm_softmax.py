"""第 6 章：softmax、rmsnorm 与 PyTorch 参考实现对比。"""
import pytest
import torch

from minillm.kernels.norm import rmsnorm
from minillm.kernels.softmax import softmax, softmax_3pass, softmax_online

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def ref_softmax(x):
    return torch.softmax(x.float(), dim=-1).to(x.dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n_cols", [1, 7, 1000, 4096, 16384, 16385, 100000])
def test_softmax(device, dtype, n_cols):
    # 16384 是一次装下的上限；16385 起走 online 循环版本
    x = torch.randn(8, n_cols, device=device, dtype=dtype) * 4
    torch.testing.assert_close(softmax(x), ref_softmax(x))


@pytest.mark.parametrize("fn", [softmax_online, softmax_3pass])
@pytest.mark.parametrize("n_cols", [1, 1000, 4097, 50000])
def test_softmax_loops(device, fn, n_cols):
    # 两个循环版本在任何长度上都要对，包括比一个 BLOCK 还短的行
    x = torch.randn(8, n_cols, device=device) * 4
    torch.testing.assert_close(fn(x), ref_softmax(x))


def test_softmax_large_logits(device):
    # 不减最大值就会 exp 溢出的输入：结果必须仍然正确、没有 nan
    x = torch.randn(4, 50000, device=device) * 10 + 1000
    for fn in (softmax, softmax_online, softmax_3pass):
        y = fn(x)
        assert not torch.isnan(y).any()
        torch.testing.assert_close(y, ref_softmax(x))


def test_softmax_row_stride(device):
    # 行之间不连续（比如从一个更大的张量里切出来的列块），最后一维连续
    big = torch.randn(16, 3000, device=device)
    x = big[:, :2000]
    torch.testing.assert_close(softmax(x), ref_softmax(x))


def ref_rmsnorm(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("hidden", [7, 1000, 4096, 5120, 16384])
def test_rmsnorm(device, dtype, hidden):
    x = torch.randn(2, 33, hidden, device=device, dtype=dtype)     # 三维：前两维都当成行
    w = torch.randn(hidden, device=device, dtype=dtype)
    torch.testing.assert_close(rmsnorm(x, w, 1e-6), ref_rmsnorm(x, w, 1e-6))
