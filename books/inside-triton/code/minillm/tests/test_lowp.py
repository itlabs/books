"""第 16 章：mxfp4 量化与「bf16 × mxfp4」矩阵乘。"""
import pytest
import torch

from minillm.kernels.lowp import E2M1_VALUES, dequantize_mxfp4, matmul_mxfp4, quantize_mxfp4


def test_mxfp4_roundtrip_exact(device):
    # 每 32 个一组里，所有能精确表示的值（乘一个 2 的幂）量化再反量化，必须原样回来
    vals = torch.tensor(E2M1_VALUES + tuple(-v for v in E2M1_VALUES), device=device)
    w = vals.repeat(4)[:32].reshape(32, 1) * torch.tensor([1.0, 0.25, 64.0, 2.0**-20], device=device)
    packed, scale = quantize_mxfp4(w)
    assert packed.shape == (16, 4) and scale.shape == (4, 1)
    torch.testing.assert_close(dequantize_mxfp4(packed, scale), w, atol=0, rtol=0)


def test_mxfp4_error(device):
    # 正态分布：e2m1 只有 1 位尾数，相对误差大约 10%（第 16 章的浏览器模型）
    w = torch.randn(1024, 256, device=device)
    err = (dequantize_mxfp4(*quantize_mxfp4(w)) - w).norm() / w.norm()
    assert 0.05 < err < 0.15


@pytest.mark.parametrize("m", [1, 16, 37, 64, 300])
@pytest.mark.parametrize("k, n", [(64, 16), (1024, 384), (4096, 1008)])
def test_matmul_mxfp4(device, m, k, n):
    w = torch.randn(k, n, device=device) * 0.05
    packed, scale = quantize_mxfp4(w)
    a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    ref = a.float() @ dequantize_mxfp4(packed, scale)            # 和反量化以后的 fp32 矩阵乘比
    out = matmul_mxfp4(a, packed, scale)
    torch.testing.assert_close(out.float(), ref, atol=2e-2 * ref.abs().max().item(), rtol=2e-2)
