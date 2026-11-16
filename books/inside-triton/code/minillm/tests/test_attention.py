"""第 17 章：FlashAttention 前向与 PyTorch 的 scaled_dot_product_attention（fp32）对比。"""
import pytest
import torch
import torch.nn.functional as F

from minillm.kernels.attention import attention


def ref_attention(q, k, v, causal):
    return F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=causal).to(q.dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("n", [1, 17, 128, 1000, 1024])
@pytest.mark.parametrize("d", [64, 128])
def test_attention(device, dtype, causal, n, d):
    # 1000、17：不是块大小的整数倍，最后一块的尾巴落在下一个头里（EVEN_M / EVEN_N 为假）
    q, k, v = (torch.randn(2, 3, n, d, device=device, dtype=dtype) for _ in range(3))
    torch.testing.assert_close(attention(q, k, v, causal), ref_attention(q, k, v, causal), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("causal", [False, True])
def test_attention_peaky(device, causal):
    # 注意力很集中（logit 很大）时，exp 的误差最容易放大：第 17 章「缩放放在哪」那一节
    q, k, v = (torch.randn(2, 4, 512, 128, device=device, dtype=torch.bfloat16) for _ in range(3))
    q, k = q * 3, k * 3
    torch.testing.assert_close(attention(q, k, v, causal), ref_attention(q, k, v, causal), atol=2e-2, rtol=2e-2)


def test_attention_empty(device):
    q = torch.empty(1, 2, 0, 64, device=device, dtype=torch.bfloat16)
    assert attention(q, q, q).shape == q.shape
