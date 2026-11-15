"""环境冒烟测试：设备是 GPU，并且能真的编译、启动一个 Triton kernel。

这个文件先于所有 kernel 存在：kernel 的测试失败时，先看这里有没有过，
就能分清是环境问题还是 kernel 的问题。
"""
import torch
import triton
import triton.language as tl


def test_device_is_gpu(device):
    assert device.type in ("cuda", "xpu"), f"Triton 的活动设备是 {device}，不是 GPU"


@triton.jit
def _fill_kernel(out_ptr, value, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, value, mask=offs < n)


def test_triton_launch(device):
    n = 1000  # 故意不是 BLOCK 的整数倍：顺带验证 mask
    out = torch.zeros(n, device=device)
    _fill_kernel[(triton.cdiv(n, 256),)](out, 3.0, n, BLOCK=256)
    torch.testing.assert_close(out, torch.full_like(out, 3.0))
