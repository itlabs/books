"""第 9 章：结果对，不代表编译对。

数值测试（test_matmul.py 等）只能保证结果正确。这里检查的是**编译产物**：
该用矩阵单元的有没有用上、该用二维块读写的有没有用上、该一次装下的有没有寄存器溢出。
这几类问题都不会报错，只会让 kernel 悄悄慢一个数量级（附录 A 的 ocloc、第 7 章的 tf32x3、
第 9 章的小尺寸 tl.dot）。

取编译产物用的是 JITFunction.warmup：只编译、不启动，返回 CompiledKernel（官方教程 02 也这么用）。
寄存器溢出量要在 kernel 真正加载到设备上以后才有，所以那一项是真的启动一次。
"""
import re

import pytest
import torch
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from minillm import DEVICE
from minillm.kernels import matmul as mm
from minillm.kernels.norm import _rmsnorm_kernel
from minillm.kernels.softmax import MAX_FUSED, _softmax_fused_kernel, num_warps_for

MATRIX_LAYOUT = re.compile(r"#ttg\.nvidia_mma|#ttig\.dpas|#ttg\.amd_mfma")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matmul_uses_matrix_units(dtype):
    k = mm._matmul_kernel.warmup(dtype, dtype, dtype, 256, 256, 256, 256, 1, 256, 1, 256, 1,
                                 BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, ACC_DTYPE=tl.float32, num_warps=8,
                                 grid=(1,))
    assert MATRIX_LAYOUT.search(k.asm["ttgir"]), "tl.dot 没有用上矩阵单元（TTGIR 里没有 mma / dpas layout）"


@pytest.mark.skipif(DEVICE.type != "xpu", reason="二维块读写是 Intel GPU 的特性")
def test_matmul_desc_uses_2d_block_io():
    a = torch.empty(512, 512, device=DEVICE, dtype=torch.float16)
    descs = [TensorDescriptor.from_tensor(a, [256, 32]), TensorDescriptor.from_tensor(a, [32, 256]),
             TensorDescriptor.from_tensor(a, [256, 256])]
    k = mm._matmul_desc_kernel.warmup(*descs, 512, BLOCK_M=256, BLOCK_N=256, BLOCK_K=32, num_warps=32, grid=(1,))
    assert "ttig.block_io" in k.asm["ttgir"], "descriptor 的读写没有被标成 block_io"
    assert "LSC2DBlockRead" in k.asm["llir"], "LLVM IR 里没有二维块读取指令"
    assert MATRIX_LAYOUT.search(k.asm["ttgir"])


@pytest.mark.parametrize("n_cols", [1024, 4096, 8192, MAX_FUSED])
def test_softmax_fused_no_spills(device, n_cols):
    # 第 6 章：按 num_warps_for 的规则，一次装下的上限以内都不应该溢出（Max 1100 实测）
    x = torch.randn(4, n_cols, device=device)
    y = torch.empty_like(x)
    k = _softmax_fused_kernel[(4,)](x, y, n_cols, x.stride(0), y.stride(0),
                                    BLOCK=n_cols, num_warps=num_warps_for(n_cols))
    assert k.n_spills == 0, f"{n_cols} 列、{num_warps_for(n_cols)} 个 warp 溢出了 {k.n_spills}"


def test_rmsnorm_no_spills(device):
    x = torch.randn(4, 8192, device=device, dtype=torch.bfloat16)
    w = torch.randn(8192, device=device, dtype=torch.bfloat16)
    y = torch.empty_like(x)
    k = _rmsnorm_kernel[(4,)](x, w, y, 8192, x.stride(0), y.stride(0), 1e-6,
                              BLOCK=8192, num_warps=num_warps_for(8192))
    assert k.n_spills == 0
