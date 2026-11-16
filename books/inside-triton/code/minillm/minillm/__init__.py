"""minillm：《深入 Triton》的贯穿项目。

每章往 minillm/kernels/ 里加 kernel，往 tests/ 里加与 PyTorch 参考实现的数值对比，
往 bench/ 里加性能测量。进度见 README.md。
"""
from .device import DEVICE

__all__ = ["DEVICE"]
