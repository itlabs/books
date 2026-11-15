"""全项目统一的设备：当前 Triton 驱动对应的 torch 设备（NVIDIA / AMD 是 cuda，Intel 是 xpu）。

kernel 和测试里一律用它，不写死 "cuda"——同一份代码要在 A10 和 Max 1100 上都跑通（BRIEF §3.1）。
"""
import triton

DEVICE = triton.runtime.driver.active.get_active_torch_device()
