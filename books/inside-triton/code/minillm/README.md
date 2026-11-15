# minillm

《深入 Triton》的贯穿项目：一个简化的 decoder-only Transformer 层，所有 kernel 用 Triton 写。
从第 4 章起每章往这里加东西，正文只贴关键片段，完整代码在这里。

## 安装与测试

先按附录 A 装好与你的 GPU 配套的 `torch` 和 `triton`（它们故意没写进依赖），然后：

```bash
cd books/inside-triton/code/minillm
uv pip install -e '.[test]'     # 或 pip install -e '.[test]'
pytest                          # 与 PyTorch 参考实现逐个对比数值
```

在仓库根目录也可以直接跑 `books/inside-triton/verify/verify-minillm.sh`。

## 结构

```
minillm/device.py      DEVICE：当前 Triton 驱动对应的 torch 设备，全项目统一用它
minillm/kernels/       各章新增的 kernel
tests/                 pytest；test_env.py 是环境冒烟测试
bench/                 triton.testing.do_bench 的性能测量
```

## 进度

| 章 | 新增 | 状态 |
| --- | --- | --- |
| ch04 | `add`、`silu_mul` | 完成（`kernels/pointwise.py`） |
| ch06 | `rmsnorm`、`softmax` | 完成（`kernels/norm.py`、`kernels/softmax.py`） |
| ch07 | `matmul`（朴素 tile 版） | 未开始 |
| ch08 | `matmul` 改用 tensor descriptor | 未开始 |
| ch10 | 全部包成 `torch.library` custom op | 未开始 |
| ch12–15 | `matmul` 逐步优化 | 未开始 |
| ch16 | `matmul_fp8`、block-scaled | 未开始 |
| ch17 | `flash_attn_fwd` | 未开始 |
| ch25 | Gluon 版 `matmul` | 未开始 |
| ch26 | Helion 版 `rmsnorm`、`matmul` | 未开始 |
| ch28 | `moe_grouped_gemm` + 路由 permute | 未开始 |
| ch29 | `decode_attn`（paged、split-k） | 未开始 |
| ch30 | 一层前向串起来 | 未开始 |
