"""各章新增的 Triton kernel。

    ch04  pointwise.py   add、silu_mul
    ch06  norm.py        rmsnorm
          softmax.py     softmax（一次装下 / online）
    ch07  matmul.py      朴素 tile 版，ch08、ch12–14 逐步优化
    ch16  lowp.py        mxfp4 权重量化与矩阵乘
    ch17  attention.py   FlashAttention 前向
    ……（完整进度见 ../README.md）
"""
