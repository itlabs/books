# bench/

每个 kernel 一个 `bench_<名字>.py`，用 `triton.testing.do_bench` 测，打印 GPU 型号、Triton 版本、输入形状和结果。
`do_bench` 的正确用法（预热、L2 清理、取哪个分位数）在第 11 章讲；那之前的章节只看量级。

正文里引用的性能数字都要注明在哪台 GPU 上测的（BRIEF §13）。
