# 附录 B 概念对照表

## 这份附录怎么用

<div class="goal-box">
<p>同一个概念，在 CUDA、Triton、Gluon、Helion、CuTe DSL 里叫法不同，由谁负责也不同。这份附录把它们排在同一张表里，方便你在读别人的代码、或者在几层之间换着写的时候快速对照。</p>
<p>读表时最该看的，不是名字怎么对应，而是<strong>每一格是「你写」还是「编译器替你做」</strong>：从左往右，CUDA 和 CuTe DSL 让你写得最多，Gluon 次之，Triton 再次，Helion 最少。第 1 章图 1-2 的那几层高度，在这里落到了一个个具体的 API 上。表里的 API 名都在本书基线的源码里核对过：Triton 3.9.0（含 Gluon）、Helion 1.5.2；CuTe DSL 一列来自 NVIDIA/cutlass 仓库 <code>examples/python/CuTeDSL/</code> 下的官方示例，只列了在那里能找到的写法，查不到的写「—」。</p>
</div>

记号：

- **编译器**：这件事由编译器决定，你在源码里写不了也不用写；
- **—**：这一层没有对应的东西，或者本书没有核实过；
- Gluon 的 `gl` 指 `triton.experimental.gluon.language`，Helion 的 `hl` 指 `helion.language`。

## 1. 执行模型

| 概念 | CUDA | Triton | Gluon | Helion | CuTe DSL |
| --- | --- | --- | --- | --- | --- |
| 定义 kernel | `__global__` 函数 | `@triton.jit` | `@gluon.jit` | `@helion.kernel` | `@cute.kernel`（设备端），外面套一个 `@cute.jit` 的 host 函数 |
| 执行单位 | thread block（CTA） | program（编译成一个 CTA） | program（同 Triton） | 编译器：每个顶层 `hl.tile` 循环的一次迭代对应一个 program | thread block |
| 执行单位的编号 | `blockIdx.x/y/z` | `tl.program_id(axis)` | `gl.program_id(axis)` | 编译器（tile 对象自带 `begin`、`end`、`block_size`） | `cute.arch.block_idx()` |
| 执行单位的个数 | `gridDim` | `tl.num_programs(axis)` | `gl.num_programs(axis)` | 编译器 | `cute.arch.grid_dim()` |
| 线程编号 | `threadIdx` | 没有 | 没有 | 没有 | `cute.arch.thread_idx()` |
| 每个执行单位的线程数 | `<<<grid, threads>>>` 的 `threads` | 启动参数 `num_warps`（× 32） | 启动参数 `num_warps` | `helion.Config(num_warps=…)` 或交给 autotune | `.launch(grid=…, block=…)` 的 `block` |
| 启动 | `kernel<<<grid, block>>>(…)` | `kernel[grid](…)` | `kernel[grid](…)` | 像普通函数一样调用 `kernel(x, y)`，grid 由 Helion 算 | 在 `@cute.jit` 函数里 `kernel(…).launch(grid=…, block=…)` |
| 编译时常量 | 模板参数 | `tl.constexpr` | `gl.constexpr` | `hl.constexpr`；tile 大小用 `hl.register_block_size` | `cutlass.Constexpr` |

## 2. 数据与内存

| 概念 | CUDA | Triton | Gluon | Helion | CuTe DSL |
| --- | --- | --- | --- | --- | --- |
| 一块数据 | 没有；每个线程自己算下标 | `tl.arange(0, BLOCK)` 构造的块，`BLOCK` 必须是 2 的幂 | `gl.arange(0, BLOCK, layout=…)`，**必须给 layout** | `for tile in hl.tile(n): … x[tile] …` | `cute.zipped_divide`、`cute.local_tile` 把张量切成 tile |
| 元素怎么分给线程（layout） | 你写的下标公式本身 | 编译器（TTGIR 里的 `#ttg.blocked` 等，第 3、20 章） | 你写：`gl.BlockedLayout(size_per_thread, threads_per_warp, warps_per_cta, order)` 等 | 编译器（生成 Triton 以后再由 Triton 推导） | 你写：`cute.make_layout`、`cute.make_layout_tv` |
| 读写全局内存 | 指针加下标 | `tl.load` / `tl.store`，用 `mask`、`other` 处理边界 | `gl.load` / `gl.store` | 张量下标 `x[tile]`，边界的 mask 由 Helion 生成 | `cute.copy` |
| 结构化的访存描述 | — | `tl.make_tensor_descriptor`（第 8 章；旧的 `tl.make_block_ptr` 在 3.9.0 里已经移除） | — | 编译器（`Config(indexing=…)` 可以选） | — |
| shared memory | `__shared__` | 编译器 | `gl.allocate_shared_memory(element_ty, shape, layout)` | 编译器 | `cutlass.utils.SmemAllocator`，再 `allocate_tensor` |
| 换 layout | 你自己经过 shared memory 搬 | 编译器插入 `ttg.convert_layout`（第 21 章） | `gl.convert_layout(value, layout)` | 编译器 | — |
| 异步拷贝（全局 → shared） | `cp.async`、`cuda::memcpy_async` | 编译器（由 `num_stages` 触发，第 13 章） | `gluon.language.nvidia.ampere.async_copy` 的 `async_copy_global_to_shared`、`commit_group`、`wait_group` | 编译器（`Config(range_num_stages=…)`） | `cute.nvgpu.cpasync.CopyG2SOp`，配合 `cute.arch.cp_async_commit_group`、`cp_async_wait_group` |

## 3. 计算与同步

| 概念 | CUDA | Triton | Gluon | Helion | CuTe DSL |
| --- | --- | --- | --- | --- | --- |
| 矩阵乘（用上矩阵单元） | `mma` PTX 指令、WMMA，或者 CUTLASS | `tl.dot`（第 7 章） | 架构相关：Ampere 是 `gluon.language.nvidia.ampere.mma_v2`；另有不走矩阵单元的 `gl.dot_fma` | `hl.dot`，或者在 tile 上直接写 `torch.matmul` / `@` | `cute.make_tiled_mma` + `cute.gemm` |
| 归约 | warp shuffle + shared memory，自己写 | `tl.sum`、`tl.max`、`tl.reduce`（编译器生成 shuffle 和 shared memory，第 3 章） | `gl.sum`、`gl.max`、`gl.reduce` | 在 tile 上写 PyTorch 运算，如 `torch.sum(x[tile, :], dim=-1)` | — |
| 扫描 | 自己写 | `tl.cumsum`、`tl.associative_scan` | `gl.cumsum`、`gl.associative_scan` | `hl.associative_scan`、`hl.cumsum` | — |
| 原子操作 | `atomicAdd` 等 | `tl.atomic_add` 等 | `gl.atomic_add` 等 | `hl.atomic_add` 等 | — |
| 执行单位内的同步 | `__syncthreads()` | 编译器；调试用 `tl.debug_barrier()` | `gl.barrier()` | 编译器 | `cutlass.pipeline.NamedBarrier` 等 |
| 全局同步 | 协作组（cooperative groups）的 grid 同步 | 没有（只有原子操作） | 没有 | `hl.barrier()`：**全 grid 范围**，只能写在两个顶层 `hl.tile` 循环之间，和 `__syncthreads` 不是一回事 | — |
| warp 分工（warp specialization） | 自己写 | 编译器（Hopper 及以上，第 15 章） | `gl.warp_specialize` | `Config(range_warp_specializes=…)` | — |

## 4. 调参、调试与看 IR

| 概念 | CUDA | Triton | Gluon | Helion | CuTe DSL |
| --- | --- | --- | --- | --- | --- |
| 自动调优 | 没有内置 | `@triton.autotune(configs, key)`（第 14 章） | — | **默认就做**：第一次调用时搜索 `Config`；也可以固定：`@helion.kernel(config=helion.Config(…))` | — |
| 调优空间里有什么 | — | 你列出的 `triton.Config`（块大小、`num_warps`、`num_stages`） | — | 由 Helion 生成：`block_sizes`、`loop_orders`、`l2_groupings`、`num_warps`、`num_stages`、`pid_type`、`indexing` 等 | — |
| kernel 里打印 | `printf` | `tl.device_print`、`tl.static_print` | `gl.device_print`、`gl.static_print` | `hl.device_print` | `cute.printf` |
| kernel 里断言 | `assert` | `tl.device_assert`、`tl.static_assert` | `gl.device_assert`、`gl.static_assert` | — | — |
| 不用 GPU 跑 | — | `TRITON_INTERPRET=1`（第 9 章） | — | — | — |
| 看生成的代码 | `nvcc -ptx`、`cuobjdump -sass` | 编译产物的 `asm` 字典，或者缓存目录（第 4、19 章） | 同 Triton | 打印它生成的 Triton 代码（第 26 章） | — |

## 5. 硬件术语

GPU 厂商各有各的叫法。本书正文统一用 NVIDIA 的说法，第一次出现时给出 Intel 的对应词（第 2 章）。

| 本书的说法 | NVIDIA | Intel（Xe-HPC） | AMD（CDNA） | Triton 里对应的东西 |
| --- | --- | --- | --- | --- |
| 计算单元 | SM（streaming multiprocessor） | Xe-core | CU（compute unit） | 一个 program 跑在一个计算单元上 |
| warp | warp（32 个线程） | sub-group（16 或 32 个 work-item） | wavefront（64 个线程） | `num_warps`；TTGIR 里的 `"ttg.threads-per-warp"` |
| 线程 | thread | work-item | thread（lane） | 看不见 |
| 线程块 | thread block / CTA | work-group | workgroup | program |
| shared memory | shared memory | SLM（shared local memory） | LDS（local data share） | 编译器分配；TTGIR 里的 `#ttg.shared` 一类 layout |
| 矩阵单元 | Tensor Core | XMX | Matrix Core | `tl.dot`；TTGIR 里的 `#ttg.nvidia_mma`、`#ttig.dpas`、`#ttg.amd_mfma` |
| 显存 | HBM / GDDR | HBM | HBM | 指针指向的地方 |

## 6. 缩写

| 缩写 | 全称 | 第一次出现 |
| --- | --- | --- |
| CTA | cooperative thread array，即 thread block | 第 2 章 |
| SIMT | single instruction, multiple threads | 第 2 章 |
| SM | streaming multiprocessor | 第 2 章 |
| SLM | shared local memory（Intel） | 第 2 章 |
| HBM | high bandwidth memory | 第 1 章 |
| TTIR | Triton IR：`tt` 方言，硬件无关 | 第 1 章 |
| TTGIR | TritonGPU IR：`ttg` 方言，带 layout | 第 1 章 |
| PTX | NVIDIA 的虚拟指令集 | 第 3 章 |
| SPIR-V | Khronos 的中间表示，Intel 后端的输出 | 第 4 章 |
| MMA | matrix multiply-accumulate，矩阵乘加 | 第 7 章 |
| DPAS | dot product accumulate systolic，Intel 的矩阵指令 | 第 7 章 |
| XMX | Xe matrix extensions，Intel 的矩阵单元 | 第 2 章 |
| TMA | tensor memory accelerator，Hopper 起的异步拷贝单元 | 第 15 章 |
| CE | Compiler Explorer（godbolt.org） | 第 1 章 |

---

**这份附录的要点**

- 从左往右，CUDA → CuTe DSL → Gluon → Triton → Helion，**你要写的越来越少，编译器替你做的越来越多**。线程编号、layout、shared memory、warp 分工这几行最能看出这个梯度。
- Triton 和 Gluon 共用同一套编译器后端：Gluon 的每一个「你写」，在 Triton 里都是「编译器」，并且能在 Triton 的 TTGIR 里看到编译器做了什么决定（第 20–22 章）。
- Helion 最后生成的是 Triton 代码。它的调优空间（`Config` 的那些字段）比 `@triton.autotune` 大得多（第 26 章）。
- 硬件术语的对照：SM ↔ Xe-core ↔ CU，warp ↔ sub-group ↔ wavefront，shared memory ↔ SLM ↔ LDS，Tensor Core ↔ XMX ↔ Matrix Core。
