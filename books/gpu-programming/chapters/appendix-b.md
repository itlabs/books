# 附录 B 术语对照表：CUDA ↔ SYCL ↔ OpenCL ↔ Triton

## 这一章想让你带走什么

<div class="goal-box">
<p>正文里反复承诺过的那张"罗塞塔石碑"就在这里。四种技术——<strong>CUDA、SYCL、OpenCL、Triton</strong>——本质上在描述同一套 GPU 执行模型，只是换了名字和外衣。这一节把等价概念<strong>横向对齐成表</strong>，让你读任何一份文档、任何一段别人的代码时，都能立刻在脑子里翻译成你熟悉的那套词。</p>
<p>表格之外，我们还挑出几个<strong>最容易咬人的"假朋友"</strong>——尤其是 SYCL 的 nd_range 尺寸约定和 CUDA 的 <code>&lt;&lt;&lt;&gt;&gt;&gt;</code> 完全相反这件事（第 5 章的经典困惑），专门讲透。</p>
</div>

## 怎么用这份表

四种技术的心智模型高度同构，但**词汇和默认约定不同**，混用时最容易出错的不是概念本身，而是"同一个词在两边指的东西不一样"或"同一个意思两边写法相反"。下面按主题分组给出对照表，每组后面跟一句要点。查表时记住一条主线索：

<div class="keypoint">
<strong>贯穿全表的核心对应</strong>
<strong>CUDA block = SYCL work-group = OpenCL work-group ≈ 一个 Triton program 处理的数据块</strong>；<strong>CUDA thread = SYCL work-item = OpenCL work-item</strong>。而 Triton 特殊：它<strong>没有暴露"单个线程"</strong>这一层——你写的一个 Triton program 直接操作一整块数据（一个向量/张量块），线程级并行由编译器在底层生成。所以把 Triton 的 program 类比成 CUDA 的 <strong>block</strong>，而不是 thread，是读懂 Triton 的第一把钥匙。
</div>

## 一、执行层级

| 概念 | CUDA | SYCL | OpenCL | Triton |
|---|---|---|---|---|
| 整个启动的线程集合 | grid | NDRange | NDRange | 一维/多维的 program 网格（grid） |
| 中层分组（映射到一个 SM） | block（thread block） | work-group | work-group | 一个 **program**（处理一块数据） |
| 最小执行个体 | thread | work-item | work-item | —（不暴露单线程，编译器生成） |
| 维度 | 1D/2D/3D | 1D/2D/3D | 1D/2D/3D | 1D/2D/3D 的 program id |

要点：CUDA/SYCL/OpenCL 都是**三层**（grid/网格 → block/组 → thread/项）。Triton 砍掉了最下面的 thread 层，让你只在**块级**思考——一个 program 就是"我负责这一块数据，我对它整块做 load/compute/store"。

## 二、索引：我是谁、我在哪

| 想问 | CUDA | SYCL（`nd_item` it） | OpenCL | Triton |
|---|---|---|---|---|
| 我在哪个 block/组？ | `blockIdx.x` | `it.get_group(0)` | `get_group_id(0)` | `tl.program_id(0)` |
| block/组内我是第几个？ | `threadIdx.x` | `it.get_local_id(0)` | `get_local_id(0)` | —（无线程内索引） |
| 全局唯一编号 | `blockIdx.x*blockDim.x+threadIdx.x` | `it.get_global_id(0)` | `get_global_id(0)` | `pid*BLOCK + tl.arange(0,BLOCK)`（一整块偏移） |
| block/组大小 | `blockDim.x` | `it.get_local_range(0)` | `get_local_size(0)` | `BLOCK`（`tl.constexpr` 常量） |
| grid 大小（组数） | `gridDim.x` | `it.get_group_range(0)` | `get_num_groups(0)` | 启动时 grid 的形状 |

要点：SYCL 有一个便利处——`get_global_id()` 直接给你全局编号，省掉 CUDA 里 `blockIdx*blockDim+threadIdx` 的手算。Triton 则完全不同：它没有"我是第几个线程"，你拿到 `program_id` 后用 `tl.arange` 生成**一整块**下标向量，一次处理整块。

## 三、调度单位（硬件真正齐步走的那一组）

| 概念 | CUDA | SYCL | OpenCL | Triton |
|---|---|---|---|---|
| SIMT 调度单位 | **warp**（32 线程，NVIDIA） | **sub-group** | **sub-group** | 由编译器处理（不直接暴露） |
| 组内 lane 编号 | `laneid`（0–31） | `sg.get_local_id()` | `get_sub_group_local_id()` | — |
| 组内广播/洗牌 | `__shfl_sync` 系 | `sycl::select_from_group` 等 | `sub_group_shuffle` | 由编译器/`tl` 原语间接实现 |

要点：warp（第 6 章）是 NVIDIA 的 32 线程，SYCL/OpenCL 的对应物叫 **sub-group**，但其大小依设备而定，不一定是 32——写可移植代码时要查询而非假设。Triton 不让你直接碰这一层，warp 级的优化交给它的编译器。

## 四、内存空间

| 语义 | CUDA | SYCL | OpenCL | Triton |
|---|---|---|---|---|
| 大而慢的显存 | global memory | global | `__global` | 指针（`tl.load/tl.store` 访问 DRAM） |
| block/组内共享的片上高速内存 | **shared memory**（`__shared__`） | **local**（`local_accessor`） | `__local` | 由编译器管理（`tl` 抽象掉） |
| 每线程私有 | registers / local memory | private | `__private` | 块内张量（存于寄存器/共享，编译器定） |
| 只读常量 | constant（`__constant__`） | constant | `__constant` | 编译期常量（`tl.constexpr`） |

<div class="warn">
<strong>假朋友警报：CUDA 的 "local" ≠ SYCL/OpenCL 的 "local"</strong>
在 <strong>CUDA</strong> 里，"local memory" 指的是<strong>每个线程私有</strong>、寄存器放不下时溢出到显存的那块空间（慢）。而在 <strong>SYCL/OpenCL</strong> 里，"local" 指的是<strong>整个 work-group 共享的片上高速内存</strong>——也就是 CUDA 说的 <strong>shared memory</strong>！这是四种技术间最坑的一个术语撞车：同一个词 "local"，在 CUDA 是"线程私有的慢内存"，在 SYCL/OpenCL 是"组内共享的快内存"。读代码时务必先确认是谁的方言。
</div>

要点：记住这条映射链——CUDA **shared** = SYCL/OpenCL **local**；CUDA **local**（私有慢内存）在 SYCL/OpenCL 里归入 **private**（寄存器溢出）。Triton 把这一整层都藏了起来，你只管声明块级张量，它决定放寄存器还是共享内存。

## 五、函数/内核限定符

| 用途 | CUDA | SYCL | OpenCL | Triton |
|---|---|---|---|---|
| 标记"这是一个 kernel" | `__global__` | 传给 `parallel_for` 的 lambda / functor | `__kernel` | `@triton.jit` |
| device 端可调用的辅助函数 | `__device__` | 普通函数（在 device 代码里调用即可） | 普通函数（OpenCL C 内） | `@triton.jit`（可被其他 jit 函数调用） |
| host 端函数 | `__host__`（默认） | 普通 C++ | host 用 C/C++ API | 普通 Python |
| 声明共享内存 | `__shared__ float s[N];` | `sycl::local_accessor<float>` | `__local float s[N];` | 编译器管理（无显式声明） |

要点：CUDA 用**函数属性**区分 host/device；SYCL 走**现代 C++**路线，kernel 就是一个传给 `parallel_for` 的 lambda，没有特殊关键字；OpenCL 用 `__kernel`；Triton 用装饰器 `@triton.jit`，写的是 Python 但被编译成 GPU 代码。

## 六、同步

| 操作 | CUDA | SYCL（`nd_item` it） | OpenCL | Triton |
|---|---|---|---|---|
| block/组内 barrier | `__syncthreads()` | `it.barrier()` | `barrier(CLK_LOCAL_MEM_FENCE)` | program 内隐式（无显式 barrier） |
| 内存栅栏 | `__threadfence()` 系 | `sycl::atomic_fence` | `mem_fence(...)` | 由编译器保证 |
| warp/sub-group 内同步 | `__syncwarp()` | sub-group barrier | sub-group barrier | 编译器处理 |

<div class="warn">
<strong>Triton 没有 <code>__syncthreads()</code></strong>
因为一个 Triton program 在语义上是"对一整块数据做操作"，块内的线程级同步由编译器在生成代码时自动插入——你在 Triton 里<strong>看不到也写不出</strong> barrier。这既是省心（不会漏同步），也意味着你放弃了 CUDA 那种手动编排 shared memory + barrier 的细粒度控制（对比第 8、11 章）。
</div>

要点：CUDA/SYCL/OpenCL 都要你**显式**在组内设 barrier 来协调共享内存读写（第 8 章）；漏了或放错位置就是数据竞争。Triton 把这层隐式化了。

## 七、原子操作

| 操作 | CUDA | SYCL | OpenCL | Triton |
|---|---|---|---|---|
| 原子加 | `atomicAdd(&x, v)` | `sycl::atomic_ref<...>{x} += v` | `atomic_add(&x, v)` | `tl.atomic_add(ptr, v)` |
| 原子 CAS | `atomicCAS(...)` | `atomic_ref::compare_exchange_*` | `atomic_cmpxchg(...)` | `tl.atomic_cas(...)` |
| 原子 max/min 等 | `atomicMax/Min/...` | `atomic_ref::fetch_max/min` | `atomic_max/min` | `tl.atomic_max/min` |

要点：语义一致（对全局/共享内存做不可分割的读改写），只是 SYCL 走的是 C++ 的 `atomic_ref` 风格，其余三者是函数调用风格。

## 八、内核启动语法

| | CUDA | SYCL | OpenCL | Triton |
|---|---|---|---|---|
| 启动写法 | `kernel<<<grid, block>>>(args)` | `q.parallel_for(nd_range<1>{{global},{local}}, kernel)` | `clEnqueueNDRangeKernel(q, k, ..., global, local, ...)` | `kernel[grid](args, BLOCK=...)` |
| 第一个尺寸参数 | **block 的数量**（grid 大小） | **全局总工作量**（global size） | **全局总工作量**（global size） | grid：**program 的数量** |
| 第二个尺寸参数 | 每个 block 的**线程数** | 每个 work-group 的大小（local size） | 每个 work-group 的大小（local size） | `BLOCK`：每个 program 处理的元素数（作为 `constexpr`） |

<div class="warn">
<strong>头号经典困惑（第 5 章）：CUDA 的 <code>&lt;&lt;&lt;&gt;&gt;&gt;</code> 和 SYCL 的 nd_range 尺寸约定<em>相反</em></strong>
<p>CUDA：<code>kernel&lt;&lt;&lt;num_blocks, block_size&gt;&gt;&gt;</code> 的第一个数是 <strong>block 的个数</strong>，总线程数 = num_blocks × block_size，要你自己乘出来。</p>
<p>SYCL：<code>nd_range&lt;1&gt;{global, local}</code> 的第一个数 <code>global</code> 是<strong>线程总数</strong>（不是组数！），<code>local</code> 是每组大小，组数 = global ÷ local，由运行时替你除。</p>
<p>于是同样想要"256 个组、每组 128 线程"（共 32768 个线程），两边写法是：CUDA 写 <code>&lt;&lt;&lt;256, 128&gt;&gt;&gt;</code>；SYCL 写 <code>nd_range&lt;1&gt;{ {32768}, {128} }</code>。一个填"组数"，一个填"总数"——把 SYCL 的 <code>global</code> 当成组数（填 256）是最常见的错，会导致只处理了 1/128 的数据。<strong>记法：CUDA 说"几个组、每组几人"，SYCL 说"一共几人、每组几人"。</strong></p>
</div>

对照代码，一眼看清这个差别：

<div class="compare">
<div class="known">
<div class="compare-head">CUDA：第一个数 = 组数</div>

```cuda-norun
// 想要 32768 个线程，每组 128
int block = 128;
int grid  = 32768 / 128;   // = 256 个 block
kernel<<<grid, block>>>(...);
// 第一个参数填"组数"256
```

</div>
<div class="cpp-side">
<div class="compare-head">SYCL：第一个数 = 总数</div>

```sycl-norun
// 想要 32768 个线程，每组 128
sycl::nd_range<1> ndr{ sycl::range<1>(32768),   // 全局总数！
                       sycl::range<1>(128) };   // 每组大小
q.parallel_for(ndr, [=](sycl::nd_item<1> it){ ... });
// 第一个参数填"总数"32768，不是 256
```

</div>
</div>

## 九、编译器 / 工具链

| | CUDA | SYCL | OpenCL | Triton |
|---|---|---|---|---|
| 编译方式 | AOT（提前编译，`nvcc`） | AOT 或 JIT（`icpx -fsycl` / AdaptiveCpp `acpp`） | 运行时 JIT（把 kernel 源码字符串编译）或 SPIR-V | JIT（首次调用时即时编译） |
| 中间表示 | PTX → SASS | SPIR-V / PTX（视后端） | SPIR-V / 厂商 IR | 经 Triton IR → LLVM → PTX |
| 主要语言 | C++（扩展） | 标准 C++ | C（OpenCL C）/ C++ | Python |
| 看编译产物 | Compiler Explorer（nvcc）看 PTX | Compiler Explorer（icpx）看 SPIR-V/PTX | 各厂商工具 | 打印 `.asm` 属性看 PTX |

要点：CUDA 是提前编译，Triton 是**运行时即时编译**（第一次调用某个形状时才编译，之后缓存），这也是 Triton 里首次调用偏慢、之后飞快的原因。SYCL 两种模式都支持。

## 一页速查（打印贴墙版）

| 主题 | CUDA | SYCL | OpenCL | Triton |
|---|---|---|---|---|
| 组 | block | work-group | work-group | program（≈ block） |
| 项 | thread | work-item | work-item | （不暴露） |
| 组索引 | `blockIdx` | `get_group` | `get_group_id` | `program_id` |
| 项索引 | `threadIdx` | `get_local_id` | `get_local_id` | — |
| 调度单位 | warp(32) | sub-group | sub-group | 编译器 |
| 共享快内存 | `__shared__` | local（`local_accessor`） | `__local` | 编译器管理 |
| 私有慢内存 | local | private | `__private` | — |
| kernel 标记 | `__global__` | lambda | `__kernel` | `@triton.jit` |
| 组内同步 | `__syncthreads()` | `it.barrier()` | `barrier(...)` | 隐式 |
| 启动第一参数 | **组数** | **总线程数** | **总线程数** | **program 数** |

---

**这一章的要点**

- 四种技术描述的是**同一套模型**：核心对应是 **CUDA block = SYCL/OpenCL work-group ≈ Triton program**，**CUDA thread = SYCL/OpenCL work-item**；Triton **不暴露单线程**，一个 program 直接处理一整块数据。
- 最坑的假朋友：**"local" 一词**在 CUDA 指线程私有的慢内存，在 SYCL/OpenCL 却指组内共享的快内存（= CUDA 的 shared）。
- 最坑的语法差：**CUDA `<<<num_blocks, block_size>>>` 的第一个数是"组数"，SYCL `nd_range{global, local}` 的第一个数是"线程总数"**——填错会只算一小部分数据（第 5 章经典坑）。
- warp（NVIDIA 32 线程）在 SYCL/OpenCL 里叫 **sub-group**，大小不保证是 32；Triton 把 warp 级和 barrier 都交给编译器。
- Triton 是**运行时 JIT**，CUDA 是 AOT——这解释了 Triton 首次调用慢、之后快的现象。
