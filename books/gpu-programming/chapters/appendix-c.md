# 附录 C 读懂 GPU 报错与用好 profiler

## 这一章想让你带走什么

<div class="goal-box">
<p>GPU 的报错有一种特有的恶意：错误常常<strong>不在出错的那一行报出来</strong>，而是拖到下一次同步时才浮现，信息还含糊。这一节先教你把最常见的几条 CUDA 报错<strong>翻译成人话</strong>，再给你一个"永远检查错误"的宏和 <strong>compute-sanitizer</strong>，最后讲清两个 profiler——<strong>Nsight Systems</strong> 看全局、<strong>Nsight Compute</strong> 看单核——各自回答什么问题。</p>
<p>读完你会有一套可复制的排错与调优工作流：<strong>先测量，定位瓶颈，再动手优化</strong>，而不是凭感觉瞎改。</p>
</div>

## 常见 CUDA 报错，以及它们<em>真正</em>的意思

CUDA 的错误码字面往往很泛。下面几条占了新手报错的绝大多数，逐一拆开。

### "an illegal memory access was encountered"

这是最常见、也最让人抓狂的一条，几乎总是**越界访问**——线程算出的下标超出了数组范围，读/写到了不属于它的显存。典型成因：

- kernel 里漏了边界检查 `if (i < n)`，最后一个 block 的部分线程算出的 `i ≥ n`；
- 下标计算写错（把 `blockDim` 打成 `gridDim`，或行列写反）；
- 传进 kernel 的指针没 `cudaMalloc` 或已经 `cudaFree`。

<div class="warn">
<strong>异步陷阱：报错的行往往<em>不是</em>出错的行</strong>
kernel 启动是<strong>异步</strong>的：CPU 发出 <code>kernel&lt;&lt;&lt;&gt;&gt;&gt;</code> 后立刻返回去干别的，kernel 在 GPU 上慢慢跑。所以 kernel 内部的非法访问<strong>不会</strong>在启动那一行报出来，而是拖到下一次<strong>同步点</strong>（<code>cudaDeviceSynchronize()</code>、<code>cudaMemcpy()</code> 等）才被发现并报出。结果就是：报错指向一句看起来无辜的 <code>cudaMemcpy</code>，真正的凶手却是几行之前那个 kernel。<strong>不理解这一点，你会在错误的地方找错误。</strong>
</div>

破解异步陷阱的办法只有一个：**每一步 CUDA 调用都检查返回值**，并在 kernel 后主动同步。这就是几乎所有 CUDA 代码都会有的 CHECK 宏：

```cuda-norun
// 放在头文件里，所有 CUDA 调用都用它包起来
#include <cstdio>
#include <cstdlib>

#define CHECK(call)                                                      \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA 错误 %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

// kernel 没有返回值，单独检查启动错误 + 同步错误：
#define CHECK_KERNEL()                                                   \
    do {                                                                 \
        CHECK(cudaGetLastError());        /* 捕获启动配置错误（同步报出）*/\
        CHECK(cudaDeviceSynchronize());   /* 等 kernel 跑完，捕获运行期错误 */\
    } while (0)

// 用法：
//   CHECK(cudaMalloc(&d_a, bytes));
//   add<<<blocks, threads>>>(d_c, d_a, d_b, N);
//   CHECK_KERNEL();     // 越界访问会在这里、而不是下一个 memcpy 报出
```

`cudaGetLastError()` 抓的是**启动配置**类错误（下面 "invalid configuration"），`cudaDeviceSynchronize()` 之后再查一次，抓的是 kernel **运行期**错误（越界等）。两个都查，报错就落在离凶手最近的地方。

### "invalid configuration argument"

启动配置非法，几乎都出在 `<<<grid, block>>>` 的尺寸上：

- **每个 block 的线程数 > 1024**（当前架构的硬上限就是 1024，是 x×y×z 的乘积）；
- 某一维为 0（比如 `(N + threads - 1) / threads` 里 `threads` 意外为 0）；
- grid/block 维度超出设备允许的最大值。

对策：把 block 大小固定在 128/256/512 这类合理值，grid 用 `(N + block - 1) / block` 向上取整，别让任何一维为 0。

### "out of memory"

`cudaMalloc` 申请不到那么多显存。常见于：数据集比显存大、忘了 `cudaFree` 导致泄漏累积、或同一进程里别的框架（PyTorch 等）已经占满了显存。用 `nvidia-smi` 看当前占用；分批处理数据；确保每个 `cudaMalloc` 都有对应的 `cudaFree`。

### "too many resources requested for launch"

这条很有迷惑性——它**不是**内存不够，而是**寄存器或 shared memory 不够**。每个 SM 的寄存器堆和 shared memory 是固定的，要在驻留的所有线程间分。如果你的 kernel 每线程用了很多寄存器，又开了很大的 block，`每线程寄存器数 × block 线程数` 就可能超过一个 SM 能给的额度，于是启动失败。

对策（详见第 16 章 occupancy）：**减小 block 大小**、用 `nvcc --ptxas-options=-v` 查看每线程寄存器用量、必要时用 `__launch_bounds__` 限制寄存器、或减少 kernel 里的局部变量/循环展开。这正是第 16 章"寄存器压力 vs block 大小"权衡的现实表现。

## compute-sanitizer：显存与竞争的照妖镜

`compute-sanitizer`（旧称 `cuda-memcheck`）是 CUDA Toolkit 自带的动态检查工具，它把上面那些含糊的报错变成**精确到源码行**的诊断。它有几个子工具：

```bash
# memcheck（默认）：检测越界访问、非法地址、未初始化访问
compute-sanitizer ./myapp
# 加 --tool 明确指定：
compute-sanitizer --tool memcheck ./myapp

# racecheck：检测 shared memory 上的数据竞争（漏写 __syncthreads 的经典 bug）
compute-sanitizer --tool racecheck ./myapp

# synccheck：检测 barrier 用法错误（比如分歧的线程里调用 __syncthreads）
compute-sanitizer --tool synccheck ./myapp

# initcheck：检测读取了未初始化的全局内存
compute-sanitizer --tool initcheck ./myapp
```

<div class="keypoint">
<strong>遇到 "illegal memory access" 的标准动作</strong>
别盯着源码干瞪眼——直接 <code>compute-sanitizer ./myapp</code>。它会告诉你<strong>哪个 kernel、哪一行、哪个线程</strong>越了界，以及读还是写。配合上面的 CHECK 宏，越界 bug 通常几分钟就能定位。用 <code>nvcc -G -g</code> 编译带上调试信息，报告还能给出源码行号。
</div>

`racecheck` 尤其值得一提：shared memory 上"写了没同步就读"的数据竞争（第 8、11 章反复警告的坑）在 CPU 上无从检测，`racecheck` 是最直接的手段——它会指出两个冲突的访问和缺失的 barrier。

## 两个 profiler：Nsight Systems 看全局，Nsight Compute 看单核

代码正确之后是**性能**。NVIDIA 的现代 profiler 分两个，回答的问题层次不同，别用错：

<div class="concept">
<svg viewBox="0 0 560 250" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">
  <text x="280" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#2b6cb0">两个 profiler，两个层次</text>
  <!-- Nsight Systems: timeline -->
  <rect x="20" y="35" width="520" height="90" rx="8" fill="#f7fbff" stroke="#2b6cb0"/>
  <text x="35" y="54" font-size="12" font-weight="bold" fill="#2b6cb0">Nsight Systems（nsys）：整条时间线</text>
  <!-- timeline bars -->
  <rect x="35" y="64" width="120" height="16" rx="2" fill="#e06a3b"/><text x="95" y="76" text-anchor="middle" font-size="9" fill="#fff">H2D 拷贝</text>
  <rect x="160" y="64" width="70" height="16" rx="2" fill="#2b6cb0"/><text x="195" y="76" text-anchor="middle" font-size="9" fill="#fff">kernel</text>
  <rect x="235" y="64" width="110" height="16" rx="2" fill="#e06a3b"/><text x="290" y="76" text-anchor="middle" font-size="9" fill="#fff">D2H 拷贝</text>
  <rect x="350" y="64" width="60" height="16" rx="2" fill="#2b6cb0"/><text x="380" y="76" text-anchor="middle" font-size="9" fill="#fff">kernel</text>
  <text x="35" y="104" font-size="10" fill="#6b7280">看：是搬数据慢还是算慢？kernel 之间有空档吗？拷贝和计算重叠了吗？</text>
  <text x="35" y="118" font-size="10" fill="#6b7280">回答"整体瓶颈在哪一段"——先用它。</text>
  <!-- Nsight Compute: single kernel deep dive -->
  <rect x="20" y="135" width="520" height="100" rx="8" fill="#f0fdf4" stroke="#3f9b5c"/>
  <text x="35" y="154" font-size="12" font-weight="bold" fill="#2f7a45">Nsight Compute（ncu）：单个 kernel 解剖</text>
  <text x="45" y="174" font-size="10" fill="#2f7a45">• occupancy（占用率，第 16 章）</text>
  <text x="300" y="174" font-size="10" fill="#2f7a45">• 内存吞吐 / 是否合并访问（第 17 章）</text>
  <text x="45" y="192" font-size="10" fill="#2f7a45">• shared memory bank conflict（第 17 章）</text>
  <text x="300" y="192" font-size="10" fill="#2f7a45">• 计算 vs 访存 谁是瓶颈（Roofline）</text>
  <text x="45" y="212" font-size="10" fill="#2f7a45">• warp 停顿原因分解</text>
  <text x="35" y="228" font-size="10" fill="#6b7280">回答"这个具体 kernel 为什么慢"——定位到某 kernel 后再用它。</text>
</svg>
<div class="caption">图 C-1　先用 Nsight Systems 看整条时间线找到最耗时的那一段，再用 Nsight Compute 深挖那个具体 kernel 的微观指标</div>
</div>

### Nsight Systems（nsys）：这是传输受限还是计算受限？

`nsys` 给你一条**全应用的时间线**：CPU 在干什么、数据拷贝（H2D/D2H）花了多久、各个 kernel 什么时候跑、它们之间有没有空档、拷贝和计算有没有重叠。它回答的是**宏观**问题：

- 时间到底花在**搬数据**（PCIe 拷贝）还是**算**（kernel）上？如果一大半时间在拷贝，优化 kernel 内部毫无意义——该做的是减少拷贝、用 pinned memory 或让拷贝与计算重叠（第 3 章那个警告的量化版）。
- kernel 之间有大片空档吗？可能是 CPU 端拖后腿或同步过多。
- 多个 stream 的 kernel 真的**并行**了吗？

```bash
# 采集一次运行的时间线，生成报告文件
nsys profile -o report ./myapp
# 命令行看摘要（也可用 Nsight Systems GUI 打开 report.nsys-rep 看可视化时间线）
nsys stats report.nsys-rep
```

### Nsight Compute（ncu）：这个 kernel 为什么慢？

一旦 `nsys` 告诉你"就是这个 kernel 占了 80% 时间"，换 `ncu` 对**单个 kernel**做显微镜级剖析。它给出的正是本书第 16、17 章讲过的那些指标：

- **occupancy（占用率）**：实际驻留的 warp 数 / 硬件上限，太低说明并行度不够掩盖延迟（第 16 章）；
- **memory throughput**：实测显存带宽利用率，配合 Roofline 判断是不是 memory-bound；
- **是否合并访问（coalescing）**：访存效率、每次事务搬了多少有用字节（第 17 章）；
- **shared memory bank conflict**：共享内存访问是否撞 bank 而被串行化（第 17 章）；
- **warp 停顿原因**：warp 大多卡在等内存、等依赖、还是等 barrier。

```bash
# 剖析全部 kernel（会显著拖慢运行，属正常）
ncu -o report ./myapp
# 只剖析感兴趣的 kernel，跑指定次数，收集完整指标集
ncu --set full -k add -c 1 ./myapp
# 直接在终端打印，不生成文件
ncu ./myapp
```

<div class="note">
<strong>为什么 profiler 跑起来那么慢</strong>
<code>ncu</code> 为了采集精确指标会<strong>重放</strong> kernel 很多次、串行化执行，慢几十倍是正常的，别以为程序卡死了。所以用 <code>-k</code> / <code>-c</code> 限定只剖析你关心的那个 kernel、只跑一两次，而不是整个应用全测。
</div>

## 调优工作流：先测量，再动手

把上面的工具串成一条纪律。**不要凭直觉优化**——GPU 上的直觉经常是错的（第 3 章已经领教过），几乎每一次"我觉得这里慢"都该先被数据验证。

1. **先让它正确**：CHECK 宏 + `compute-sanitizer` 清掉越界和数据竞争。错误的快代码没有意义。
2. **测量，画 Roofline**：用 `nsys` 看整体——瓶颈在拷贝还是计算？再用 `ncu` 看目标 kernel 落在 Roofline 的哪个区（第 16 章）：memory-bound 还是 compute-bound？
3. **对症下药**：
   - **memory-bound** → 优化访存：合并访问、用 shared memory 复用、向量化加载、算子融合减少读写趟数（第 7、11、14、17 章）。绝大多数真实 kernel 在这里。
   - **compute-bound** → 减少指令、用更快的数学、上 Tensor Core（第 19 章）。
   - **occupancy 太低** → 调 block 大小、降寄存器压力（第 16 章）。
   - **拷贝占主导** → 减少 H2D/D2H、重叠传输与计算、别为小任务上 GPU（第 3 章）。
4. **再测一次**：改完重新剖析，用数据确认真的变快了，而不是自我感觉良好。优化是一个**测量→改→再测量**的闭环。

## 排错速查表（症状 → 病因 → 对策）

| 症状 / 报错 | 最可能的原因 | 怎么办（相关章节） |
|---|---|---|
| illegal memory access | 数组越界（漏边界检查 / 下标算错 / 野指针） | CHECK 宏 + `compute-sanitizer memcheck`；补 `if(i<n)`（第 9 章） |
| 报错指向无辜的 cudaMemcpy | 异步：真正出错的是前面的 kernel | 在 kernel 后 `CHECK_KERNEL()` 主动同步定位（本附录） |
| invalid configuration argument | block 线程数 > 1024 或某维为 0 | 把 block 设成 128/256，grid 向上取整（第 4 章） |
| too many resources requested | 寄存器/shared 超额，非显存问题 | 减小 block、`--ptxas-options=-v` 查寄存器、`__launch_bounds__`（第 16 章） |
| out of memory | 显存不够 / 泄漏 / 别的进程占用 | `nvidia-smi` 看占用、分批、配平 malloc/free |
| 结果偶尔错、时对时错 | shared memory 数据竞争（漏 barrier） | `compute-sanitizer racecheck`；补 `__syncthreads()`（第 8、11 章） |
| kernel 正确但很慢 | 未知瓶颈 | `nsys` 找热点 → `ncu` 看 occupancy/带宽/合并（第 16、17 章） |
| 换更强的卡也不快 | memory-bound，卡在带宽 | 优化访存、融合算子，别指望算力（第 3、16 章） |
| 大量时间在拷贝 | H2D/D2H 主导，PCIe 瓶颈 | 减少拷贝、pinned memory、重叠传输（第 3 章） |
| 访存吞吐远低于峰值 | 未合并访问 / bank conflict | `ncu` 查 coalescing 与 bank conflict（第 17 章） |

---

**这一章的要点**

- CUDA 报错要**翻译**：illegal memory access ≈ 越界；invalid configuration ≈ block > 1024 或维度非法；too many resources ≈ **寄存器/shared 不够**（非显存）；out of memory ≈ 真的显存不够。
- **异步陷阱**：kernel 错误拖到下次同步才报，会指向无辜的行——用 **CHECK 宏 + kernel 后主动同步**把报错拉回凶手附近。
- **compute-sanitizer** 是照妖镜：`memcheck` 抓越界、`racecheck` 抓 shared memory 数据竞争、`synccheck` 抓 barrier 误用。
- 两个 profiler 分工：**Nsight Systems（nsys）** 看整条时间线定位宏观瓶颈（拷贝 vs 计算）；**Nsight Compute（ncu）** 解剖单个 kernel 的 occupancy、带宽、合并访问、bank conflict（对应第 16、17 章）。
- 工作流铁律：**先正确，再测量画 Roofline 找瓶颈，最后对症优化，改完再测**——不要凭直觉优化。
