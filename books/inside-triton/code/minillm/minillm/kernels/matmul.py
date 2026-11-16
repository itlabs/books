"""矩阵乘 C = A @ B（第 7 章：朴素的 tile 版本；第 8 章：tensor descriptor 版本）。

每个 program 算 C 的一个 BLOCK_M × BLOCK_N 的 tile：沿 K 方向循环，每次读 A 的一条 BLOCK_M × BLOCK_K、
B 的一条 BLOCK_K × BLOCK_N，用 tl.dot 乘起来，累加进一个 fp32 的累加器。
第 8 章换成 tensor descriptor，第 12–15 章逐步优化（执行顺序、流水、autotune……）。
"""
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from ..device import DEVICE


@triton.jit
def _matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                   stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                   ACC_DTYPE: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    # 指向 A 的第一条（BLOCK_M × BLOCK_K）和 B 的第一条（BLOCK_K × BLOCK_N）
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    for k in range(0, K, BLOCK_K):
        # 三个方向都可能不是 BLOCK 的整数倍：越界的位置填 0，乘出来也是 0，不影响累加
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=ACC_DTYPE)
        a_ptrs += BLOCK_K * stride_ak             # 沿 K 往前走一条
        b_ptrs += BLOCK_K * stride_bk
    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul(a: torch.Tensor, b: torch.Tensor, BLOCK_M: int = 128, BLOCK_N: int = 128, BLOCK_K: int = 32,
           num_warps: int = 8, acc_dtype=tl.float32) -> torch.Tensor:
    assert a.dim() == 2 and b.dim() == 2 and a.shape[1] == b.shape[0], f"形状不对：{a.shape} @ {b.shape}"
    assert a.dtype == b.dtype and a.device == b.device
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_kernel[grid](a, b, c, M, N, K, *a.stride(), *b.stride(), *c.stride(),
                         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, ACC_DTYPE=acc_dtype,
                         num_warps=num_warps)
    return c


# ---------------------------------------------------------------- 第 8 章 ----
# tensor descriptor 版本：不再逐个元素地算地址，而是告诉编译器「这是一个 M×K 的矩阵，
# stride 是多少，我每次要读一个 BLOCK_M × BLOCK_K 的块」。越界的部分由 descriptor 自己处理
# （读的时候填 0，写的时候不写），kernel 里不用写 mask。
@triton.jit
def _matmul_desc_kernel(a_desc, b_desc, c_desc, K,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = a_desc.load([pid_m * BLOCK_M, k])           # 块左上角的坐标，单位是元素
        b = b_desc.load([k, pid_n * BLOCK_N])
        acc = tl.dot(a, b, acc)
    c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], acc.to(c_desc.dtype))


def descriptor_ok(*tensors) -> bool:
    """TensorDescriptor 的要求（python/triton/tools/tensor_descriptor.py）：
    基地址 16 字节对齐、除最后一维外的 stride 是 16 字节的倍数、最后一维连续。"""
    return all(t.data_ptr() % 16 == 0 and t.stride(-1) == 1 and
               all(s * t.element_size() % 16 == 0 for s in t.stride()[:-1]) for t in tensors)


def matmul_desc(a: torch.Tensor, b: torch.Tensor, BLOCK_M: int = 256, BLOCK_N: int = 256, BLOCK_K: int = 32,
                num_warps: int = 32) -> torch.Tensor:
    assert a.dim() == 2 and b.dim() == 2 and a.shape[1] == b.shape[0], f"形状不对：{a.shape} @ {b.shape}"
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)
    assert descriptor_ok(a, b, c), "不满足 tensor descriptor 的对齐要求，请用 matmul()"
    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(b, [BLOCK_K, BLOCK_N])
    c_desc = TensorDescriptor.from_tensor(c, [BLOCK_M, BLOCK_N])
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_desc_kernel[grid](a_desc, b_desc, c_desc, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                              num_warps=num_warps)
    return c


def prefer_descriptor() -> bool:
    """哪些平台默认走 descriptor（第 8 章的实测与分析）：
    - Intel GPU：descriptor 能编成二维块读写（2D block IO），Max 1100 上快约 10 倍；
    - NVIDIA Hopper（sm_90）及以上：能编成 TMA；
    - 其它（包括 A10 这样的 Ampere）：host 端 descriptor 在 sm_86 上编成普通的读取再写 shared memory，
      没有用上异步拷贝，未必比指针版快，所以默认还是走指针版。"""
    if DEVICE.type == "xpu":
        return True
    if DEVICE.type == "cuda" and torch.version.cuda is not None:
        return torch.cuda.get_device_capability() >= (9, 0)
    return False


def mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """minillm 默认使用的矩阵乘：能用 descriptor 就用，否则退回指针版。"""
    n = b.shape[1]
    out_ok = n * a.element_size() % 16 == 0       # 输出是新分配的连续张量，它的行 stride 就是 N
    k_ok = a.shape[1] * a.element_size() % 16 == 0  # matmul_grouped 还要求 K 对齐（第 15 章）
    if prefer_descriptor() and a.shape[0] and n and out_ok and k_ok and descriptor_ok(a, b):
        return matmul_grouped(a, b)                 # 第 12 章起：带 L2 swizzle 的版本（第 8 章是 matmul_desc）
    return matmul(a, b)


@triton.jit
def _matmul_desc_dev_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_bk, stride_cm,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """和 _matmul_desc_kernel 一样，只是 descriptor 在 kernel 里创建（第 8 章的两种创建方式）。
    第 10 章的 minillm.ops.mm 用它：在 torch 2.15.0 + triton-xpu 3.9.0 上实测，
    torch.compile 遇到 triton_op 里传 host 端 TensorDescriptor 的 kernel，会把这次调用整个删掉。"""
    a_desc = tl.make_tensor_descriptor(a_ptr, [M, K], [stride_am, 1], [BLOCK_M, BLOCK_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, [K, N], [stride_bk, 1], [BLOCK_K, BLOCK_N])
    c_desc = tl.make_tensor_descriptor(c_ptr, [M, N], [stride_cm, 1], [BLOCK_M, BLOCK_N])
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        acc = tl.dot(a_desc.load([pid_m * BLOCK_M, k]), b_desc.load([k, pid_n * BLOCK_N]), acc)
    c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], acc.to(c_desc.dtype))


# ---------------------------------------------------------------- 第 12 章 ----
@triton.jit
def _matmul_grouped_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_bk, stride_cm,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                           GROUP_M: tl.constexpr, HINT_ALIGN: tl.constexpr = False):
    """一维 grid，按 GROUP_M 分组安排执行顺序（L2 swizzle）。descriptor 在 kernel 里创建（第 10 章）。

    program 的编号 pid 按「GROUP_M 行 tile 为一组，组内一列一列地走」映射到 (pid_m, pid_n)：
    同一时刻在跑的 program 集中在少数几行 A 和几列 B 上，它们从 L2 里共享数据。"""
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    in_group = GROUP_M * num_n                       # 一组里有多少个 program
    first_m = (pid // in_group) * GROUP_M            # 这一组从第几行 tile 开始
    size_m = tl.minimum(num_m - first_m, GROUP_M)    # 最后一组可能不满 GROUP_M 行
    pid_m = first_m + (pid % in_group) % size_m
    pid_n = (pid % in_group) // size_m
    if HINT_ALIGN:
        # 调用方保证了 K、N 和各个行 stride 都是 16 字节的倍数，但其中有不能被 16 整除的整数（比如 N = 3000）。
        # Triton 只给能被 16 整除的整数参数自动打上 divisibility 标记，其余的编译器一无所知，证明不了
        # 行 pitch 对齐，Intel 后端就放弃 2D block IO，慢 6–7 倍（第 15 章）。tl.multiple_of 对标量不起作用；
        # 写成「先除再乘」，运行时值不变，编译器从「× ALIGN」看得出它能被 ALIGN 整除。
        # 只在需要时这么做：对本来就能被 16 整除的值，这样反而丢掉了「能被 16 整除」的信息，慢 4%–7%
        ALIGN: tl.constexpr = 128 // c_ptr.dtype.element_ty.primitive_bitwidth
        K = K // ALIGN * ALIGN
        N = N // ALIGN * ALIGN
        stride_am = stride_am // ALIGN * ALIGN
        stride_bk = stride_bk // ALIGN * ALIGN
        stride_cm = stride_cm // ALIGN * ALIGN
    a_desc = tl.make_tensor_descriptor(a_ptr, [M, K], [stride_am, 1], [BLOCK_M, BLOCK_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, [K, N], [stride_bk, 1], [BLOCK_K, BLOCK_N])
    c_desc = tl.make_tensor_descriptor(c_ptr, [M, N], [stride_cm, 1], [BLOCK_M, BLOCK_N])
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        acc = tl.dot(a_desc.load([pid_m * BLOCK_M, k]), b_desc.load([k, pid_n * BLOCK_N]), acc)
    c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], acc.to(c_desc.dtype))


def needs_align_hint(*ints) -> bool:
    """_matmul_grouped_kernel 的 HINT_ALIGN：有不能被 16 整除的尺寸或 stride 时才要。"""
    return any(x % 16 for x in ints)


def matmul_grouped(a: torch.Tensor, b: torch.Tensor, BLOCK_M: int = 256, BLOCK_N: int = 256, BLOCK_K: int = 32,
                   GROUP_M: int = 8, num_warps: int = 32) -> torch.Tensor:
    """第 12 章的默认配置：256×256×32、32 个 warp（每个线程 128 个累加器元素，见第 12 章的 tile 扫描），
    GROUP_M=8（L2 swizzle）。要求满足 descriptor 的对齐条件。"""
    assert a.dim() == 2 and b.dim() == 2 and a.shape[1] == b.shape[0], f"形状不对：{a.shape} @ {b.shape}"
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)
    assert descriptor_ok(a, b, c) and K * a.element_size() % 16 == 0, "不满足对齐要求，请用 matmul()"
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _matmul_grouped_kernel[grid](a, b, c, M, N, K, a.stride(0), b.stride(0), c.stride(0),
                                 BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
                                 HINT_ALIGN=needs_align_hint(N, K, a.stride(0), b.stride(0)),
                                 num_warps=num_warps)
    return c


# ---------------------------------------------------------------- 第 14 章 ----
_TUNE_SPACE = [(64, 64, 32, 4), (128, 128, 32, 4), (128, 128, 32, 8), (128, 256, 32, 8), (128, 256, 32, 16),
               (256, 128, 32, 16), (256, 256, 32, 16), (256, 256, 32, 32), (256, 256, 64, 32)]
_TUNE_CONFIGS = [triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8}, num_warps=nw)
                 for bm, bn, bk, nw in _TUNE_SPACE]


def _register_prune(configs, named_args, **kwargs):
    """第 12 章的寄存器模型：每个线程的累加器超过 128 个就会大量溢出，不必测。
    在本书的 Max 1100 上，这一步还有一个更实际的理由：那几个配置会让 GPU 驱动里的编译器崩溃，
    同一个进程里崩第二次，整个进程就会退出（第 14 章）。"""
    tpw = 16 if DEVICE.type == "xpu" else 32
    kept = [c for c in configs if c.kwargs["BLOCK_M"] * c.kwargs["BLOCK_N"] // (tpw * c.num_warps) <= 128]
    return kept or configs[:1]


@triton.autotune(configs=_TUNE_CONFIGS, key=["M_BUCKET", "N", "K"],
                 prune_configs_by={"early_config_prune": _register_prune})
@triton.jit
def _matmul_tuned_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_bk, stride_cm, M_BUCKET,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                         GROUP_M: tl.constexpr, HINT_ALIGN: tl.constexpr):
    """和 _matmul_grouped_kernel 一样，只是 BLOCK_* 和 num_warps 交给 autotune。
    M_BUCKET 只用在 autotune 的 key 里，kernel 本身不用它。"""
    # 在 @triton.jit 函数里可以直接调用另一个 @triton.jit 函数
    _matmul_grouped_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_bk, stride_cm,
                           BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, HINT_ALIGN)


def matmul_tuned(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """自动调优版。key 用的是 M 按 2 的幂分档以后的值：M 在同一档里变化时不会重新调优（第 14 章）。"""
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)
    assert descriptor_ok(a, b, c) and K * a.element_size() % 16 == 0, "不满足对齐要求，请用 matmul()"
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    _matmul_tuned_kernel[grid](a, b, c, M, N, K, a.stride(0), b.stride(0), c.stride(0), triton.next_power_of_2(M),
                               HINT_ALIGN=needs_align_hint(N, K, a.stride(0), b.stride(0)))
    return c


@triton.jit
def _matmul_persistent_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_bk, stride_cm,
                              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                              GROUP_M: tl.constexpr, NUM_PROGS: tl.constexpr, HINT_ALIGN: tl.constexpr):
    """persistent 版（第 15 章）：只启动 NUM_PROGS 个 program（通常 = SM / Xe core 的个数），
    每个 program 循环处理第 pid、pid + NUM_PROGS、pid + 2·NUM_PROGS……个 tile。tile 的顺序和
    _matmul_grouped_kernel 一样按 GROUP_M 分组。"""
    if HINT_ALIGN:                                   # 见 _matmul_grouped_kernel
        ALIGN: tl.constexpr = 128 // c_ptr.dtype.element_ty.primitive_bitwidth
        K = K // ALIGN * ALIGN
        N = N // ALIGN * ALIGN
        stride_am = stride_am // ALIGN * ALIGN
        stride_bk = stride_bk // ALIGN * ALIGN
        stride_cm = stride_cm // ALIGN * ALIGN
    a_desc = tl.make_tensor_descriptor(a_ptr, [M, K], [stride_am, 1], [BLOCK_M, BLOCK_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, [K, N], [stride_bk, 1], [BLOCK_K, BLOCK_N])
    c_desc = tl.make_tensor_descriptor(c_ptr, [M, N], [stride_cm, 1], [BLOCK_M, BLOCK_N])
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    in_group = GROUP_M * num_n
    for tile in range(tl.program_id(0), num_m * num_n, NUM_PROGS):
        first_m = (tile // in_group) * GROUP_M
        size_m = tl.minimum(num_m - first_m, GROUP_M)
        pid_m = first_m + (tile % in_group) % size_m
        pid_n = (tile % in_group) // size_m
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        # 用 i * BLOCK_K 而不是 for k in range(0, K, BLOCK_K)：嵌在外层循环里时，Intel 后端证明不了
        # 后者能被 2 整除，A 的读取会丢掉 2D block IO，慢 3 倍多（第 15 章）
        for i in range(0, tl.cdiv(K, BLOCK_K)):
            k = i * BLOCK_K
            acc = tl.dot(a_desc.load([pid_m * BLOCK_M, k]), b_desc.load([k, pid_n * BLOCK_N]), acc)
        c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], acc.to(c_desc.dtype))


def num_compute_units() -> int:
    """SM（NVIDIA）或 Xe core（Intel）的个数。"""
    return triton.runtime.driver.active.utils.get_device_properties(DEVICE.index or 0)["multiprocessor_count"]


def matmul_persistent(a: torch.Tensor, b: torch.Tensor, BLOCK_M: int = 128, BLOCK_N: int = 256, BLOCK_K: int = 32,
                      GROUP_M: int = 8, num_warps: int = 32, num_progs: int | None = None) -> torch.Tensor:
    """默认配置是第 15 章在 Max 1100 上扫出来的：128×256、32 个 warp、每个 Xe core 一个 program。"""
    assert a.dim() == 2 and b.dim() == 2 and a.shape[1] == b.shape[0], f"形状不对：{a.shape} @ {b.shape}"
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)
    assert descriptor_ok(a, b, c) and K * a.element_size() % 16 == 0, "不满足对齐要求，请用 matmul()"
    tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    progs = max(1, min(tiles, num_progs or num_compute_units()))
    _matmul_persistent_kernel[(progs,)](a, b, c, M, N, K, a.stride(0), b.stride(0), c.stride(0),
                                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
                                        NUM_PROGS=progs, HINT_ALIGN=needs_align_hint(N, K, a.stride(0), b.stride(0)),
                                        num_warps=num_warps)
    return c
