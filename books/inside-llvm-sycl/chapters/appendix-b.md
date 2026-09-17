# 附录 B 术语与缩写对照：LLVM / SYCL / SPIR-V / UR

## 这份附录怎么用

<div class="goal-box">
<p>编译器世界的缩写密度极高——一句话里可能同时出现 IR、SSA、SPIR-V、UR、USM。这份附录把全书出现过的术语按四大领域（LLVM 基础设施 / SYCL 编程模型 / SPIR-V 与设备 / UR 与运行时）汇成一张对照表，末尾再单列一节 <strong>ClangIR</strong>（第五部分的词汇），每条给出<strong>英文全称 + 一句话中文解释 + 正文对应章节</strong>，方便你读到一半忘了某个词时回来查。</p>
<p>它是<strong>索引</strong>而非教程——想真正理解某个概念，跟着"正文章节"回去读那一章。术语解释力求"够用就走"，不追求 Khronos 规范级别的严谨。</p>
</div>

## LLVM 基础设施

| 术语 | 全称 | 一句话解释 | 正文 |
|---|---|---|---|
| **LLVM** | Low Level Virtual Machine（现已只是专有名） | 模块化的编译器基础设施集合：前端、优化、后端、工具链 | 第 1 章 |
| **IR** | Intermediate Representation | 编译器内部的中间表示；LLVM IR 是全书的核心语言 | 第 4 章 |
| **SSA** | Static Single Assignment | 每个变量只赋值一次的 IR 形式，便于优化分析 | 第 5 章 |
| **Module** | — | LLVM IR 的顶层容器，一个编译单元；含函数、全局量、metadata | 第 4 章 |
| **Pass** | — | 对 IR 做一次分析或变换的单元；串成 pipeline | 第 6 章 |
| **PassManager** | — | 组织与调度 Pass 的执行；新版为 New PM | 第 6 章 |
| **CFG** | Control Flow Graph | 基本块 + 跳转边构成的控制流图 | 第 5 章 |
| **DominatorTree** | — | 支配关系树，很多优化的分析基础 | 第 5 章 |
| **TableGen** | — | LLVM 的声明式代码生成 DSL；`.td` 文件描述、生成 C++ | 第 7、22 章 |
| **Attr.td** | — | clang 里用 TableGen 定义所有属性（attribute）的文件 | 第 22 章 |
| **CodeGen** | Code Generation | 广义指 IR → 目标代码；书中也指 clang 的 AST → LLVM IR 阶段 | 第 15、22 章 |
| **Sema** | Semantic Analysis | clang 的语义分析阶段；`SemaSYCL` 是其 SYCL 专用部分 | 第 22 章 |
| **AST** | Abstract Syntax Tree | clang 解析源码得到的抽象语法树 | 第 12 章 |
| **bitcode / `.bc`** | — | LLVM IR 的二进制序列化形式 | 第 4、19 章 |
| **`opt`** | — | 对 IR 单独跑 Pass 的命令行工具 | 第 6 章 |
| **`llvm-dis` / `llvm-as`** | — | bitcode ↔ 文本 IR 互转 | 第 4 章 |

## SYCL 编程模型

| 术语 | 全称 | 一句话解释 | 正文 |
|---|---|---|---|
| **SYCL** | （不是缩写，读 "sickle"） | Khronos 的单源异构 C++ 编程标准 | 第 12 章 |
| **DPC++** | Data Parallel C++ | Intel 基于 clang/LLVM 的 SYCL 实现（即 intel/llvm） | 第 12 章 |
| **单源** | single-source | 主机代码和设备代码写在同一份 `.cpp` 里 | 第 12 章 |
| **两趟编译** | host/device passes | `-fsycl` 下同一份源码分别按主机、设备编两趟 | 第 12 章 |
| **kernel** | — | 在设备上并行执行的函数体（lambda 或自由函数） | 第 12、21 章 |
| **queue** | — | 向某设备提交命令的入口 | 第 16 章 |
| **handler** | — | 在 `submit` 回调里配置并提交一个命令的对象 | 第 16 章 |
| **accessor** | — | 声明对 buffer/image 的访问，并让运行时追踪依赖 | 第 16、17 章 |
| **buffer** | — | 托管的数据容器，运行时管理主机↔设备的数据搬运 | 第 16 章 |
| **USM** | Unified Shared Memory | 指针式内存模型，替代 buffer/accessor；手动管理 | 第 16、19 章 |
| **nd_range / nd_item** | — | N 维执行范围 / 单个工作项在其中的坐标查询 | 第 21 章 |
| **kernel_bundle** | — | 一组处于某状态（input/object/executable）的 kernel 集合 | 第 16、23 章 |
| **kernel_id** | — | 标识一个 kernel 的句柄，用于在 bundle 里查找 | 第 16、21 章 |
| **集成头文件** | integration header | 设备编译趟生成、让主机端对上 kernel 名字/参数的头 | 第 14 章 |
| **名字修饰** | name mangling | 把 kernel（类型/函数）编码成稳定字符串名的规则 | 第 12、14 章 |
| **feature-test 宏** | — | `SYCL_EXT_*` 宏，编译期检测扩展是否可用、版本几何 | 第 20 章 |
| **扩展三位一体** | — | 扩展名 = 宏名 = 文件名（大小写差异） | 第 20 章 |
| **`__sycl_detail__`** | — | DPC++ 前端专用属性命名空间，把意图编码进 IR | 第 21、22 章 |
| **free function kernel** | — | 把 kernel 写成具名自由函数而非 lambda 的扩展 | 第 21 章 |
| **device_global** | — | 设备端的全局变量扩展（走通用 IR 属性路线） | 第 22 章 |
| **RTC** | Runtime Compilation | 运行时把源码字符串编成 kernel（kernel_compiler 扩展） | 第 23 章 |

## SPIR-V 与设备编译

| 术语 | 全称 | 一句话解释 | 正文 |
|---|---|---|---|
| **SPIR-V** | Standard Portable Intermediate Representation - V | Khronos 的可移植设备中间表示；SYCL 设备镜像的常见形态 | 第 9 章 |
| **`llvm-spirv`** | — | LLVM IR ↔ SPIR-V 双向翻译器（intel/llvm 顶层 `llvm-spirv/`） | 第 9 章 |
| **OpExtInst** | — | SPIR-V 里调用扩展指令集（如 OpenCL.std）的指令 | 第 9、19 章 |
| **OpenCL.std** | — | SPIR-V 的 OpenCL 扩展指令集，含 `sqrt`/`ceil` 等 | 第 19 章 |
| **`__spirv_ocl_*`** | — | 编译器声明的 SPIR-V/OpenCL 内建（如 `__spirv_ocl_sqrt`） | 第 19 章 |
| **libclc** | — | 提供 `__spirv_*` 内建原语实现的顶层库，尤为 NVPTX/AMDGPU | 第 11、19 章 |
| **libdevice** | — | 提供 `<cmath>`/`<cstring>`/assert/IMF 的设备端 fallback 的顶层库 | 第 19 章 |
| **weak 符号** | weak symbol | `__attribute__((weak))`；有强符号就用强的，否则用它兜底 | 第 19 章 |
| **fallback** | — | 保底实现；驱动原生支持时被丢弃、否则链入 | 第 19 章 |
| **IMF** | Intel Math Functions | Intel 特有的高性能/特定舍入数学函数集 | 第 19 章 |
| **sycl-post-link** | — | 设备 IR 链接后处理，消费 `sycl-*` 字符串属性、切分镜像 | 第 8、22 章 |
| **NVPTX / AMDGCN** | — | NVIDIA / AMD GPU 的 LLVM 后端目标三元组 | 第 11、19 章 |
| **Level Zero (L0)** | — | Intel 的低层 GPU 运行时 API（一种 UR 后端） | 第 18 章 |
| **sycl-jit** | — | 顶层子项目，进程内用 clang 库把 SYCL 源码 RTC 成镜像 | 第 23 章 |
| **ocloc** | Offline Compiler | Intel 的离线编译器（`libocloc.so`），编 OpenCL C | 第 23 章 |

## UR 与运行时

| 术语 | 全称 | 一句话解释 | 正文 |
|---|---|---|---|
| **UR** | Unified Runtime | SYCL 运行时与各后端之间的统一适配层 | 第 18 章 |
| **adapter** | — | UR 下对接具体后端（L0/OpenCL/CUDA/HIP）的插件 | 第 18 章 |
| **`ur*` API** | — | UR 的 C API 入口（`urQueueCreate` 等）；`*Exp` 为实验入口 | 第 18、23 章 |
| **backend** | — | SYCL 支持的执行后端（Level Zero / OpenCL / CUDA / HIP） | 第 18 章 |
| **program manager** | — | 运行时管理设备镜像、按 kernel 名查找、加载的组件 | 第 12、19、21 章 |
| **device image** | — | 设备端可加载的镜像（SPIR-V / 目标二进制 + 元信息） | 第 8、23 章 |
| **property set** | — | 设备镜像里的键值元信息（如 bfloat16 native/fallback 版本） | 第 8、19 章 |
| **command graph** | — | 运行时按 accessor 依赖排出的执行顺序图 | 第 17 章 |
| **`SYCL_UR_TRACE`** | — | 环境变量，打印运行时实际发出的 UR 调用 | 第 18 章 |
| **XPTI** | — | intel/llvm 的插桩/追踪框架（顶层 `xpti`/`xptifw`） | 第 18 章 |

## ClangIR（第五部分）

这一节的锚点是上游 `llvm/llvm-project`（以及孵化器 `llvm/clangir`），不是 `intel/llvm`。

| 术语 | 全称 | 一句话解释 | 正文 |
|---|---|---|---|
| **MLIR** | Multi-Level Intermediate Representation | LLVM 生态里造中间表示的基础设施；提供方言、操作、region 等积木 | 第 25 章 |
| **方言** | dialect | MLIR 里一组自成体系的操作/类型/属性；`cir` 就是一个方言 | 第 25 章 |
| **CIR / ClangIR** | Clang IR | Clang AST 与 LLVM IR 之间新增的一层 MLIR 方言（`name = "cir"`） | 第 25 章 |
| **OG / OGCG** | Original CodeGen | 社区对经典 `clang/lib/CodeGen/` 那条路径的叫法；`OGCG` 是 lit 测试里的 FileCheck 前缀 | 第 25 章 |
| **parity** | — | CIR 的验收线：产出与经典路径**语义等价**的 LLVM IR（不要求文本相同） | 第 25 章 |
| **`-fclangir`** | — | 打开 CIR 路径的 cc1 开关（默认关闭） | 第 25 章 |
| **`-emit-cir`** | — | 让编译停在 `cir` 方言，输出可读的 `.cir` 文本 | 第 25 章 |
| **`cir-opt` / `cir-translate`** | — | CIR 世界里对应 `opt` / `llc` 的工具；可对手写 `.cir` 单跑某个 pass | 第 25 章 |
| **Region** | — | MLIR 的一层嵌套结构（Operation → Region → Block → Operation），CIR 保留结构化控制流的本钱 | 第 26 章 |
| **`cir.yield` / `cir.condition`** | — | region 的终结子；把值/条件交回外层操作 | 第 26 章 |
| **`cir-flatten-cfg`** | — | 把 `cir.if`/`cir.for` 等结构化操作拍成 `cir.brcond` + 基本块的 pass | 第 26 章 |
| **`!cir.struct`** | — | CIR 的记录类型；有完整/不完整/匿名三态，成员带 `data`/`pad`/`empty`/`bitfield` 种类 | 第 26 章 |
| **`!cir.vptr`** | — | 虚表指针的专用类型，配 `cir.vtable.*` 四个具名操作使用 | 第 26 章 |
| **`!cir.eh_token` / `!cir.catch_token`** | — | 带类型的异常 token，取代 LLVM IR 里无类型的 landingpad 值 | 第 26 章 |
| **`cir-cxxabi-lowering`** | — | 把成员指针（`!cir.data_member` / `!cir.method`）等 ABI 决策落地的 pass | 第 26 章 |
| **`cir-target-lowering`** | — | 把语言地址空间翻成目标地址空间编号等目标相关决策落地的 pass | 第 27 章 |
| **`lang_address_space(...)`** | — | 语言层地址空间（`offload_global` / `offload_constant` / `offload_local` …） | 第 27 章 |
| **`target_address_space(N)`** | — | 目标层地址空间编号（NVPTX：global=1、constant=4、shared=3；0 为默认） | 第 27 章 |
| **`MemorySpaceAttrInterface`** | — | MLIR 接口，让上面两个属性能出现在 `!cir.ptr<...>` 的同一个语法位置 | 第 27 章 |
| **device stub** | — | 主机端代表某个 `__global__` kernel 的存根函数（`__device_stub__` 前缀） | 第 27 章 |
| **`cu.kernel_name`** | — | 挂在 stub 上的属性，记住设备端 kernel 的 mangled name | 第 27 章 |
| **`cu.var_registration`** | — | 挂在主机端影子全局上的属性，记录注册所需的名字与标志 | 第 27 章 |
| **`cu.binary_handle`** | — | 挂在 ModuleOp 上的属性，记录传给主机端的设备二进制文件名 | 第 27 章 |
| **fatbin** | fat binary | 打包了一个或多个架构的设备二进制；主机端在 `__cuda_module_ctor` 里注册它 | 第 27 章 |
| **host–device co-optimization** | — | 让主机与设备表示在同一层共存后才可能做的跨边界优化（探索中） | 第 27 章 |
| **FIR** | Fortran IR | Flang 在 AST 与 LLVM IR 之间的 MLIR 方言；CIR 之于 Clang 相当于 FIR 之于 Flang | 第 25 章 |

<div class="keypoint">
<strong>四层坐标系：LLVM（基础设施）→ SYCL（编程模型）→ SPIR-V（设备 IR）→ UR（运行时适配）</strong>
<strong>读全书任何一段代码，先定位它在哪一层：</strong>LLVM 层是 IR/Pass/TableGen 那套通用基础设施；SYCL 层是 queue/accessor/kernel_bundle 那套用户 API 与 clang 前端（Sema/CodeGen）；SPIR-V 层是设备镜像与 `llvm-spirv`/libclc/libdevice；UR 层是运行时到各后端的适配。<strong>同一个功能往往贯穿四层</strong>（如第 19 章 <code>sycl::sqrt</code>：SYCL 头 → `__spirv_ocl_sqrt`（SPIR-V）→ libclc/驱动，第 23 章 kernel_compiler：kernel_bundle（SYCL）→ sycl-jit（SPIR-V）→ 镜像加载（UR）），认清分层就不会在缩写里迷路。
</div>

## 小结

- 本附录按四层——**LLVM / SYCL / SPIR-V / UR**——汇总全书术语，每条给全称、一句话解释、正文章节；末尾单列 **ClangIR** 一节，对应第五部分（第 25–27 章），锚点是上游 `llvm/llvm-project` 而非 `intel/llvm`。
- 记不清某缩写时回来查；想真正理解，跟"正文"列回到对应章节。
- 最有用的心智模型是那条**四层坐标系**：一个功能常从 SYCL 用户 API，穿过 clang 前端、SPIR-V 设备 IR，落到 UR 运行时与具体后端——每一层都有自己的一套词汇。
