# 附录 C 源码导览地图：想看某个功能去哪个目录

## 这份附录怎么用

<div class="goal-box">
<p>intel/llvm 是个几十万文件的 monorepo。全书教了你"某个功能在哪个文件"，这份附录把它们<strong>倒过来组织成一张地图</strong>：从"我想看 X 功能"出发，直接告诉你去哪个目录、哪个关键文件。它是你合上书、真正动手改代码时最先翻的一页。</p>
<p>用法：先按<strong>顶层目录总览</strong>定位大区（编译器？运行时？设备库？），再查<strong>按功能索引</strong>找到具体文件。所有路径相对 intel/llvm（<code>sycl</code> 分支）根目录，<strong>行号会随分支演进漂移，以你手上的树为准</strong>（正文反复强调的"源码为锚"——先 <code>grep</code>/<code>find</code> 再引用）。</p>
</div>

## 顶层目录总览

intel/llvm 根目录下和 SYCL/DPC++ 最相关的顶层目录（与 `llvm/`、`clang/` 平级）：

| 顶层目录 | 是什么 | 正文 |
|---|---|---|
| `llvm/` | LLVM 本体：IR、Pass、后端、工具（`opt`/`llc`……） | 第 1–11 章 |
| `clang/` | Clang 前端：Sema、CodeGen、Driver、属性定义 | 第 12–15、22 章 |
| `sycl/` | SYCL 运行时、头文件、扩展文档、测试 | 第 13–24 章 |
| `libdevice/` | 设备端 C 库/数学 fallback（编成 `libsycl-*.bc`） | 第 19 章 |
| `libclc/` | 底层 SPIR-V/OpenCL 内建原语（尤为 NVPTX/AMDGPU） | 第 11、19 章 |
| `llvm-spirv/` | LLVM IR ↔ SPIR-V 翻译器 | 第 9 章 |
| `sycl-jit/` | 进程内 SYCL RTC（kernel_compiler 的 SYCL 路径） | 第 23 章 |
| `unified-runtime/` | UR：运行时到各后端的统一适配层 + adapters | 第 18 章 |
| `xpti/` `xptifw/` | 插桩/追踪框架 | 第 18 章 |
| `libsycl/` | 新版 SYCL 运行时库（演进中） | — |
| `offload/` | LLVM offload 基础设施 | — |

## clang/：编译器前端

想看**编译器怎么处理 SYCL**，几乎都在 `clang/` 下。关键子目录与文件：

| 想看什么 | 去哪 | 正文 |
|---|---|---|
| SYCL 语义分析（认 kernel、free function） | `clang/lib/Sema/SemaSYCL.cpp` | 第 22 章 |
| SYCL 属性的 handle/add/merge 检查 | `clang/lib/Sema/SemaSYCLDeclAttr.cpp` | 第 22 章 |
| 所有属性的 TableGen 定义 | `clang/include/clang/Basic/Attr.td` | 第 22 章 |
| 属性文档（每个属性必须写） | `clang/include/clang/Basic/AttrDocs.td` | 第 22 章 |
| SYCL 相关诊断消息 | `clang/include/clang/Basic/DiagnosticSemaKinds.td` | 第 22 章 |
| AST/属性 → LLVM IR（属性落 metadata/字符串） | `clang/lib/CodeGen/CodeGenFunction.cpp` `CodeGenModule.cpp` `CGCall.cpp` | 第 15、22 章 |
| 语言选项（`SYCLIsDevice` 等） | `clang/include/clang/Basic/LangOptions.def` | 第 22 章 |
| 驱动/前端开关（`-fsycl` 等） | `clang/include/clang/Options/Options.td`（**已从 `Driver/` 移来**） | 第 22 章 |
| SYCL 设备库选择（`getDeviceLibNames`） | `clang/lib/Driver/ToolChains/SYCL.cpp` | 第 19 章 |
| SPIR-V 内建表 | `clang/lib/Sema/SPIRVBuiltins.td` | 第 19 章 |
| SYCL 架构开发者文档 | `clang/docs/SYCLSupport.rst` | 第 12、22 章 |

## sycl/：运行时、头文件、扩展、测试

`sycl/` 是 SYCL 侧的家，内部分工清晰：

| 想看什么 | 去哪 | 正文 |
|---|---|---|
| 用户 API 头文件（`queue`/`buffer`/`handler`……） | `sycl/include/sycl/` | 第 16 章 |
| 扩展 API 头（按 vendor 分） | `sycl/include/sycl/ext/{oneapi,intel,codeplay}/` | 第 20、21 章 |
| 数学内建映射（`sycl::sqrt`→`__spirv_ocl_*`） | `sycl/include/sycl/detail/builtins/math_functions.hpp` | 第 19 章 |
| feature-test 宏（生成 `feature_test.hpp`） | `sycl/source/feature_test.hpp.in` | 第 20 章 |
| 运行时实现（调度、镜像管理） | `sycl/source/detail/` | 第 16–18 章 |
| 程序/镜像管理器 | `sycl/source/detail/program_manager/` | 第 12、19 章 |
| bindless images 运行时 | `sycl/source/detail/bindless_images.cpp` | 第 23 章 |
| kernel_compiler 运行时（三语言分派） | `sycl/source/detail/kernel_compiler/` | 第 23 章 |
| 扩展规范文档（五态目录） | `sycl/doc/extensions/{proposed,experimental,supported,deprecated,removed}/` | 第 20、24 章 |
| 扩展流程/模板 | `sycl/doc/extensions/README-process.md` `template.asciidoc` | 第 24 章 |
| 设计文档（编译期属性、RTC……） | `sycl/doc/design/` （`CompileTimeProperties.md` `SYCL-RTC.md`） | 第 22、23 章 |
| 上手指南 | `sycl/doc/GetStartedGuide.md` | 附录 A |
| **LIT 测试**（不上真机） | `sycl/test/` | 第 24 章 |
| **E2E 测试**（上真设备） | `sycl/test-e2e/` | 第 24 章 |

## 设备库与 SPIR-V

设备端"标准库从哪来"和"IR 怎么变 SPIR-V"分散在几个顶层项目：

| 想看什么 | 去哪 | 正文 |
|---|---|---|
| C 库/数学 fallback 的 wrapper | `libdevice/*_wrapper.cpp`（`crt_`/`cmath_`/`imf_`） | 第 19 章 |
| fallback 具体实现 | `libdevice/fallback-*.hpp`（`fallback-bfloat16.cpp` 仍是 `.cpp`） | 第 19 章 |
| weak 属性宏定义 | `libdevice/device.h`（`DEVICE_EXTERNAL`） | 第 19 章 |
| `__spirv_*` 内建实现（各目标） | `libclc/libspirv/lib/<target>/` | 第 11、19 章 |
| LLVM IR → SPIR-V 翻译 | `llvm-spirv/lib/SPIRV/` | 第 9 章 |
| SYCL 源码 RTC（进程内 clang） | `sycl-jit/jit-compiler/lib/rtc/DeviceCompilation.cpp` | 第 23 章 |
| IR → SPIR-V（sycl-jit 内） | `sycl-jit/jit-compiler/lib/.../SPIRVLLVMTranslation.cpp` | 第 23 章 |

## UR 与后端

运行时到硬件的最后一段：

| 想看什么 | 去哪 | 正文 |
|---|---|---|
| UR C API 定义（`ur*`/`ur*Exp`） | `unified-runtime/include/unified-runtime/ur_api.h` | 第 18、23 章 |
| 各后端 adapter | `unified-runtime/source/adapters/{level_zero,opencl,cuda,hip}/` | 第 18 章 |
| bindless images 的 UR 入口 | `ur_api.h` 里 `urBindlessImages*Exp` | 第 23 章 |

## 一个实用工作流：从"现象"倒查到"源码"

合上书真动手时，最常见的不是"我知道去哪个文件"，而是"我看到某个行为/字符串，想找它从哪来"。推荐的倒查流程：

```bash
# 1. 看到 IR 里一个陌生的字符串属性 "sycl-xxx"，全仓搜它从哪发出
grep -rn '"sycl-xxx"' clang/lib sycl/source

# 2. 看到一个 __spirv_ 或 __devicelib_ 符号，定位它的声明/实现
grep -rn '__devicelib_ceilf' libdevice/

# 3. 想知道某扩展宏在哪定义、值多少
grep -n 'SYCL_EXT_ONEAPI_XXX' sycl/source/feature_test.hpp.in

# 4. 想找某功能的测试当"活文档"看用法
grep -rl 'feature_you_want' sycl/test/ sycl/test-e2e/

# 5. 想找一个 clang 属性的全链（定义→检查→codegen）
grep -rn 'SYCLReqdWorkGroupSize' clang/include clang/lib
```

<div class="keypoint">
<strong>定位三问：哪一层？哪个顶层目录？哪个文件？</strong>
<strong>找任何功能先答三个问题：① 它属于哪一层（附录 B 的 LLVM/SYCL/SPIR-V/UR 四层坐标系）；② 对应哪个顶层目录（编译器→<code>clang/</code>，运行时→<code>sycl/source/</code>，设备库→<code>libdevice/</code>/<code>libclc/</code>，翻译→<code>llvm-spirv/</code>，适配→<code>unified-runtime/</code>）；③ 具体文件（查本附录的功能索引表）。</strong>找不准就用"倒查工作流"——从你看到的字符串/符号 <code>grep -rn</code> 反向定位，比记路径更可靠。行号一定以你手上的树为准，先搜再引。
</div>

## 小结

- 本附录把全书源码锚点倒过来组织成地图：**顶层目录总览 + 按功能索引（clang / sycl / 设备库+SPIR-V / UR）**。
- 找功能的顺序：先定层（附录 B 四层）→ 定顶层目录 → 查功能表定文件；记不住路径就用 `grep -rn` 从现象倒查。
- 高频落点：编译器前端 `clang/lib/Sema/SemaSYCL*.cpp` + `clang/include/clang/Basic/Attr.td`；运行时 `sycl/source/detail/`；扩展文档 `sycl/doc/extensions/`；设备库 `libdevice/`+`libclc/`；翻译 `llvm-spirv/`；适配 `unified-runtime/`。
- 铁律：**行号会漂移，路径可能搬家（如 `Options.td` 从 `Driver/` 到 `Options/`）——永远先在你手上的树里搜一遍再引用**。这也是全书"源码为锚"精神的收尾。
