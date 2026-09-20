# MLIR 代码校验脚本

本书的 MLIR 代码块要真的过 `mlir-opt` 才算数——pass 名、op 名、类型语法这些东西版本一变就错，
光靠肉眼审不住。这里的脚本把各章的**完整代码块**抽出来逐个跑一遍，另有一个脚本负责构建贯穿全书的
pix 方言项目。

## 用法

在**仓库根目录**运行（不带参数=全书；也可只传章名）：

```bash
# 校验所有 ```mlir / ```mlir-lower / ```mlir-translate 块
books/inside-mlir/verify/verify-mlir.sh
books/inside-mlir/verify/verify-mlir.sh ch16 ch17                        # 只校验指定章
MLIR_BIN=~/llvm-project/build/bin books/inside-mlir/verify/verify-mlir.sh  # 指定工具位置
VERBOSE=1 books/inside-mlir/verify/verify-mlir.sh                        # 顺便打印输出 IR
CE=1 books/inside-mlir/verify/verify-mlir.sh                             # 没装本地 MLIR：走 godbolt

# 构建 pix 方言项目（code/pix/）并跑 lit 测试
books/inside-mlir/verify/verify-pix.sh
MLIR_DIR=~/llvm-project/build/lib/cmake/mlir books/inside-mlir/verify/verify-pix.sh
BUILD_DIR=/tmp/pix-build books/inside-mlir/verify/verify-pix.sh          # 复用构建目录，增量更快
NOTEST=1 books/inside-mlir/verify/verify-pix.sh                          # 只编译不跑测试
```

工具链缺失时脚本干净退出（退出码 127）并给出安装提示——步骤见**附录 A**。
全通过退出 0，任一块失败退出 1，可直接接 CI。这些脚本不参与 `node build.js` 的网页构建。

## `// RUN:` 行：一处写，两处用

`mlir-lower` / `mlir-translate` 块的第一行用 lit 风格注释写出要跑的管线：

```
// RUN: mlir-opt %s --pass-pipeline='builtin.module(func.func(canonicalize))'
```

`build.js` 和 `extract-blocks.js` 用**同一套解析**（取工具名之后、`|` 之前的部分，丢掉 `%s`）：

- 网页上 **🔗 在 Compiler Explorer 打开** 的按钮带的就是这些参数；
- 这里的校验脚本跑的也是这些参数。

于是「读者点开看到的」和「作者验证过的」不会各说一套。没写 `RUN:` 行的块按语言取默认参数：
`mlir` → `--verify-roundtrip`，`mlir-lower` → `--canonicalize`，`mlir-translate` → `--mlir-to-llvmir`。
`RUN:` 行是 MLIR 的 `//` 注释，留在正文里不影响解析，读者可以直接照抄到终端。

## 文件说明

- `extract-blocks.js` —— 代码块提取器（Node）。按语言从章节 md 抽出完整块（`mlir-norun` 片段不抽）
  到临时目录，并解析出每块要用的参数。
- `verify-mlir.sh` —— 逐块跑 `mlir-opt` / `mlir-translate`。`mlir` 块用 `--verify-roundtrip`：
  既查语法，也跑 verifier，还确认 IR 能原样打印回来。
- `ce-compile.js` —— 没有本地 MLIR 时的兜底：把块发给 Compiler Explorer 的 trunk 版工具
  （`mliropttrunk` / `mlirtranslatetrunk`）跑，用它的退出码判定。**会把代码块上传到 godbolt.org**；
  本书代码块都是公开示例，但你若不想走网络，就装本地 `mlir-opt`。
- `verify-pix.sh` —— 配置并构建 `code/pix/`（out-of-tree 方言项目），再跑 `check-pix`。
  项目还没落地（第 7 章批次之前）时干净跳过。

> 拿不准某个 pass 名或选项在当前版本还存不存在时，最快的确认方式就是写成一个小块跑
> `verify-mlir.sh`（或 `CE=1`）——工具会直接告诉你 `Unknown command line argument`。
