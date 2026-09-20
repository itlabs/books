# pix 方言 —— 《深入 MLIR》贯穿案例

这是一个 **out-of-tree** 的 MLIR 项目，从本书第 7 章开始搭建，之后每一章都在它身上增量推进：
第 8 章补齐 op 集、第 9 章加自定义类型 `!pix.kernel` 与枚举属性 `#pix.border`、第 10 章写 verifier、第 11 章写 folder、
第 12 章配齐 lit 测试，第三、四部分再给它写 pass、pattern、conversion，一路下降到 LLVM 与 GPU。

## 构建

需要一份装好的 LLVM/MLIR（见本书附录 A）。最省事的方式是用随书脚本：

```bash
books/inside-mlir/verify/verify-pix.sh          # 自动探测 MLIR_DIR，构建 + 跑 lit 测试
MLIR_DIR=~/llvm-project/build/lib/cmake/mlir books/inside-mlir/verify/verify-pix.sh
```

手动构建：

```bash
cmake -G Ninja -S . -B build \
  -DMLIR_DIR=<llvm-build>/lib/cmake/mlir \
  -DLLVM_DIR=<llvm-build>/lib/cmake/llvm \
  -DCMAKE_BUILD_TYPE=Release
ninja -C build            # 产物：build/bin/pix-opt
ninja -C build check-pix  # 跑 test/ 下的 lit 测试
```

## 目录

```
CMakeLists.txt        顶层：find_package(MLIR) + 一串 include(...)
include/Pix/
  PixDialect.td       方言声明 + op 基类 Pix_Op
  PixOps.td           op 的 ODS 定义（第 8 章的主场）
  PixDialect.h        方言的 C++ 入口（include 生成的 .h.inc）
  PixOps.h            op 的 C++ 声明
lib/Pix/
  PixDialect.cpp      initialize()：把 op 注册进方言
  PixOps.cpp          op 实现（verify/fold 以后加在这里）
pix-opt/pix-opt.cpp   "你自己的 mlir-opt"：注册 pass + 方言，交给 MlirOptMain
test/Pix/             lit 测试：roundtrip / cse / invalid
```

## 用它做实验

```bash
build/bin/pix-opt test/Pix/roundtrip.mlir                      # 解析 + 验证 + 打印
build/bin/pix-opt test/Pix/roundtrip.mlir --mlir-print-op-generic
build/bin/pix-opt test/Pix/cse.mlir --cse                       # 免费得到的 CSE
build/bin/pix-opt --show-dialects | grep pix                    # 确认方言注册上了
```
