#!/usr/bin/env bash
# 构建本书的贯穿案例——out-of-tree 的 pix 方言项目（books/inside-mlir/code/pix/），
# 并跑它自带的 lit 测试。第 7 章起每一章都在这个项目上增量推进，所以它必须一直是能编过的。
#
# 需要：一份**装好的** LLVM/MLIR（带 MLIRConfig.cmake / LLVMConfig.cmake）+ cmake + ninja。
# 只有源码构建目录也行：MLIR_DIR 指到 <build>/lib/cmake/mlir。搭建见附录 A。
#
# 用法：
#   books/inside-mlir/verify/verify-pix.sh
#   MLIR_DIR=~/llvm-project/build/lib/cmake/mlir LLVM_DIR=~/llvm-project/build/lib/cmake/llvm \
#     books/inside-mlir/verify/verify-pix.sh
#   BUILD_DIR=/tmp/pix-build books/inside-mlir/verify/verify-pix.sh   # 复用构建目录（增量，快）
#   NOTEST=1 books/inside-mlir/verify/verify-pix.sh                   # 只编译，不跑 lit
#   PIX_LIT=<llvm-build>/bin/llvm-lit books/inside-mlir/verify/verify-pix.sh  # 指定 lit
set -u
BOOK="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$BOOK/code/pix"
NOTEST="${NOTEST:-0}"

if [ ! -f "$SRC/CMakeLists.txt" ]; then
  echo "－ 还没有 $SRC（pix 项目在第 7 章批次里落地），跳过。"
  exit 0
fi
for t in cmake ninja; do
  command -v "$t" >/dev/null 2>&1 || { echo "✗ 找不到 $t。见附录 A。"; exit 127; }
done

# MLIR_DIR/LLVM_DIR 没给就试着从 llvm-config 推出来
if [ -z "${MLIR_DIR:-}" ] && command -v llvm-config >/dev/null 2>&1; then
  libdir="$(llvm-config --libdir 2>/dev/null)"
  [ -d "$libdir/cmake/mlir" ] && MLIR_DIR="$libdir/cmake/mlir"
  [ -z "${LLVM_DIR:-}" ] && [ -d "$libdir/cmake/llvm" ] && LLVM_DIR="$libdir/cmake/llvm"
fi
if [ -z "${MLIR_DIR:-}" ]; then
  echo "✗ 不知道 MLIR 装在哪。请指定："
  echo "    MLIR_DIR=<llvm-build>/lib/cmake/mlir LLVM_DIR=<llvm-build>/lib/cmake/llvm $0"
  echo "  （源码构建或发行版的 mlir-<版本>-dev 包都可以，见附录 A）"
  exit 127
fi

# lit：out-of-tree 项目必须显式告诉 CMake lit 在哪，否则 check-pix 会去找
# <build>/bin/llvm-lit（那个只存在于 LLVM 自己的构建树里）而报 "not found"。
if [ -z "${PIX_LIT:-}" ]; then
  llvm_root="$(dirname "$(dirname "$MLIR_DIR")")"   # …/lib/cmake/mlir → …/
  for c in "$llvm_root/../bin/llvm-lit" "$llvm_root/bin/llvm-lit" \
           "$(command -v llvm-lit 2>/dev/null)" "$(command -v lit 2>/dev/null)" \
           "${LLVM_SRC:-}/llvm/utils/lit/lit.py"; do
    [ -n "$c" ] && [ -f "$c" ] && { PIX_LIT="$(cd "$(dirname "$c")" && pwd)/$(basename "$c")"; break; }
  done
fi

BUILD_DIR="${BUILD_DIR:-$(mktemp -d)}"
echo "源码：$SRC"
echo "MLIR_DIR：$MLIR_DIR"
echo "构建目录：$BUILD_DIR"
if [ -n "${PIX_LIT:-}" ]; then
  echo "lit：$PIX_LIT"
else
  echo "lit：没找到。编译没问题，但 check-pix 会失败；可以 NOTEST=1 只编译，"
  echo "     或用 PIX_LIT=<llvm-build>/bin/llvm-lit 指定（也可以指到源码树的 llvm/utils/lit/lit.py）。"
fi

cmake -G Ninja -S "$SRC" -B "$BUILD_DIR" \
  -DMLIR_DIR="$MLIR_DIR" ${LLVM_DIR:+-DLLVM_DIR="$LLVM_DIR"} \
  ${PIX_LIT:+-DLLVM_EXTERNAL_LIT="$PIX_LIT"} \
  -DCMAKE_BUILD_TYPE=Release || { echo "✗ cmake 配置失败"; exit 1; }

ninja -C "$BUILD_DIR" || { echo "✗ 编译失败"; exit 1; }
echo "✓ pix-opt 编译通过：$BUILD_DIR/bin/pix-opt"

if [ "$NOTEST" = "1" ]; then
  echo "（NOTEST=1，跳过 lit 测试）"
  exit 0
fi
# lit 测试目标名与第 12 章保持一致
if ninja -C "$BUILD_DIR" -t targets all 2>/dev/null | grep -q '^check-pix'; then
  ninja -C "$BUILD_DIR" check-pix || { echo "✗ lit 测试失败"; exit 1; }
  echo "✓ check-pix 全部通过"
else
  echo "－ 这一版项目还没有 check-pix 目标（第 12 章才加），只做了编译校验。"
fi
