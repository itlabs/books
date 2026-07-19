#!/usr/bin/env bash
# 校验本书所有完整 SYCL 代码块（```sycl）都能用 -fsycl 编译；
# 有 main 的编译并运行、打印输出（截断）。
# 支持三种 SYCL 编译器（都走 -fsycl，除 acpp 外）：
#   · Intel oneAPI DPC++ 的 icpx（默认）
#   · intel/llvm 源码构建出的 clang++（开源 DPC++，同一套 LLVM、同样的 -fsycl）
#   · AdaptiveCpp 的 acpp（CXX=acpp，不需要 -fsycl）
#
# 用法：
#   books/gpu-programming/verify/verify-sycl.sh                     # 校验全书所有 sycl 块
#   books/gpu-programming/verify/verify-sycl.sh ch12 ch20           # 只校验指定章
#   NORUN=1 books/gpu-programming/verify/verify-sycl.sh             # 只编译不运行
#   CXX=$HOME/sycl_workspace/llvm/build/bin/clang++ books/.../verify-sycl.sh   # 用 intel/llvm clang++
#   SYCL_TARGETS=spir64 books/gpu-programming/verify/verify-sycl.sh # 显式指定 -fsycl-targets（intel/llvm 常需）
#       CPU/OpenCL: spir64  | NVIDIA: nvptx64-nvidia-cuda  | AMD: amdgcn-amd-amdhsa
#   ONEAPI_DEVICE_SELECTOR=opencl:cpu books/.../verify-sycl.sh      # 强制用 CPU 设备运行（无 GPU 时）
#   CXX=acpp books/gpu-programming/verify/verify-sycl.sh            # 改用 AdaptiveCpp
set -u
BOOK="$(cd "$(dirname "$0")/.." && pwd)"   # 脚本在 $BOOK/verify/ 里
CHDIR="$BOOK/chapters"
EXTRACT="$BOOK/verify/extract-blocks.js"
NORUN="${NORUN:-0}"
CXX="${CXX:-icpx}"
SYCL_TARGETS="${SYCL_TARGETS:-}"

if ! command -v "$CXX" >/dev/null 2>&1; then
  echo "✗ 找不到编译器 '$CXX'。可选：Intel oneAPI 的 icpx、intel/llvm 构建的 clang++（用 CXX 指全路径）、或 AdaptiveCpp（CXX=acpp）。见附录 A。"
  echo "  提示：oneAPI 需先 source 环境，如：source /opt/intel/oneapi/setvars.sh"
  echo "        intel/llvm clang++ 需把 <build>/bin 加进 PATH，或用 CXX 指到它的全路径。"
  exit 127
fi

# 组装编译命令：icpx / intel-llvm clang++ 都用 -fsycl；acpp 不需要。
# intel/llvm 构建常需显式 -fsycl-targets，用 SYCL_TARGETS 传入。
if [ "$CXX" = "acpp" ]; then
  BUILD=("$CXX" -O2 -std=c++17)
else
  BUILD=("$CXX" -fsycl -O2 -std=c++17)
  [ -n "$SYCL_TARGETS" ] && BUILD+=("-fsycl-targets=$SYCL_TARGETS")
fi
echo "编译器: $($CXX --version 2>/dev/null | head -1)  | NORUN=$NORUN"
[ -n "$SYCL_TARGETS" ] && echo "  -fsycl-targets=$SYCL_TARGETS"
[ -n "${ONEAPI_DEVICE_SELECTOR:-}" ] && echo "  ONEAPI_DEVICE_SELECTOR=$ONEAPI_DEVICE_SELECTOR"
[ -n "${SYCL_DEVICE_FILTER:-}" ] && echo "  SYCL_DEVICE_FILTER=$SYCL_DEVICE_FILTER（已废弃，建议用 ONEAPI_DEVICE_SELECTOR）"

if [ "$#" -gt 0 ]; then CHS=("$@"); else
  CHS=(ch05 ch09 ch12 ch20 appendix-a)
fi

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
fail=0; total=0; ok=0

mds=(); for ch in "${CHS[@]}"; do [ -f "$CHDIR/$ch.md" ] && mds+=("$CHDIR/$ch.md"); done
manifest="$(node "$EXTRACT" sycl "$TMP" "${mds[@]}")"

if [ -z "$manifest" ]; then echo "  (指定章节无完整 SYCL 块)"; exit 0; fi

while IFS=$'\t' read -r f hasmain; do
  [ -z "$f" ] && continue
  total=$((total+1))
  b="$(basename "$f")"; err="$TMP/$b.err"
  if "${BUILD[@]}" "$f" -o "$TMP/$b.out" 2>"$err"; then
    if [ "$hasmain" = "1" ] && [ "$NORUN" = "0" ]; then
      if out="$("$TMP/$b.out" 2>&1)"; then
        echo "  ✓ $b => $(printf '%s' "$out" | tr '\n' '|' | cut -c1-100)"; ok=$((ok+1))
      else
        echo "  ✗ $b 编译通过但运行失败:"; printf '%s\n' "$out" | sed 's/^/      /' | head -6; fail=1
      fi
    else
      echo "  ✓ $b (编译通过$([ "$NORUN" = 1 ] && echo '，NORUN 未运行'))"; ok=$((ok+1))
    fi
  else
    echo "  ✗ $b 编译失败:"; sed 's/^/      /' "$err" | head -8; fail=1
  fi
done <<< "$manifest"

echo ""
echo "SYCL 校验完成：$ok/$total 通过。"
exit $fail
