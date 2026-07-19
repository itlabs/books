#!/usr/bin/env bash
# 校验本书所有完整 SYCL 代码块（```sycl）都能用 icpx -fsycl 编译；
# 有 main 的编译并运行、打印输出（截断）。
# 需要：Intel oneAPI DPC++（icpx）。默认编到 CPU/host 也能跑；有 GPU 时可加 target。
#
# 用法：
#   books/gpu-programming/verify/verify-sycl.sh                     # 校验全书所有 sycl 块
#   books/gpu-programming/verify/verify-sycl.sh ch12 ch20           # 只校验指定章
#   NORUN=1 books/gpu-programming/verify/verify-sycl.sh             # 只编译不运行
#   SYCL_DEVICE_FILTER=cpu books/gpu-programming/verify/verify-sycl.sh   # 强制用 CPU 设备运行（无 GPU 时）
#   CXX=acpp books/gpu-programming/verify/verify-sycl.sh            # 改用 AdaptiveCpp 编译器
set -u
BOOK="$(cd "$(dirname "$0")/.." && pwd)"   # 脚本在 $BOOK/verify/ 里
CHDIR="$BOOK/chapters"
EXTRACT="$BOOK/verify/extract-blocks.js"
NORUN="${NORUN:-0}"
CXX="${CXX:-icpx}"

if ! command -v "$CXX" >/dev/null 2>&1; then
  echo "✗ 找不到编译器 '$CXX'。请安装 Intel oneAPI（icpx）或 AdaptiveCpp（设 CXX=acpp）。见附录 A。"
  echo "  提示：oneAPI 需先 source 环境，如：source /opt/intel/oneapi/setvars.sh"
  exit 127
fi

# 组装编译命令：icpx 用 -fsycl；acpp 不需要
if [ "$CXX" = "acpp" ]; then
  BUILD=("$CXX" -O2 -std=c++17)
else
  BUILD=("$CXX" -fsycl -O2 -std=c++17)
fi
echo "编译器: $($CXX --version 2>/dev/null | head -1)  | NORUN=$NORUN"
[ -n "${SYCL_DEVICE_FILTER:-}" ] && echo "  SYCL_DEVICE_FILTER=$SYCL_DEVICE_FILTER"

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
