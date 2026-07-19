#!/usr/bin/env bash
# 校验本书所有完整 CUDA 代码块（```cuda / ```cuda-ptx）都能用 nvcc 编译；
# 有 main 的编译并运行、打印输出（截断）；cuda-ptx 只生成 PTX（--ptx，不需要 GPU 也能过）。
# 需要：NVIDIA CUDA Toolkit（nvcc）。运行/执行需要一块 NVIDIA GPU；仅编译不需要。
#
# 用法：
#   books/gpu-programming/verify/verify-cuda.sh                # 校验全书所有 cuda 块
#   books/gpu-programming/verify/verify-cuda.sh ch09 ch11      # 只校验指定章
#   ARCH=sm_75 books/gpu-programming/verify/verify-cuda.sh     # 指定架构（默认 sm_70）
#   NORUN=1 books/gpu-programming/verify/verify-cuda.sh        # 只编译不运行（无 GPU 时用）
set -u
BOOK="$(cd "$(dirname "$0")/.." && pwd)"   # 脚本在 $BOOK/verify/ 里
CHDIR="$BOOK/chapters"
EXTRACT="$BOOK/verify/extract-blocks.js"
ARCH="${ARCH:-sm_70}"
NORUN="${NORUN:-0}"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "✗ 找不到 nvcc。请先安装 CUDA Toolkit，并确保 nvcc 在 PATH 中（见附录 A）。"
  exit 127
fi
echo "nvcc: $(nvcc --version | grep -oE 'release [0-9.]+' | head -1)  | 目标架构: $ARCH  | NORUN=$NORUN"
if [ "$NORUN" = "0" ] && ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "  注意：未检测到 nvidia-smi（可能无 GPU）。若运行失败，可用 NORUN=1 只做编译校验。"
fi

# 章节列表：默认全书 cuda 块所在章（含 appendix-a）
if [ "$#" -gt 0 ]; then CHS=("$@"); else
  CHS=(ch01 ch04 ch07 ch08 ch09 ch10 ch11 ch17 ch18 ch20 ch21 appendix-a)
fi

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
fail=0; total=0; ok=0

for lang in cuda cuda-ptx; do
  mds=(); for ch in "${CHS[@]}"; do [ -f "$CHDIR/$ch.md" ] && mds+=("$CHDIR/$ch.md"); done
  [ "${#mds[@]}" -eq 0 ] && continue
  manifest="$(node "$EXTRACT" "$lang" "$TMP" "${mds[@]}")"
  [ -z "$manifest" ] && continue
  while IFS=$'\t' read -r f hasmain; do
    [ -z "$f" ] && continue
    total=$((total+1))
    b="$(basename "$f")"
    err="$TMP/$b.err"
    if [ "$lang" = "cuda-ptx" ]; then
      # PTX：只生成中间码，不需要 GPU
      if nvcc -std=c++17 -arch=$ARCH --ptx "$f" -o "$TMP/$b.ptx" 2>"$err"; then
        echo "  ✓ $b (生成 PTX 通过)"; ok=$((ok+1))
      else
        echo "  ✗ $b PTX 生成失败:"; sed 's/^/      /' "$err" | head -6; fail=1
      fi
      continue
    fi
    if nvcc -std=c++17 -arch=$ARCH "$f" -o "$TMP/$b.out" 2>"$err"; then
      if [ "$hasmain" = "1" ] && [ "$NORUN" = "0" ]; then
        if out="$("$TMP/$b.out" 2>&1)"; then
          echo "  ✓ $b => $(printf '%s' "$out" | tr '\n' '|' | cut -c1-100)"; ok=$((ok+1))
        else
          echo "  ✗ $b 编译通过但运行失败（退出码 $?）:"; printf '%s\n' "$out" | sed 's/^/      /' | head -6; fail=1
        fi
      else
        echo "  ✓ $b (编译通过$([ "$NORUN" = 1 ] && echo '，NORUN 未运行'))"; ok=$((ok+1))
      fi
    else
      echo "  ✗ $b 编译失败:"; sed 's/^/      /' "$err" | head -8; fail=1
    fi
  done <<< "$manifest"
done

echo ""
echo "CUDA 校验完成：$ok/$total 通过。"
exit $fail
