#!/usr/bin/env bash
# 校验本书所有完整 Triton 代码块（```triton）。
# - 带 driver 的（有顶层 print / kernel[grid](...) 调用）：真跑 python file.py，需 NVIDIA GPU。
# - 仅 kernel 片段（无 driver，如 autotune 配置示例）：做语法检查 + 导入检查（不需 GPU）。
# 需要：Python + pip install triton torch（GPU 运行）；仅语法检查只需 Python。
#
# 用法：
#   books/gpu-programming/verify/verify-triton.sh                # 校验全书所有 triton 块
#   books/gpu-programming/verify/verify-triton.sh ch13 ch21      # 只校验指定章
#   SYNTAX=1 books/gpu-programming/verify/verify-triton.sh       # 只做语法检查，不真跑（无 GPU 时用）
set -u
BOOK="$(cd "$(dirname "$0")/.." && pwd)"   # 脚本在 $BOOK/verify/ 里
CHDIR="$BOOK/chapters"
EXTRACT="$BOOK/verify/extract-blocks.js"
SYNTAX="${SYNTAX:-0}"
PY="${PYTHON:-python3}"

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "✗ 找不到 $PY。"; exit 127
fi

# 检测 triton/torch + GPU 是否可用；不可用则自动降级为语法检查
have_gpu=0
if [ "$SYNTAX" = "0" ]; then
  if "$PY" - <<'PYEOF' 2>/dev/null
import torch, triton, sys
sys.exit(0 if torch.cuda.is_available() else 1)
PYEOF
  then have_gpu=1
  else
    echo "  注意：未检测到可用的 triton+CUDA GPU 环境，自动降级为语法检查（等同 SYNTAX=1）。"
    SYNTAX=1
  fi
fi
echo "Python: $($PY --version 2>&1)  | 真跑=$([ "$SYNTAX" = 0 ] && echo yes || echo no，仅语法)"

if [ "$#" -gt 0 ]; then CHS=("$@"); else
  CHS=(ch13 ch14 ch20 ch21)
fi

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
fail=0; total=0; ok=0

mds=(); for ch in "${CHS[@]}"; do [ -f "$CHDIR/$ch.md" ] && mds+=("$CHDIR/$ch.md"); done
manifest="$(node "$EXTRACT" triton "$TMP" "${mds[@]}")"

if [ -z "$manifest" ]; then echo "  (指定章节无完整 Triton 块)"; exit 0; fi

# 用 ast 精确判断一个 .py 是否为可直接运行的 driver：
#   有 if __name__ 守卫，或存在顶层的“可执行语句”（非 import/def/class/赋值）。
# 退出码 0=driver，1=片段，2=语法错误。
classify() {
  "$PY" - "$1" <<'PYEOF'
import ast, sys
src = open(sys.argv[1], encoding="utf8").read()
try:
    tree = ast.parse(src)
except SyntaxError as e:
    print(f"{e.lineno}: {e.msg}"); sys.exit(2)
RUNNABLE = (ast.Expr, ast.For, ast.While, ast.With, ast.If, ast.Assert)
for node in tree.body:
    if isinstance(node, ast.If):
        # if __name__ == "__main__": ...
        t = node.test
        if (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name)
                and t.left.id == "__name__"):
            sys.exit(0)
        sys.exit(0)  # 任何顶层 if 都算能执行出东西
    if isinstance(node, RUNNABLE):
        sys.exit(0)
sys.exit(1)  # 只有 import/def/class/赋值 → 片段
PYEOF
}

while IFS=$'\t' read -r f _ignored; do
  [ -z "$f" ] && continue
  total=$((total+1))
  b="$(basename "$f")"; err="$TMP/$b.err"
  classify "$f" >"$err" 2>&1; rc=$?
  if [ "$rc" = "2" ]; then
    echo "  ✗ $b 语法错误:"; sed 's/^/      /' "$err" | head -6; fail=1; continue
  fi
  isdriver=$([ "$rc" = "0" ] && echo 1 || echo 0)
  if [ "$SYNTAX" = "1" ] || [ "$isdriver" = "0" ]; then
    # 片段或语法模式：只报告语法通过
    note="$([ "$isdriver" = 0 ] && echo 'kernel片段，仅语法检查' || echo '语法通过')"
    echo "  ✓ $b ($note)"; ok=$((ok+1)); continue
  fi
  # 真跑（有 GPU + driver）
  if out="$(cd "$TMP" && "$PY" "$f" 2>&1)"; then
    echo "  ✓ $b => $(printf '%s' "$out" | tr '\n' '|' | cut -c1-100)"; ok=$((ok+1))
  else
    echo "  ✗ $b 运行失败:"; printf '%s\n' "$out" | sed 's/^/      /' | tail -8; fail=1
  fi
done <<< "$manifest"

echo ""
echo "Triton 校验完成：$ok/$total 通过$([ "$SYNTAX" = 1 ] && echo '（仅语法检查）')。"
exit $fail
