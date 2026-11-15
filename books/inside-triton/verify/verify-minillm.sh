#!/usr/bin/env bash
# 跑 minillm 的全部 pytest（每个 kernel 与 PyTorch 参考实现对比数值）。
#
# 用法（在仓库根目录）：
#   books/inside-triton/verify/verify-minillm.sh                 # 全部
#   books/inside-triton/verify/verify-minillm.sh -k rmsnorm      # 其余参数原样传给 pytest
#   PYTHON=<venv>/bin/python books/inside-triton/verify/verify-minillm.sh
#
# 不要求先 pip install -e：用 PYTHONPATH 直接指向源码树，改完代码不用重装。
# Python 的挑法与 verify-triton.sh 相同；Intel GPU 上还要先 source env-xpu.sh。
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
CODE="$HERE/../code/minillm"
PY="${PYTHON:-}"
if [ -z "$PY" ] && [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then PY="$VIRTUAL_ENV/bin/python"; fi
PY="${PY:-$(command -v python3)}"
if ! "$PY" -c "import torch, triton, pytest" 2>/dev/null; then
  echo "✗ $PY 里缺 torch / triton / pytest。装法：cd $CODE && uv pip install -e '.[test]'（torch、triton 见附录 A）。"
  exit 127
fi
cd "$CODE" && PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}" exec "$PY" -m pytest "$@"
