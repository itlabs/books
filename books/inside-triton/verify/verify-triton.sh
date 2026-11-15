#!/usr/bin/env bash
# 校验本书各章的完整代码块（```triton / ```gluon / ```helion 真跑，```triton-ir 编译）。
# 逻辑都在 verify-triton.py 里，这里只负责挑 Python。
#
# 用法（在仓库根目录）：
#   books/inside-triton/verify/verify-triton.sh               # 全书
#   books/inside-triton/verify/verify-triton.sh ch12 ch13     # 只校验指定章
#   CE=1 books/inside-triton/verify/verify-triton.sh          # triton-ir 块走 godbolt（会上传代码块）
#   VERBOSE=1 books/inside-triton/verify/verify-triton.sh     # 每块的输出都打印
#   PYTHON=<venv>/bin/python books/inside-triton/verify/verify-triton.sh
#
# Python 默认取 $PYTHON，其次当前激活的 venv，最后 PATH 里的 python3。
# 它必须装了 torch 和 triton（以及 helion，如果要跑 helion 块）。Intel GPU 上还要先 source env-xpu.sh。
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-}"
if [ -z "$PY" ] && [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then PY="$VIRTUAL_ENV/bin/python"; fi
PY="${PY:-$(command -v python3)}"
if ! "$PY" -c "import torch, triton" 2>/dev/null; then
  echo "✗ $PY 里没有 torch / triton。用 PYTHON=<venv>/bin/python 指定装好的环境（见附录 A）。"
  exit 127
fi
exec "$PY" "$HERE/verify-triton.py" "$@"
