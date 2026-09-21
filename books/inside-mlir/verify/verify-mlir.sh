#!/usr/bin/env bash
# 校验本书所有完整 MLIR 代码块（```mlir / ```mlir-lower / ```mlir-translate）都真的能过工具：
#   ```mlir            -> mlir-opt --verify-roundtrip（语法 + verifier 都过，且能原样打回来）
#   ```mlir-lower      -> mlir-opt <块首 RUN: 行里的管线>（默认 --canonicalize）
#   ```mlir-translate  -> mlir-translate <块首 RUN: 行，默认 --mlir-to-llvmir>
# 参数取自块首 lit 风格的 `// RUN: mlir-opt %s --foo` 行——与 build.js 给网页按钮用的是同一行，
# 所以"读者点开看到的"和"这里校验过的"永远是同一条管线。
#
# 需要：本地 mlir-opt / mlir-translate（LLVM/MLIR 构建或安装包，见附录 A）。
# 没有本地工具时可以用 CE=1 走 Compiler Explorer 的 trunk 版工具（需要联网）。
#
# 特别一类：RUN 行写 `pix-opt` 的块（第 7 章起的 pix 方言）只能用**本地编出来的 pix-opt**
# 验证——CE 上的 mlir-opt 不认识 pix 方言。用 PIX_OPT 指定它的位置，或让脚本自动探测
# verify-pix.sh 的构建目录；找不到就跳过（不算失败）。
#
# 用法：
#   books/inside-mlir/verify/verify-mlir.sh                    # 校验全书
#   books/inside-mlir/verify/verify-mlir.sh ch16 ch17          # 只校验指定章
#   MLIR_BIN=~/llvm-project/build/bin books/inside-mlir/verify/verify-mlir.sh
#   CE=1 books/inside-mlir/verify/verify-mlir.sh               # 无本地工具，走 godbolt（会上传代码块）
#   VERBOSE=1 books/inside-mlir/verify/verify-mlir.sh          # 同时打印每个块的输出 IR
#   PIX_OPT=/path/to/pix-build/bin/pix-opt books/inside-mlir/verify/verify-mlir.sh
set -u
BOOK="$(cd "$(dirname "$0")/.." && pwd)"   # 脚本在 $BOOK/verify/ 里
CHDIR="$BOOK/chapters"
EXTRACT="$BOOK/verify/extract-blocks.js"
CE_COMPILE="$BOOK/verify/ce-compile.js"
CE="${CE:-0}"
VERBOSE="${VERBOSE:-0}"

# 定位工具：MLIR_BIN 优先，其次 PATH。
tool_path() {
  if [ -n "${MLIR_BIN:-}" ] && [ -x "$MLIR_BIN/$1" ]; then echo "$MLIR_BIN/$1"; return 0; fi
  command -v "$1" 2>/dev/null; return $?
}
OPT="$(tool_path mlir-opt)"
TRANSLATE="$(tool_path mlir-translate)"

# pix-opt：PIX_OPT 优先，其次几个常见的构建目录，最后看 PATH
if [ -z "${PIX_OPT:-}" ]; then
  for c in "$BOOK/code/pix/build/bin/pix-opt" "${BUILD_DIR:-}/bin/pix-opt" "$(command -v pix-opt 2>/dev/null)"; do
    [ -n "$c" ] && [ -x "$c" ] && { PIX_OPT="$c"; break; }
  done
fi

# 只有 pix-opt 而没有 mlir-opt 也是合理状态（有些章全是 pix 方言的块），
# 那种情况下给个提醒继续跑，缺 mlir-opt 的块按"跳过"处理。
if [ "$CE" = "0" ] && [ -z "$OPT" ] && [ -n "${PIX_OPT:-}" ]; then
  echo "注意：没找到 mlir-opt，只有 pix-opt —— 内建方言的块会被跳过。"
elif [ "$CE" = "0" ] && [ -z "$OPT" ]; then
  echo "✗ 找不到 mlir-opt。三条路："
  echo "    1) 构建 LLVM/MLIR 后指定：MLIR_BIN=~/llvm-project/build/bin $0"
  echo "    2) 装发行版包（如 Ubuntu 的 mlir-<版本>-tools），确保 mlir-opt 在 PATH 里"
  echo "    3) 没有本地工具就走网络：CE=1 $0   （用 godbolt 的 trunk 版 mlir-opt）"
  echo "  安装步骤见附录 A。"
  exit 127
fi

if [ "$CE" = "1" ]; then
  command -v node >/dev/null 2>&1 || { echo "✗ CE=1 需要 node。"; exit 127; }
  echo "工具：Compiler Explorer trunk（mliropttrunk / mlirtranslatetrunk）——代码块会上传到 godbolt.org"
elif [ -n "$OPT" ]; then
  echo "工具：$OPT"
  "$OPT" --version 2>/dev/null | sed -n '1,2p' | sed 's/^/      /'
  [ -z "$TRANSLATE" ] && echo "  注意：没找到 mlir-translate，mlir-translate 块会被跳过。"
fi
if [ -n "${PIX_OPT:-}" ]; then
  echo "pix-opt：$PIX_OPT"
else
  echo "pix-opt：没找到（pix 方言的代码块会被跳过；先跑 verify-pix.sh 构建它，或用 PIX_OPT= 指定）"
fi

# 章节列表：默认全书（按实际存在的文件来，新章不用改脚本）
if [ "$#" -gt 0 ]; then
  CHS=("$@")
else
  CHS=()
  for f in "$CHDIR"/*.md; do [ -e "$f" ] && CHS+=("$(basename "$f" .md)"); done
fi
[ "${#CHS[@]}" -eq 0 ] && { echo "（$CHDIR 下还没有章节，无事可做）"; exit 0; }

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
fail=0; total=0; ok=0; skipped=0

for lang in mlir mlir-lower mlir-translate; do
  mds=(); for ch in "${CHS[@]}"; do [ -f "$CHDIR/$ch.md" ] && mds+=("$CHDIR/$ch.md"); done
  [ "${#mds[@]}" -eq 0 ] && continue
  manifest="$(node "$EXTRACT" "$lang" "$TMP" "${mds[@]}")"
  [ -z "$manifest" ] && continue

  case "$lang" in
    mlir)           defargs="--verify-roundtrip"; bin="$OPT";       ceid="mliropttrunk" ;;
    mlir-lower)     defargs="--canonicalize";     bin="$OPT";       ceid="mliropttrunk" ;;
    mlir-translate) defargs="--mlir-to-llvmir";   bin="$TRANSLATE"; ceid="mlirtranslatetrunk" ;;
  esac

  while IFS=$'\t' read -r f blocktool args; do
    [ -z "$f" ] && continue
    b="$(basename "$f")"
    [ -z "$args" ] && args="$defargs"

    # RUN 行里常有带引号的整体参数，例如
    #   --pix-to-arith='convert-signatures=true one-to-n=true'
    # 直接用未加引号的 $args 会被 shell 再切一刀，得到
    #   "Too many positional arguments specified!"
    # 所以先按 shell 词法解析成数组。
    eval "argv=($args)"

    # 本块该用哪个可执行文件？pix-opt 只能用本地的，其余可走 CE
    use_ce="$CE"; runner="$bin"
    if [ "$blocktool" = "pix-opt" ]; then
      use_ce=0; runner="${PIX_OPT:-}"
      if [ -z "$runner" ]; then
        echo "  － $b 跳过（pix 方言，需要本地 pix-opt）"; skipped=$((skipped+1)); continue
      fi
    fi
    if [ "$use_ce" = "0" ] && [ -z "$runner" ]; then
      echo "  － $b 跳过（缺工具）"; skipped=$((skipped+1)); continue
    fi
    total=$((total+1))
    out="$TMP/$b.out"; err="$TMP/$b.err"
    if [ "$use_ce" = "1" ]; then
      node "$CE_COMPILE" "$ceid" "$f" "${argv[@]}" >"$out" 2>"$err"; rc=$?
    else
      # shellcheck disable=SC2086
      "$runner" "${argv[@]}" "$f" >"$out" 2>"$err"; rc=$?
    fi
    if [ "$rc" -eq 0 ]; then
      echo "  ✓ $b  [$blocktool $args]"; ok=$((ok+1))
      [ "$VERBOSE" = "1" ] && sed 's/^/      /' "$out"
    elif [ "$rc" -eq 2 ] && [ "$use_ce" = "1" ]; then
      echo "  ! $b 网络/服务异常（不计为代码错误）:"
      printf '%s\n' "$(sed 's/^/      /' "$err" | head -3)"   # 诊断常常没有末尾换行，补上
    else
      echo "  ✗ $b  [$blocktool $args] 失败:"
      printf '%s\n' "$(sed 's/^/      /' "$err" | head -10)"
      fail=1
    fi
  done <<< "$manifest"
done

echo ""
echo "MLIR 校验完成：$ok/$total 通过$([ "$skipped" -gt 0 ] && echo "，$skipped 个跳过")。"
exit $fail
