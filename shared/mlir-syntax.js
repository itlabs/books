// mlir-syntax.js —— 给 Prism 加一个 mlir 语言（在 vendor/prism-llvm.min.js 之后加载）。
//
// 为什么不直接用 llvm 的高亮：MLIR 与 LLVM IR 的文本长得像，但有几处关键差别——
//   · 注释是 `//`，不是 LLVM IR 的 `;`
//   · 块标签写成 `^bb0`，`^` 不在 llvm 语法的变量字符集里
//   · 类型多了一大批：index / f32 / bf16 / tensor<> / memref<> / vector<> / !pix.image
// 所以这里派生一个独立的 mlir 语言，**不动** llvm 本身（《深入 LLVM 与 SYCL》里的 LLVM IR
// 代码块仍按 LLVM IR 的规则高亮，免得 `;` 与 `//` 互相干扰）。
(function (Prism) {
  if (!Prism || !Prism.languages || !Prism.languages.llvm) return;

  Prism.languages.mlir = Prism.languages.extend("llvm", {
    // MLIR 用 // 行注释
    comment: /\/\/.*/,
    // 内建类型 + 形状化类型的头部关键字（tensor<4x4xf32> 里的 tensor）。
    // 末尾的 (?!\.) 很关键：`memref<8xf32>` 里的 memref 是类型，
    // 而 `memref.load` 里的 memref 是方言名（应当按关键字走），靠"后面跟不跟点"区分。
    type: {
      pattern:
        /\b(?:index|none|i[1-9]\d*|si[1-9]\d*|ui[1-9]\d*|f16|bf16|f32|f64|f80|f128|tensor|memref|vector|complex|tuple|strided)\b(?!\.)/,
      alias: "class-name",
    },
  });

  // 形状：tensor<4x?x8xf32> 里的 `4x?x8x`（含末尾那个 x，好让后面的 f32 落到 type 规则上）。
  Prism.languages.insertBefore("mlir", "type", {
    shape: {
      pattern: /(?:\b|(?<=<))[\d?]+(?:x[\d?]+)*x(?=[a-z!])/,
      alias: "number",
    },
  });

  // `^bb0`、`^merge` 这类块标签：llvm 的 variable 规则不含 `^`，单独加一条
  Prism.languages.insertBefore("mlir", "variable", {
    "block-label": {
      pattern: /\^[\w$.-]+/,
      alias: "symbol",
    },
  });

  // ODS（TableGen）里的 `$image`、`$result`、`$_builder`：llvm 的 variable 规则
  // 只认 % @ ! # 四个前缀，`$` 不在其中。不加这条的话，`$` 本身不着色，紧跟的
  // 名字被 keyword 规则吃掉——于是 `$border`（引用 attribute）和 `` `border` ``
  // （要打印进 IR 的字面量）同色，assemblyFormat 里这两者常紧挨着出现，看不出区别。
  //
  // 插入点必须是 "label" 而不是 "variable"：Prism 按规则顺序对整串依次扫描，
  // 排在 variable 之前就会先匹配到符号内部的 `$`，把 `@_Z3foo$clone` 劈成
  // variable(`@_Z3foo`) + (`$clone`)。排在 variable 之后、keyword 之前才两头都对。
  Prism.languages.insertBefore("mlir", "label", {
    "ods-arg": {
      pattern: /\$[\w$]+/,
      alias: "variable",
    },
  });

  // `!pix.image<...>`、`#map`、`@my_func`、`%arg0` 交给 llvm 原有的 variable 规则即可，
  // 它已经覆盖了 % @ ! # 四个前缀。

  // ---- ttgir：《深入 Triton》读 TTIR / TTGIR 用（```ttir-norun / ```ttgir-norun）----
  // 在 mlir 之上再派生一层，**不动** mlir 本身（MLIR 书、SYCL 书的高亮保持原样）。
  // 那本书第四部分的主角是 layout 和 op，而 mlir 规则里两者都不显眼：
  //   · `#blocked1`、`#ttg.nvidia_mma<…>`、`#ttig.dpas<…>` 与 `%x` 同走 variable 规则，同色，
  //     读者一眼找不到「这个张量是什么 layout」；
  //   · `tt.load`、`ttg.convert_layout` 这些 op 名不着色，混在 SSA 名和类型中间。
  Prism.languages.ttgir = Prism.languages.extend("mlir", {});
  Prism.languages.insertBefore("ttgir", "variable", {
    // layout 别名与 encoding 属性：#blocked、#blocked1、#mma、#shared、#ttg.blocked、#ttig.dpas ……
    // 只认 # 开头，% 的 SSA 值、! 的类型、@ 的符号仍走 variable。
    layout: {
      pattern: /#[A-Za-z_][\w$.]*/,
      alias: "important",
    },
    // 方言 op：`tt.load`、`ttg.convert_layout`、`arith.addf`、`scf.for`。
    // 要求 `方言.名字` 后面跟空格或 `(`，并且前面不是 # ! % @ 或字母数字，
    // 免得吃掉 `#ttg.blocked`、`!tt.ptr` 里的方言前缀，或属性名 `tt.divisibility =` 里的那段。
    op: {
      pattern: /(^|[^\w#!%@.$])[a-z_][a-z0-9_]*\.[a-z_][\w.]*(?=[ (])(?! =)/m,
      lookbehind: true,
      alias: "function",
    },
    // layout 参数：sizePerThread = [4]、warpsPerCTA = [2, 2]、versionMajor = 2 ……
    // 只匹配 `{ ` 或 `, ` 之后、` = ` 之前的驼峰名，外加几个全小写的 layout 参数（order、parent、
    // version、vec）；不碰 op 的普通属性（它们多半是小写加点，如 tt.divisibility）。
    "layout-param": {
      pattern: /(?<=[{,]\s*)(?:[a-z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*|order|parent|version|vec)(?=\s*=)/,
      alias: "property",
    },
  });
})(typeof Prism !== "undefined" ? Prism : null);
