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
})(typeof Prism !== "undefined" ? Prism : null);
