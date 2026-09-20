// extract-blocks.js —— 从章节 md 里抽取 MLIR 代码块，写到目标目录，供 verify-mlir.sh 用。
//
// 用法：node extract-blocks.js <lang> <outDir> <ch1.md> [ch2.md ...]
//   lang: mlir | mlir-lower | mlir-translate
//   只抽取 ```<lang> 围栏（`mlir-norun` 是片段/伪 IR，不抽）。
//   输出文件名：<章名>__<lang>__<序号>.mlir
// 在 stdout 打印清单，一行一个，制表符分隔：
//   <文件路径>\t<工具>\t<要传给工具的参数>
// 工具取自 RUN 行（mlir-opt / mlir-translate / pix-opt），没有 RUN 行时按围栏的默认工具。
// 参数来自块首 lit 风格的 `// RUN: mlir-opt %s --foo` 行；没有这一行则为空字符串，
// 由 verify-mlir.sh 套用该语言的默认参数。这与 build.js 的 mlirRunArgs() 是同一套约定，
// 保证网页按钮和校验脚本跑的是同一条管线。
const fs = require("fs");
const path = require("path");

const lang = process.argv[2];
const outDir = process.argv[3];
const files = process.argv.slice(4);

const TOOL = {
  mlir: "mlir-opt",
  "mlir-lower": "mlir-opt",
  "mlir-translate": "mlir-translate",
};
const tool = TOOL[lang];
if (!tool) {
  console.error("未知语言：" + lang + "（应为 mlir / mlir-lower / mlir-translate）");
  process.exit(2);
}

// 与 build.js 的 mlirRunArgs() 保持一致：取首行 RUN: 里的工具名，以及它之后、管道之前的参数。
function runInfo(code) {
  const m = code.match(/^[ \t]*\/\/[ \t]*RUN:[ \t]*(.+)$/m);
  if (!m) return { tool, args: "" };
  const line = m[1].split("|")[0];
  const named = /\b(pix-opt|mlir-translate|mlir-opt)\b/.exec(line);
  const actual = named ? named[1] : tool;
  const args = line
    .replace(/%\w+\b/g, "")
    .replace(new RegExp("^.*\\b" + actual + "\\b"), "")
    .trim();
  return { tool: actual, args };
}

const idx = {};
const manifest = [];
for (const mdPath of files) {
  const ch = path.basename(mdPath, ".md");
  const md = fs.readFileSync(mdPath, "utf8");
  // 精确匹配 ```lang 换行 ... ```，lang 后必须紧跟换行（排除 lang-norun）
  const re = new RegExp("```" + lang.replace(/-/g, "\\-") + "\\n([\\s\\S]*?)```", "g");
  let m;
  while ((m = re.exec(md))) {
    const code = m[1];
    idx[ch] = idx[ch] || 0;
    const fname = `${ch}__${lang}__${idx[ch]}.mlir`;
    idx[ch]++;
    const outPath = path.join(outDir, fname);
    fs.writeFileSync(outPath, code);
    const info = runInfo(code);
    manifest.push([outPath, info.tool, info.args].join("\t"));
  }
}
process.stdout.write(manifest.join("\n") + (manifest.length ? "\n" : ""));
