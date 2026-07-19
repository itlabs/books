// .extract-blocks.js —— 从章节 md 里抽取指定语言的"完整"代码块，写到目标目录。
// 供 .verify-cuda.sh / .verify-sycl.sh / .verify-triton.sh 共用，不入库（开发工具）。
//
// 用法：node .extract-blocks.js <lang> <outDir> <ch1.md> [ch2.md ...]
//   lang: cuda | cuda-ptx | sycl | triton | cutedsl
//   只抽取 ```<lang> 围栏（不含 -norun 片段）。
//   输出文件名：<章名>__<lang>__<序号>.<ext>，ext 由 lang 决定（.cu/.cpp/.py）。
// 在 stdout 打印每个抽出文件的清单，一行一个：
//   <文件路径>\t<hasMain: 1/0>
const fs = require("fs");
const path = require("path");

const lang = process.argv[2];
const outDir = process.argv[3];
const files = process.argv.slice(4);

const EXT = { cuda: "cu", "cuda-ptx": "cu", sycl: "cpp", triton: "py", cutedsl: "py" };
const ext = EXT[lang];
if (!ext) {
  console.error("未知语言：" + lang);
  process.exit(2);
}

// 判断一个块是否"可运行"（有入口）。
// C/C++：看 int main（正则足够可靠）。
// Python：交给消费方（verify-triton.sh 用 Python ast 精确判断顶层可执行语句），这里一律标 "?"。
function hasEntry(code, lang) {
  if (ext === "cu" || ext === "cpp") return /\bint\s+main\s*\(/.test(code) ? "1" : "0";
  return "?"; // python：由消费方用 ast 判定
}

let idx = {};
const manifest = [];
for (const mdPath of files) {
  const ch = path.basename(mdPath, ".md");
  const md = fs.readFileSync(mdPath, "utf8");
  // 精确匹配 ```lang 换行 ... ```，且 lang 后必须紧跟换行（排除 lang-norun）
  const re = new RegExp("```" + lang.replace(/[-]/g, "\\-") + "\\n([\\s\\S]*?)```", "g");
  let m;
  while ((m = re.exec(md))) {
    const code = m[1];
    idx[ch] = (idx[ch] || 0);
    const fname = `${ch}__${lang}__${idx[ch]}.${ext}`;
    idx[ch]++;
    const outPath = path.join(outDir, fname);
    fs.writeFileSync(outPath, code);
    manifest.push(outPath + "\t" + hasEntry(code, lang));
  }
}
process.stdout.write(manifest.join("\n") + (manifest.length ? "\n" : ""));
