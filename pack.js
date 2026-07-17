// pack.js —— 一键打包某一本书的离线分享版：先构建，再把 docs/<slug>/ 压成 zip
// 用法：
//   node pack.js <slug>      打包指定的书（如 node pack.js python-teens）
//   npm run pack -- <slug>
const { execSync } = require("child_process");
const fs = require("fs");
const path = require("path");
const os = require("os");

const ROOT = __dirname;
const BOOKS_DIR = path.join(ROOT, "books");

// ---- 选定要打包的书 ----
const slugs = fs
  .readdirSync(BOOKS_DIR, { withFileTypes: true })
  .filter((d) => d.isDirectory() && fs.existsSync(path.join(BOOKS_DIR, d.name, "book.json")))
  .map((d) => d.name);

let slug = process.argv[2];
if (!slug) {
  if (slugs.length === 1) {
    slug = slugs[0]; // 只有一本书时，省略参数也行
  } else {
    console.error(`请指定要打包哪一本：node pack.js <slug>\n可选：${slugs.join(", ")}`);
    process.exit(1);
  }
}
if (!slugs.includes(slug)) {
  console.error(`找不到书：${slug}\n可选：${slugs.join(", ")}`);
  process.exit(1);
}

// 用纯英文的文件夹名和 zip 名：部分解压软件（尤其 Windows 自带、某些手机 App）
// 会把 zip 内的中文路径解成乱码，导致 HTML 找不到 assets/vendor 而丢失排版。
const FOLDER_NAME = `${slug}-offline`; // 解压后读者看到的文件夹名
const ZIP_NAME = `${slug}-offline.zip`;

// 1. 先重新构建这本书，保证打包的是最新内容
console.log(`① 正在构建《${slug}》最新网页……`);
execSync(`node build.js ${slug}`, { cwd: ROOT, stdio: "inherit" });

// 2. 把 docs/<slug>/ 复制到临时目录下的目标文件夹里
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "book-"));
const staged = path.join(tmp, FOLDER_NAME);
fs.cpSync(path.join(ROOT, "docs", slug), staged, { recursive: true });

// 3. 用系统能力压缩（zip 优先，没有则退回 Python zipfile）
const outZip = path.join(ROOT, ZIP_NAME);
if (fs.existsSync(outZip)) fs.rmSync(outZip);

function hasCmd(cmd) {
  try {
    execSync(`command -v ${cmd}`, { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

console.log("② 正在压缩……");
if (hasCmd("zip")) {
  execSync(`zip -rq "${outZip}" "${FOLDER_NAME}"`, { cwd: tmp });
} else if (hasCmd("python3")) {
  // 退路：用 Python 的 zipfile（保留顶层文件夹结构）
  // 写成临时脚本文件再执行，避免命令行里的换行引发的引号问题
  const py = [
    "import zipfile, os",
    `out = ${JSON.stringify(outZip)}`,
    `folder = ${JSON.stringify(FOLDER_NAME)}`,
    'with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:',
    "    for root, _, files in os.walk(folder):",
    "        for f in files:",
    "            full = os.path.join(root, f)",
    "            z.write(full, full)",
  ].join("\n");
  const pyFile = path.join(tmp, "_pack.py");
  fs.writeFileSync(pyFile, py);
  execSync(`python3 "${pyFile}"`, { cwd: tmp });
} else {
  console.error("找不到 zip 也找不到 python3，无法压缩。请手动压缩 docs/ 文件夹。");
  process.exit(1);
}

// 4. 清理临时目录
fs.rmSync(tmp, { recursive: true, force: true });

const sizeKB = Math.round(fs.statSync(outZip).size / 1024);
console.log(`\n✅ 打包完成：${ZIP_NAME}（${sizeKB} KB）`);
console.log(`   位置：${outZip}`);
console.log(`   直接把这个 zip 发给别人即可，解压后双击 index.html 阅读。`);
