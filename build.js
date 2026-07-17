// build.js —— 把 books/<slug>/chapters/*.md 转成交互式 HTML，输出到 docs/<slug>/
// 用法：
//   node build.js            构建全部书 + 书架首页
//   node build.js <slug>     只构建某一本（仍会刷新书架首页）
const fs = require("fs");
const path = require("path");
const { marked } = require("marked");

const ROOT = __dirname;
const BOOKS_DIR = path.join(ROOT, "books");
const OUT_DIR = path.join(ROOT, "docs");
const SHARED_DIR = path.join(ROOT, "shared"); // 各书共用的 style.css / runner.js / vendor/

// ---- 读取所有书的元数据（books/<slug>/book.json）----
function loadBooks() {
  return fs
    .readdirSync(BOOKS_DIR, { withFileTypes: true })
    .filter((d) => d.isDirectory())
    .map((d) => d.name)
    .filter((slug) =>
      fs.existsSync(path.join(BOOKS_DIR, slug, "book.json")),
    )
    .map((slug) => {
      const meta = JSON.parse(
        fs.readFileSync(path.join(BOOKS_DIR, slug, "book.json"), "utf8"),
      );
      meta.slug = meta.slug || slug;
      meta.dir = path.join(BOOKS_DIR, slug);
      return meta;
    });
}

// ---- 自定义 marked 渲染：给代码块加"运行"按钮 ----
// 约定（Python 书）：
//   ```python        -> 可高亮 + 可点击运行（浏览器里跑真 Python）
//   ```python-norun  -> 只高亮，不给运行按钮（如需网络/密钥的 AI 代码）
// 约定（C++ 书）：
//   ```cpp           -> 高亮 + "🔗 在 Compiler Explorer 打开" + 附本地编译命令
//   ```cpp-asm       -> 同上，但 godbolt 默认 -O2 并展开汇编面板（讲"编译器处理"用）
//   ```cpp-norun     -> 只高亮（片段、故意报错示范、伪代码）
//   ```bash / 其它    -> 只高亮
let blockId = 0;
const renderer = new marked.Renderer();

// 把整段 C++ 源码打包进一个 Compiler Explorer 分享链接。
// 用 godbolt 的 clientstate（base64 的 JSON）方案：点开即带上代码、编译器、参数、面板。
function godboltUrl(code, { asm = false } = {}) {
  const compiler = "g142"; // x86-64 gcc 14.2
  const options = "-std=c++23 -O2 -Wall" + (asm ? "" : "");
  // 汇编视角展开 compiler 面板；普通视角额外展开一个 execution（运行）面板
  const state = {
    sessions: [
      {
        id: 1,
        language: "c++",
        source: code,
        compilers: [{ id: compiler, options }],
        executors: asm
          ? []
          : [{ compiler: { id: compiler, options }, wrap: true }],
      },
    ],
  };
  const b64 = Buffer.from(JSON.stringify(state), "utf8").toString("base64");
  return "https://godbolt.org/clientstate/" + encodeURIComponent(b64);
}

renderer.code = function (token) {
  // marked v12+ 传入的是 token 对象 {text, lang, escaped}
  const code = typeof token === "string" ? token : token.text;
  const lang = ((typeof token === "string" ? arguments[1] : token.lang) || "")
    .trim();
  const escape = (s) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  if (lang === "python") {
    const id = "code" + blockId++;
    return `<div class="code-block runnable">
  <div class="code-head"><span class="lang-tag">Python</span>
    <button class="run-btn" data-target="${id}">▶ 运行</button>
    <button class="reset-btn" data-target="${id}">↺ 还原</button></div>
  <pre class="line-numbers"><code class="language-python" id="${id}" data-original="${encodeURIComponent(code)}">${escape(code)}</code></pre>
  <div class="output" id="${id}-out" hidden></div>
</div>`;
  }

  if (lang === "cpp" || lang === "cpp-asm") {
    const asm = lang === "cpp-asm";
    const url = godboltUrl(code, { asm });
    const label = asm ? "🔬 看汇编 / 编译器处理" : "🔗 在 Compiler Explorer 打开";
    // 本地编译命令：把源码存成 main.cpp 后编译运行
    const buildCmd = "g++ -std=c++23 -O2 -Wall main.cpp && ./a.out";
    return `<div class="code-block cpp">
  <div class="code-head"><span class="lang-tag">C++</span>
    <a class="ce-btn" href="${url}" target="_blank" rel="noopener">${label}</a></div>
  <pre class="line-numbers"><code class="language-cpp">${escape(code)}</code></pre>
  <div class="local-cmd"><span class="local-cmd-label">本地编译运行：</span><code>${escape(buildCmd)}</code></div>
</div>`;
  }

  const langClass =
    lang === "python-norun"
      ? "language-python"
      : lang === "cpp-norun"
        ? "language-cpp"
        : lang
          ? "language-" + lang
          : "language-none";
  const tag =
    lang === "python-norun"
      ? "Python"
      : lang === "cpp-norun"
        ? "C++"
        : lang === "bash"
          ? "终端"
          : lang || "";
  return `<div class="code-block">
  ${tag ? `<div class="code-head"><span class="lang-tag">${tag}</span></div>` : ""}
  <pre class="line-numbers"><code class="${langClass}">${escape(code)}</code></pre>
</div>`;
};

marked.setOptions({ renderer, breaks: false, gfm: true });

// ---- HTML 页面模板（书内页面：assets/vendor 都在本书目录下）----
function page(title, bodyHtml, prevLink, nextLink, tocHtml) {
  return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>${title}</title>
<link href="vendor/prism-tomorrow.min.css" rel="stylesheet">
<link href="vendor/prism-line-numbers.min.css" rel="stylesheet">
<link href="assets/style.css" rel="stylesheet">
</head>
<body>
<button id="menu-toggle" aria-label="目录">☰ 目录</button>
<nav id="sidebar">${tocHtml}</nav>
<main>
<article>
${bodyHtml}
<div class="chapter-nav">
  ${prevLink || "<span></span>"}
  ${nextLink || "<span></span>"}
</div>
</article>
</main>
<!-- 语法高亮：本地文件，离线可用 -->
<script src="vendor/prism-core.min.js"></script>
<script src="vendor/prism-python.min.js"></script>
<!-- C++ 高亮：cpp 依赖 c，c 依赖 clike，须按此顺序加载 -->
<script src="vendor/prism-clike.min.js"></script>
<script src="vendor/prism-c.min.js"></script>
<script src="vendor/prism-cpp.min.js"></script>
<script src="vendor/prism-bash.min.js"></script>
<script src="vendor/prism-line-numbers.min.js"></script>
<!-- Pyodide（浏览器内运行 Python）：需联网从 CDN 加载，仅"▶ 运行"按钮用到 -->
<script src="https://cdn.jsdelivr.net/pyodide/v0.26.2/full/pyodide.js"></script>
<script src="assets/runner.js"></script>
</body>
</html>`;
}

// 目录侧边栏（每本书内共用）
function buildToc(book, currentSlug) {
  const items = book.chapters
    .map(([slug, title]) => {
      const cls = slug === currentSlug ? ' class="active"' : "";
      return `<li${cls}><a href="${slug}.html">${title}</a></li>`;
    })
    .join("\n");
  return `<div class="toc-title"><a href="index.html">📘 目录</a></div>
<div class="toc-home"><a href="../index.html">← 返回书架</a></div>
<ul class="toc-list">${items}</ul>`;
}

// ---- 构建单本书 ----
function buildBook(book) {
  const outDir = path.join(OUT_DIR, book.slug);
  const assetDir = path.join(outDir, "assets");
  const vendorDir = path.join(outDir, "vendor");
  fs.mkdirSync(assetDir, { recursive: true });
  fs.mkdirSync(vendorDir, { recursive: true });

  // 复制共享资源到本书输出目录（每次构建都刷新，离线自洽）
  for (const f of fs.readdirSync(SHARED_DIR)) {
    const src = path.join(SHARED_DIR, f);
    if (fs.statSync(src).isFile()) fs.copyFileSync(src, path.join(assetDir, f));
  }
  for (const f of fs.readdirSync(path.join(SHARED_DIR, "vendor"))) {
    fs.copyFileSync(
      path.join(SHARED_DIR, "vendor", f),
      path.join(vendorDir, f),
    );
  }

  const chDir = path.join(book.dir, "chapters");
  let built = 0;
  book.chapters.forEach(([slug, title], i) => {
    const mdPath = path.join(chDir, slug + ".md");
    if (!fs.existsSync(mdPath)) {
      console.log("  跳过（还没写）：" + book.slug + "/" + slug);
      return;
    }
    blockId = 0;
    const md = fs.readFileSync(mdPath, "utf8");
    const body = marked.parse(md);
    const prev = book.chapters[i - 1];
    const next = book.chapters[i + 1];
    const prevLink = prev
      ? `<a class="nav-prev" href="${prev[0]}.html">← ${prev[1]}</a>`
      : "";
    const nextLink = next
      ? `<a class="nav-next" href="${next[0]}.html">${next[1]} →</a>`
      : "";
    fs.writeFileSync(
      path.join(outDir, slug + ".html"),
      page(title, body, prevLink, nextLink, buildToc(book, slug)),
    );
    built++;
  });

  // 本书首页（目录）：来自 books/<slug>/README.md
  const readmePath = path.join(book.dir, "README.md");
  if (fs.existsSync(readmePath)) {
    const indexMd = fs.readFileSync(readmePath, "utf8");
    // 把 README 里指向 chapters/xxx.md 的链接改成 xxx.html
    const indexBody = marked.parse(
      indexMd.replace(/chapters\/([\w-]+)\.md/g, "$1.html"),
    );
    fs.writeFileSync(
      path.join(outDir, "index.html"),
      page(book.title, indexBody, "", "", buildToc(book, "index")),
    );
  }

  // 离线阅读说明（随书打包）
  fs.writeFileSync(
    path.join(outDir, "READ-ME.txt"),
    `《${book.title}》${book.subtitle ? "——" + book.subtitle : ""}\n` +
      "=".repeat(40) +
      "\n\n" +
      "【怎么看这本书】\n" +
      "双击打开 index.html，就能在浏览器里阅读整本书。左侧是目录，可点章节跳转。\n\n" +
      "【功能说明】\n" +
      "· 排版、文字、代码高亮、插图 —— 完全离线，不联网也能看。\n" +
      "· 代码块右上角的绿色\"▶ 运行\"按钮 —— 需要联网（临时下载浏览器版 Python）。\n" +
      "  第一次点击要等十几秒，请耐心。断网时点它会提示需要联网，不影响阅读。\n\n" +
      "【推荐用最新版 Chrome / Edge 打开，阅读体验最好。】\n",
  );

  console.log(`  ✓ ${book.slug}：${built} 章 + 首页`);
  return built;
}

// ---- 书架首页 docs/index.html ----
function buildShelf(books) {
  const cards = books
    .map((b) => {
      const sub = b.subtitle ? `<p class="shelf-sub">${b.subtitle}</p>` : "";
      const desc = b.description
        ? `<p class="shelf-desc">${b.description}</p>`
        : "";
      return `<a class="shelf-card" href="${b.slug}/index.html">
  <h2>${b.title}</h2>
  ${sub}
  ${desc}
</a>`;
    })
    .join("\n");

  const html = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>书架</title>
<style>
  body { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
    max-width: 860px; margin: 0 auto; padding: 48px 20px; color: #24292f;
    background: #f6f8fa; }
  h1 { text-align: center; margin-bottom: 8px; }
  .shelf-intro { text-align: center; color: #656d76; margin-bottom: 40px; }
  .shelf-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 20px; }
  .shelf-card { display: block; padding: 24px; border-radius: 12px; background: #fff;
    border: 1px solid #d0d7de; text-decoration: none; color: inherit;
    transition: box-shadow .15s, transform .15s; }
  .shelf-card:hover { box-shadow: 0 6px 20px rgba(0,0,0,.1); transform: translateY(-2px); }
  .shelf-card h2 { margin: 0 0 6px; font-size: 20px; color: #0969da; }
  .shelf-sub { margin: 0 0 10px; font-weight: 600; color: #24292f; }
  .shelf-desc { margin: 0; color: #656d76; font-size: 14px; }
</style>
</head>
<body>
<h1>📚 书架</h1>
<p class="shelf-intro">点击任意一本，开始阅读</p>
<div class="shelf-grid">
${cards}
</div>
</body>
</html>`;
  fs.writeFileSync(path.join(OUT_DIR, "index.html"), html);
  // GitHub Pages：禁用 Jekyll，保证 assets/vendor 原样发布
  fs.writeFileSync(path.join(OUT_DIR, ".nojekyll"), "");
}

// ---- 主流程 ----
fs.mkdirSync(OUT_DIR, { recursive: true });
const allBooks = loadBooks();
const only = process.argv[2];
const target = only ? allBooks.filter((b) => b.slug === only) : allBooks;

if (only && target.length === 0) {
  console.error(`找不到书：${only}（可选：${allBooks.map((b) => b.slug).join(", ")}）`);
  process.exit(1);
}

console.log("正在构建：");
target.forEach(buildBook);
buildShelf(allBooks); // 书架始终列出全部书

console.log(`\n构建完成：${target.length} 本书 + 书架首页，输出在 docs/`);
