// build.js —— 把 chapters/*.md 转成交互式 HTML，输出到 docs/
// 用法：node build.js
const fs = require("fs");
const path = require("path");
const { marked } = require("marked");

const ROOT = __dirname;
const CH_DIR = path.join(ROOT, "chapters");
const OUT_DIR = path.join(ROOT, "docs");

// ---- 章节顺序与标题（用于导航和目录）----
const CHAPTERS = [
  ["ch01", "第 1 章 你好，世界！"],
  ["ch02", "第 2 章 变量与数据类型"],
  ["ch03", "第 3 章 输入与输出"],
  ["ch04", "第 4 章 条件判断"],
  ["ch05", "第 5 章 循环"],
  ["ch06", "第 6 章 列表、字典和函数"],
  ["ch07", "第 7 章 猜数字大作战"],
  ["ch08", "第 8 章 文字冒险游戏"],
  ["ch09", "第 9 章 石头剪刀布"],
  ["ch10", "第 10 章 用海龟画图"],
  ["ch11", "第 11 章 第一个游戏窗口"],
  ["ch12", "第 12 章 贪吃蛇大作战"],
  ["ch13", "第 13 章 AI 是什么"],
  ["ch14", "第 14 章 会聊天的机器人"],
  ["ch15", "第 15 章 给游戏装上 AI"],
  ["ch16", "第 16 章 AI 故事工厂"],
  ["ch17", "第 17 章 更多好玩的 AI 应用"],
  ["appendix-a", "附录 A 安装环境"],
  ["appendix-b", "附录 B 报错速查表"],
  ["appendix-c", "附录 C 继续前进"],
  ["appendix-d", "附录 D 换用别家 AI"],
];

// ---- 自定义 marked 渲染：给代码块加"运行"按钮 ----
// 约定：
//   ```python        -> 可高亮 + 可点击运行（浏览器里跑真 Python）
//   ```python-norun  -> 只高亮，不给运行按钮（如需网络/密钥的 AI 代码）
//   ```bash / 其它    -> 只高亮
let blockId = 0;
const renderer = new marked.Renderer();
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

  const langClass =
    lang === "python-norun"
      ? "language-python"
      : lang
        ? "language-" + lang
        : "language-none";
  const tag =
    lang === "python-norun" ? "Python" : lang === "bash" ? "终端" : lang || "";
  return `<div class="code-block">
  ${tag ? `<div class="code-head"><span class="lang-tag">${tag}</span></div>` : ""}
  <pre class="line-numbers"><code class="${langClass}">${escape(code)}</code></pre>
</div>`;
};

marked.setOptions({ renderer, breaks: false, gfm: true });

// ---- HTML 页面模板 ----
function page(title, bodyHtml, prevLink, nextLink, tocHtml) {
  return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>${title}</title>
<link href="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/themes/prism-tomorrow.min.css" rel="stylesheet">
<link href="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/plugins/line-numbers/prism-line-numbers.min.css" rel="stylesheet">
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
<script src="https://cdn.jsdelivr.net/pyodide/v0.26.2/full/pyodide.js"></script>
<script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-core.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-python.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-bash.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/plugins/line-numbers/prism-line-numbers.min.js"></script>
<script src="assets/runner.js"></script>
</body>
</html>`;
}

// ---- 构建 ----
if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true });
const ASSET_DIR = path.join(OUT_DIR, "assets");
if (!fs.existsSync(ASSET_DIR)) fs.mkdirSync(ASSET_DIR, { recursive: true });

// 目录侧边栏（每页共用）
function buildToc(currentSlug) {
  const items = CHAPTERS.map(([slug, title]) => {
    const cls = slug === currentSlug ? ' class="active"' : "";
    return `<li${cls}><a href="${slug}.html">${title}</a></li>`;
  }).join("\n");
  return `<div class="toc-title"><a href="index.html">📘 目录</a></div><ul class="toc-list">${items}</ul>`;
}

let built = 0;
CHAPTERS.forEach(([slug, title], i) => {
  const mdPath = path.join(CH_DIR, slug + ".md");
  if (!fs.existsSync(mdPath)) {
    console.log("  跳过（还没写）：" + slug);
    return;
  }
  blockId = 0;
  const md = fs.readFileSync(mdPath, "utf8");
  const body = marked.parse(md);
  const prev = CHAPTERS[i - 1];
  const next = CHAPTERS[i + 1];
  const prevLink = prev
    ? `<a class="nav-prev" href="${prev[0]}.html">← ${prev[1]}</a>`
    : "";
  const nextLink = next
    ? `<a class="nav-next" href="${next[0]}.html">${next[1]} →</a>`
    : "";
  fs.writeFileSync(
    path.join(OUT_DIR, slug + ".html"),
    page(title, body, prevLink, nextLink, buildToc(slug)),
  );
  built++;
});

// 首页（目录）
const indexMd = fs.readFileSync(path.join(ROOT, "README.md"), "utf8");
// 把 README 里指向 chapters/xxx.md 的链接改成 xxx.html
const indexBody = marked.parse(
  indexMd.replace(/chapters\/([\w-]+)\.md/g, "$1.html"),
);
fs.writeFileSync(
  path.join(OUT_DIR, "index.html"),
  page("跟 Python 一起玩", indexBody, "", "", buildToc("index")),
);

console.log(`\n构建完成：共生成 ${built} 章 + 首页，输出在 docs/`);
