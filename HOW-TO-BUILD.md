# 如何构建和阅读

本仓库是一个**多书**工程：每本书的正文用 Markdown 写在 `books/<书名>/chapters/` 里，
通过统一的构建脚本转成交互式网页，输出到 `docs/`，可直接用浏览器打开，也能发布到 GitHub Pages。

## 目录结构

```
books/
├── build.js            构建脚本：遍历 books/ 下每本书，Markdown → 交互式 HTML
├── pack.js             打包脚本：把某一本压成离线 zip
├── STYLE.md            写作规范（改书/加章时看它）
├── shared/             所有书共用的资源
│   ├── style.css       样式
│   ├── runner.js       浏览器内运行 Python 的脚本
│   └── vendor/         Prism.js 高亮（本地，离线可用）
├── books/              每本书一个子目录
│   └── python-teens/
│       ├── book.json   书的元数据：标题、章节顺序与导航标题
│       ├── README.md   这本书的封面 + 目录页
│       └── chapters/   章节 Markdown 源（ch01.md … appendix-d.md）
└── docs/               构建产物（发布到 GitHub Pages）
    ├── index.html      书架首页（自动列出所有书）
    └── python-teens/   每本书一个子目录
        ├── index.html  本书目录页
        ├── ch01.html … 各章
        ├── assets/     从 shared/ 复制来的 style.css / runner.js
        └── vendor/     从 shared/vendor/ 复制来的高亮文件
```

> GitHub Pages 指向 `docs/`；打开站点先看到书架首页，点进任意一本即可阅读。

## 怎么构建

装好 Node.js 后，在本目录运行：

```bash
npm install          # 只需第一次
node build.js        # 构建全部书 + 书架首页
node build.js python-teens   # 只构建某一本（书架首页仍会刷新）
```

## 怎么打包成离线 zip 分享

```bash
node pack.js python-teens    # 生成 python-teens-offline.zip
```

只有一本书时可省略书名参数。zip 解压后双击 `index.html` 即可离线阅读。

## 怎么新增一本书

1. 在 `books/` 下新建一个目录，如 `books/<新书名>/`。
2. 放入三样东西：
   - `book.json` —— 抄 `python-teens/book.json` 改：`slug`、`title`、`subtitle`、
     `description`，以及 `chapters` 数组（每项 `["文件名不带后缀", "导航显示的标题"]`，按顺序）。
   - `README.md` —— 本书封面 + 目录（指向 `chapters/xxx.md` 的链接会自动改写成 `.html`）。
   - `chapters/` —— 章节 Markdown 源文件。
3. 运行 `node build.js`，新书会自动出现在书架首页上。

样式、代码运行、语法高亮、离线打包等能力所有书自动共用，无需重复配置。

## 怎么给一本书增改章节

1. 在该书的 `chapters/` 编辑或新增 `.md` 文件（写作请遵循 `STYLE.md`）。
2. 如果是新章节，在该书的 `book.json` 的 `chapters` 数组里按顺序加一行。
3. 重新 `node build.js`（或 `node build.js <书名>`）。

### 代码块约定（重要）
- ` ```python ` —— 会带"运行"按钮，能在浏览器里跑。**必须是完整、可独立运行**的代码。
- ` ```python-norun ` —— 只高亮、不给运行按钮。用于：故意报错的示范、需要联网/密钥的 AI 代码、turtle/pygame 代码。
- ` ```bash ` —— 终端命令，只高亮。
