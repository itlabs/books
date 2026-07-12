# 如何构建和阅读这本书

这本书的正文用 Markdown 写在 `chapters/` 里，通过一个构建脚本转成交互式网页放在 `docs/`。

## 目录结构

```
books/
├── README.md          书的封面 + 目录
├── STYLE.md           写作规范（改书/加章时看它）
├── build.js           构建脚本：Markdown → 交互式 HTML
├── chapters/          所有章节的 Markdown 源文件（ch01.md ~ ch17.md、appendix-a~d.md）
└── docs/              构建产物（网页），直接用浏览器打开
    ├── index.html     首页（目录）
    ├── ch01.html ...  各章
    └── assets/        样式 style.css 和运行脚本 runner.js
```

## 怎么阅读

直接用浏览器打开 `docs/index.html` 就行。

- **代码高亮**：所有代码都有语法高亮（Prism.js）。
- **点击运行**：基础章和游戏章的 Python 代码块右上角有绿色 **▶ 运行** 按钮，
  点它就能在浏览器里直接跑真正的 Python（用 Pyodide），还能改代码再运行、点 **↺ 还原** 恢复。
  需要输入的程序（如 input）会弹出输入框。
- **概念图示**：抽象概念（变量、循环、AI 一问一答等）都配了 SVG 插图。
- 第 10—12 章（turtle / pygame / 贪吃蛇）和第 13—17 章（AI）的代码需要真实电脑环境或密钥，
  这些代码块不带运行按钮，请按书里说明在自己电脑上运行。

> 联网说明：网页用到的高亮、Python 运行环境（Pyodide）都从 CDN 加载，
> 所以第一次打开、第一次点"运行"时需要联网，且会稍微等一下（Python 环境要下载）。

## 怎么重新构建（修改内容后）

装好 Node.js 后，在本目录运行：

```bash
npm install marked   # 只需第一次
node build.js
```

它会读取 `chapters/*.md` 和 `README.md`，重新生成 `docs/` 里的全部网页。

## 怎么增改章节

1. 在 `chapters/` 编辑或新增 `.md` 文件（写作请遵循 `STYLE.md`）。
2. 如果是新章节，在 `build.js` 顶部的 `CHAPTERS` 列表里按顺序加一行 `["文件名不带后缀", "导航显示的标题"]`。
3. 重新 `node build.js`。

### 代码块约定（重要）
- ` ```python ` —— 会带"运行"按钮，能在浏览器里跑。**必须是完整、可独立运行**的代码。
- ` ```python-norun ` —— 只高亮、不给运行按钮。用于：故意报错的示范、需要联网/密钥的 AI 代码、turtle/pygame 代码。
- ` ```bash ` —— 终端命令，只高亮。
