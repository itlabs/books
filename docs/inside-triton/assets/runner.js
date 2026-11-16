// runner.js —— 在浏览器里运行 Python（Pyodide），支持 input() 交互
let pyodideReady = null;

async function getPyodide() {
  if (typeof loadPyodide === "undefined") {
    // Pyodide 从 CDN 加载，没联网 / 访问不了 CDN 时会走到这里
    throw new Error(
      "无法加载 Python 运行环境（需要联网）。你可以先阅读代码，或按附录 A 在自己电脑上运行。",
    );
  }
  if (!pyodideReady) {
    pyodideReady = loadPyodide().then((py) => {
      // 用一个可替换的函数处理 print 输出
      py.setStdout({ batched: (s) => window.__emit && window.__emit(s + "\n") });
      py.setStderr({ batched: (s) => window.__emit && window.__emit(s + "\n") });
      return py;
    });
  }
  return pyodideReady;
}

// 让代码区可编辑，同时保持高亮（编辑时去掉高亮，失焦后恢复）
function makeEditable(codeEl) {
  codeEl.setAttribute("contenteditable", "true");
  codeEl.setAttribute("spellcheck", "false");
}

function getCode(codeEl) {
  // textContent 保留纯文本（去掉高亮生成的标签）
  return codeEl.textContent;
}

async function runCode(codeEl, outEl) {
  outEl.hidden = false;
  outEl.className = "output";
  outEl.textContent = "正在启动 Python……第一次会慢一点，请稍等。\n";

  const py = await getPyodide();
  outEl.textContent = "";

  // 把 print 输出实时写进输出框
  window.__emit = (text) => {
    outEl.textContent += text;
    outEl.scrollTop = outEl.scrollHeight;
  };

  // 处理 input()：弹出浏览器输入框，并把"提示+回答"回显到输出区
  const jsInput = (promptText) => {
    const answer = window.prompt(promptText || "请输入：");
    const shown = (promptText || "") + (answer === null ? "" : answer);
    outEl.textContent += shown + "\n";
    return answer === null ? "" : answer;
  };
  py.globals.set("__js_input", jsInput);

  const code = getCode(codeEl);
  try {
    // 覆盖内置 input，改用我们的浏览器输入框
    await py.runPythonAsync(
      "import builtins\n" +
        "builtins.input = lambda prompt='': __js_input(prompt)\n",
    );
    await py.runPythonAsync(code);
    if (outEl.textContent.trim() === "") {
      outEl.textContent = "（程序运行完毕，没有打印任何内容）";
    }
  } catch (err) {
    outEl.className = "output error";
    outEl.textContent += "\n" + String(err.message || err);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  // 让所有可运行代码块可编辑
  document
    .querySelectorAll(".runnable code.language-python")
    .forEach(makeEditable);

  document.querySelectorAll(".run-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const id = btn.dataset.target;
      const codeEl = document.getElementById(id);
      const outEl = document.getElementById(id + "-out");
      btn.disabled = true;
      const label = btn.textContent;
      btn.textContent = "运行中…";
      try {
        await runCode(codeEl, outEl);
      } finally {
        btn.disabled = false;
        btn.textContent = label;
      }
    });
  });

  // 还原按钮：把代码恢复成书里的原始版本
  document.querySelectorAll(".reset-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = btn.dataset.target;
      const codeEl = document.getElementById(id);
      const outEl = document.getElementById(id + "-out");
      codeEl.textContent = decodeURIComponent(codeEl.dataset.original);
      if (window.Prism) Prism.highlightElement(codeEl);
      outEl.hidden = true;
      outEl.textContent = "";
    });
  });

  // 侧边栏开关（手机）
  const toggle = document.getElementById("menu-toggle");
  const sidebar = document.getElementById("sidebar");
  if (toggle) {
    toggle.addEventListener("click", () => sidebar.classList.toggle("open"));
  }
});
