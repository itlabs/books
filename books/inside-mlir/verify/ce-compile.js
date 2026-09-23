// ce-compile.js —— 没有本地 LLVM/MLIR 时的兜底：把一个 .mlir 文件交给 Compiler Explorer
// 的 trunk 版 mlir-opt / mlir-translate 跑一遍，用它的退出码判定通过与否。
//
// 用法：node ce-compile.js <compilerId> <file.mlir> [传给工具的参数 …]
//   compilerId: mliropttrunk | mlirtranslatetrunk（见 https://godbolt.org/api/compilers/mlir）
// 成功（工具退出码 0）时把结果 IR 打到 stdout，本进程退出 0；
// 失败时把工具的诊断打到 stderr，本进程退出 1。网络/服务异常退出 2（与"代码错了"区分开）。
//
// 注意：这会把代码块内容发到 godbolt.org。本书代码块都是公开示例，没有敏感内容；
// 若不希望走网络，请装好本地 mlir-opt 并直接用 verify-mlir.sh 的本地路径。
const fs = require("fs");
const https = require("https");

const [compiler, file, ...args] = process.argv.slice(2);
if (!compiler || !file) {
  console.error("用法：node ce-compile.js <compilerId> <file.mlir> [args…]");
  process.exit(2);
}

const payload = JSON.stringify({
  source: fs.readFileSync(file, "utf8"),
  options: { userArguments: args.join(" "), filters: {} },
});

const req = https.request(
  {
    hostname: "godbolt.org",
    path: `/api/compiler/${encodeURIComponent(compiler)}/compile`,
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      // 少了 Accept 会返回纯文本而不是 JSON
      Accept: "application/json",
      "Content-Length": Buffer.byteLength(payload),
    },
    timeout: 60000,
  },
  (res) => {
    let body = "";
    res.on("data", (d) => (body += d));
    res.on("end", () => {
      let j;
      try {
        j = JSON.parse(body);
      } catch {
        console.error("CE 返回了非 JSON（HTTP " + res.statusCode + "）：" + body.slice(0, 300));
        process.exit(2);
      }
      const text = (arr) => (arr || []).map((x) => x.text).join("\n");
      if (j.code === 0) {
        // 成功时 stderr 往往也有内容（--mlir-print-ir-after-all、pass 统计、诊断 note），
        // 原样转到我们自己的 stderr，别丢掉；判定成败只看退出码。
        const err = text(j.stderr);
        if (err) process.stderr.write(err + "\n");
        process.stdout.write(text(j.asm) + "\n");
        process.exit(0);
      }
      process.stderr.write(text(j.stderr) || "（工具没有输出诊断，退出码 " + j.code + "）\n");
      process.exit(1);
    });
  },
);
req.on("timeout", () => {
  req.destroy();
  console.error("CE 请求超时");
  process.exit(2);
});
req.on("error", (e) => {
  console.error("CE 请求失败：" + e.message);
  process.exit(2);
});
req.end(payload);
