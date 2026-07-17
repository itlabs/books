# 附录 D 换用别家 AI：Bedrock / Gemini / OpenAI / 国产大模型

正文里我们一直用 Anthropic 的 **Claude**。但世界上厉害的 AI 不止一家——美国的 OpenAI（ChatGPT 背后的公司）、Google 的 **Gemini**，中国的 **DeepSeek（深度求索）**、**智谱 GLM**、**Kimi（月之暗面）** 都很强。这个附录教你：**怎么把书里的代码，改成用这些 AI。**

好消息是：**换 AI 通常只要改几行**，你学的编程思路完全不用变。

> ⚠️ **先读这条**：申请任何一家的 API 密钥、可能产生的费用，都请**和家长或老师一起**完成。密钥永远是**秘密**，绝不写进代码、绝不外传——这条规矩对每一家都一样。所有密钥都用正文教过的**环境变量**方式保存。

> 💡 **关于型号名字会过时**：AI 更新很快，下面写的型号名（如 `gpt-4o-mini`、`gemini-2.5-flash`）是本书写作时的。等你真正动手时，**去对应官网的文档查一下当前最新、最便宜的型号名**再填进去。base_url（服务器地址）则比较稳定。

## 一个关键发现：很多家共用同一套写法

这是本附录最重要的一句话：**OpenAI、DeepSeek、智谱 GLM、Kimi，甚至 Google Gemini，都能用同一个叫 `openai` 的 Python 工具包来调用。**

为什么？因为这些公司都提供了一种"和 OpenAI 长得一样"的接口（业内叫 **OpenAI 兼容接口**）。于是你只要换三样东西：

<div class="concept">
<svg viewBox="0 0 560 200" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">
  <rect x="180" y="15" width="200" height="34" rx="8" fill="#e8f0fa" stroke="#2f6db3"/>
  <text x="280" y="37" text-anchor="middle" font-size="14" fill="#2f6db3" font-weight="bold">同一套 openai 代码</text>

  <rect x="30" y="90" width="150" height="90" rx="10" fill="#fff7ec" stroke="#f0a94b" stroke-width="2"/>
  <text x="105" y="112" text-anchor="middle" font-size="13" fill="#a5762b" font-weight="bold">只改 3 样东西</text>
  <text x="105" y="134" text-anchor="middle" font-size="12" fill="#6b5638">① base_url 地址</text>
  <text x="105" y="153" text-anchor="middle" font-size="12" fill="#6b5638">② api_key 密钥</text>
  <text x="105" y="172" text-anchor="middle" font-size="12" fill="#6b5638">③ model 型号名</text>

  <rect x="220" y="80" width="150" height="26" rx="6" fill="#eafaef" stroke="#3f9b5c"/>
  <text x="295" y="98" text-anchor="middle" font-size="12" fill="#2f7a45">→ OpenAI</text>
  <rect x="220" y="112" width="150" height="26" rx="6" fill="#eafaef" stroke="#3f9b5c"/>
  <text x="295" y="130" text-anchor="middle" font-size="12" fill="#2f7a45">→ DeepSeek</text>
  <rect x="220" y="144" width="150" height="26" rx="6" fill="#eafaef" stroke="#3f9b5c"/>
  <text x="295" y="162" text-anchor="middle" font-size="12" fill="#2f7a45">→ GLM / Kimi / Gemini</text>

  <rect x="410" y="112" width="140" height="26" rx="6" fill="#fbe6dc" stroke="#e06a3b"/>
  <text x="480" y="130" text-anchor="middle" font-size="12" fill="#a53f1c">Claude / Bedrock 另讲</text>
  <path d="M180 110 L218 100" stroke="#666" stroke-width="1.5" marker-end="url(#dd)"/>
  <path d="M180 130 L218 125" stroke="#666" stroke-width="1.5" marker-end="url(#dd)"/>
  <defs><marker id="dd" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#666"/></marker></defs>
</svg>
<div class="caption">图 D-1　OpenAI、DeepSeek、GLM、Kimi、Gemini 共用同一套 openai 代码，只改地址、密钥、型号三处</div>
</div>

先装好这个工具包：

```bash
pip install openai
```

## 通用模板：一份代码，换着用

下面这份代码是"万能模板"。**它的骨架永远不变**，你只需要按后面各家的表格，替换 `base_url`、`api_key`、`model` 三处。

```python-norun
# universal_ai.py —— 换 base_url/密钥/型号，就能用不同家的 AI
import os
from openai import OpenAI

client = OpenAI(
    base_url="各家的地址填这里",                 # ① 换成对应服务商的地址
    api_key=os.environ.get("MY_AI_KEY"),        # ② 从环境变量读密钥
)

response = client.chat.completions.create(
    model="各家的型号名填这里",                   # ③ 换成对应型号
    messages=[
        {"role": "user", "content": "用一句话介绍一下你自己。"}
    ],
)

# openai 这套取回答的写法，比 Claude 稍简单一点：
print(response.choices[0].message.content)
```

> 💡 **注意和 Claude 的两点不同**：
> 1. 调用的是 `client.chat.completions.create(...)`（Claude 是 `client.messages.create(...)`）。
> 2. 取回答用 `response.choices[0].message.content`（Claude 要遍历 `response.content` 取 `text`）。
> `messages` 列表的写法（`{"role": ..., "content": ...}`）两边是一样的，多轮对话、`system` 人设的思路也一样——你正文学的全都用得上。

## 各家怎么填？（速查表）

下表是各家的地址和型号。**密钥都去对应官网申请**（记得和家长/老师一起）。

| 服务商 | base_url（地址） | 一个型号名（会更新，以官网为准） | 去哪申请密钥 |
|---|---|---|---|
| **DeepSeek 深度求索** | `https://api.deepseek.com` | `deepseek-chat` | platform.deepseek.com |
| **智谱 GLM** | `https://open.bigmodel.cn/api/paas/v4/` | `glm-4-flash`（有免费型号！） | open.bigmodel.cn |
| **Kimi 月之暗面** | `https://api.moonshot.cn/v1` | `moonshot-v1-8k` | platform.moonshot.cn |
| **OpenAI** | 不用填（默认就是） | `gpt-4o-mini` | platform.openai.com |
| **Google Gemini** | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-2.5-flash` | Google AI Studio |

下面挑几家给出完整例子。

### DeepSeek（国内，性价比高）

```python-norun
# deepseek_demo.py
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "用一句话夸夸正在学编程的我。"}],
)
print(response.choices[0].message.content)
```

运行前先设置密钥（在同一个终端窗口）：

```bash
set DEEPSEEK_API_KEY=你的DeepSeek密钥
```

Mac/Linux 用 `export DEEPSEEK_API_KEY=...`。

### 智谱 GLM（有免费型号，最适合练手）

智谱有个 `glm-4-flash` 型号常常是**免费**的，特别适合学生反复练习，不用担心花钱。

```python-norun
# glm_demo.py
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    api_key=os.environ.get("GLM_API_KEY"),
)

response = client.chat.completions.create(
    model="glm-4-flash",                       # 常见的免费型号
    messages=[{"role": "user", "content": "给我讲一个关于程序员的冷笑话。"}],
)
print(response.choices[0].message.content)
```

### Kimi（月之暗面）

```python-norun
# kimi_demo.py
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.moonshot.cn/v1",
    api_key=os.environ.get("KIMI_API_KEY"),
)

response = client.chat.completions.create(
    model="moonshot-v1-8k",
    messages=[{"role": "user", "content": "帮我给一只橘猫起 5 个可爱的名字。"}],
)
print(response.choices[0].message.content)
```

### Google Gemini（从 Google AI Studio 拿密钥）

Gemini 的密钥在 **Google AI Studio**（网址 `https://aistudio.google.com`）申请，通常有免费额度。申请步骤（和家长/老师一起）：

1. 用 Google 账号登录 `https://aistudio.google.com`。
2. 找到 "Get API key" / "API 密钥"，创建一个。
3. 复制保存好那串密钥。

Gemini 也支持 OpenAI 兼容写法，所以还是那套模板：

```python-norun
# gemini_demo.py
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.environ.get("GEMINI_API_KEY"),
)

response = client.chat.completions.create(
    model="gemini-2.5-flash",                  # 型号会更新，以官网为准
    messages=[{"role": "user", "content": "用小学生能懂的话解释什么是彩虹。"}],
)
print(response.choices[0].message.content)
```

### OpenAI（ChatGPT 官方接口）

OpenAI 是这套写法的"原厂"，所以 `base_url` 都不用填：

```python-norun
# openai_demo.py
import os
from openai import OpenAI

# 不写 base_url，默认就是 OpenAI 官方地址
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o-mini",                       # 又便宜又够用的型号
    messages=[{"role": "user", "content": "推荐三本适合初中生的科普书。"}],
)
print(response.choices[0].message.content)
```

## 把正文的项目改成用别家 AI

正文第 14—17 章的聊天机器人、AI 出题、故事工厂等，都是围绕 Claude 的 `client.messages.create` 写的。想改成上面这些 AI，只要做这几处"翻译"：

| 正文里的 Claude 写法 | 换成 openai 那套 |
|---|---|
| `import anthropic` | `from openai import OpenAI` |
| `client = anthropic.Anthropic()` | `client = OpenAI(base_url=..., api_key=...)` |
| `client.messages.create(...)` | `client.chat.completions.create(...)` |
| `max_tokens=300`（必填） | 可省略；如需限制写 `max_tokens=300` |
| `system="人设"`（单独参数） | 放进 messages：`{"role": "system", "content": "人设"}` |
| 遍历 `response.content` 取 `text` | `response.choices[0].message.content` |

> 💡 注意 `system` 人设的位置不同：Claude 是单独的 `system=` 参数；openai 这套是把它作为 `messages` 列表里的**第一条**消息，`"role"` 写 `"system"`。多轮对话记忆的思路（把每句话 append 进 messages）两边完全一样。

## 进阶：AWS Bedrock（用亚马逊云调用 Claude）

**AWS Bedrock** 是亚马逊云提供的 AI 服务，通过它也能用上 Claude（以及其他厂商的模型）。很多公司因为已经在用亚马逊云，就顺便用 Bedrock 调 Claude。

Bedrock 的登录方式和前面都不同——它不用简单的 API 密钥，而是用**亚马逊云账号（AWS 账号）的凭证**，配置比较复杂，通常是公司或懂云服务的大人来搭。这里给你看一眼代码长什么样，理解"原来还能这样连"，具体配置请让熟悉 AWS 的老师或家长帮忙。

Bedrock 用的是我们正文的老朋友 `anthropic` 工具包，但换一个专门的客户端：

```python-norun
# bedrock_demo.py —— 通过 AWS Bedrock 调用 Claude
# 需要先装：pip install "anthropic[bedrock]"
# 并且事先配置好 AWS 账号凭证（请让懂 AWS 的大人协助）
from anthropic import AnthropicBedrock

client = AnthropicBedrock(aws_region="us-east-1")   # 你所在的区域

response = client.messages.create(
    model="anthropic.claude-haiku-4-5",             # Bedrock 上的型号名带 anthropic. 前缀
    max_tokens=300,
    messages=[{"role": "user", "content": "你好，请介绍一下自己。"}],
)

for block in response.content:
    if block.type == "text":
        print(block.text)
```

> 💡 看出来了吗？Bedrock 上取回答的写法（遍历 `response.content` 取 `text`）和正文的 Claude **一模一样**——因为它本来就是 Claude，只是换了个"入口"。区别只在：客户端换成 `AnthropicBedrock`、要指定 `aws_region`、型号名带 `anthropic.` 前缀、登录靠 AWS 账号凭证。

## 该选哪一家？

给你几条实用建议：

- **想免费、零花费地练手**：优先试 **智谱 GLM 的 `glm-4-flash`**（常有免费额度），或用有免费额度的 **Gemini**。
- **在中国大陆、图方便稳定**：**DeepSeek、GLM、Kimi** 这些国产的访问通常更顺畅，文档也是中文。
- **想用 ChatGPT 同款**：用 **OpenAI**。
- **家里/学校已经在用亚马逊云**：可以考虑 **Bedrock**。
- **就想用书里的 Claude**：那就继续用正文的 `anthropic` 写法，最简单。

**最重要的是**：不管用哪家，你在这本书里学到的东西——变量、循环、函数、`messages` 列表、写好提示词、把密钥保密——**全都通用**。换 AI 就像换个牌子的电池，装电池的手电筒（你的程序）还是你亲手做的那个。

> 🔑 **最后再唠叨一遍**：无论哪一家的密钥，都别写进代码、别发给别人、别传到网上。用完记得关掉终端窗口。安全第一。
