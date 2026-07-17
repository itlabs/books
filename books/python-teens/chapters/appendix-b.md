# 附录 B 报错看不懂？速查表

写代码时，屏幕上突然冒出一大段红红的英文——很多人第一反应是心里一紧："我是不是很笨？"

**先说最重要的一句话：报错不是你笨，是电脑在帮你找问题。**

想象你在写作文，老师用红笔圈出一个错别字。老师不是在骂你，而是在帮你改得更好。Python 的报错就是那支红笔：它发现了一个它看不懂的地方，于是停下来告诉你"喂，这里好像不太对，你来看看"。**每一个程序员，包括写了几十年代码的高手，每天都在和报错打交道。** 会看报错，是编程最有用的本领之一。

这一节教你两件事：**怎么读报错**，以及一张**常见报错速查表**，遇到不懂的翻过来查就行。

---

## 第一招：报错要从下往上看

Python 的报错信息（英文叫 **Traceback**，意思是"回溯、往回追查"）经常有好几行，看起来吓人。但记住一个诀窍：

> 小贴士：**Traceback 从下往上看，最后一行最重要。** 上面那些行是电脑在"讲它一路是怎么走到出错这里的"（对新手用处不大），而**最后一行**才直接告诉你：错的是什么类型、大概什么原因。先看最后一行，往往就够了。

我们看个例子。假设你运行了这段代码：

```python-norun
print("开始")
print(score)
```

屏幕上会出现类似这样的一段：

```
Traceback (most recent call last):
  File "hello.py", line 2, in <module>
    print(score)
NameError: name 'score' is not defined
```

别被前面几行吓到，我们**从下往上**拆开看：

- **最后一行** `NameError: name 'score' is not defined`：这是重点！`NameError` 是错误类型（名字错误），冒号后面是解释——"名字 score 没有被定义过"。也就是说，你用了一个电脑没见过的名字 `score`。
- **倒数第二、三行** `File "hello.py", line 2`：告诉你出错在 **hello.py 文件的第 2 行**。太贴心了，直接把出事地点报给你。

所以读报错的顺口溜是：**先看最后一行（错在哪类），再看行号（错在哪行），最后回去改代码。** 就这么简单。

<div class="concept">
<svg viewBox="0 0 540 190" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">
  <rect x="20" y="20" width="500" height="150" rx="10" fill="#1b1b1b"/>
  <text x="35" y="48" font-size="13" fill="#8a8a8a" font-family="monospace">Traceback (most recent call last):</text>
  <text x="35" y="72" font-size="13" fill="#8a8a8a" font-family="monospace">  File "hello.py", line 2, in ...</text>
  <text x="35" y="96" font-size="13" fill="#8a8a8a" font-family="monospace">    print(score)</text>
  <text x="35" y="130" font-size="14" fill="#ff8a6b" font-family="monospace">NameError: name 'score' is not defined</text>
  <path d="M300 145 L300 138" stroke="#3f9b5c" stroke-width="0"/>
  <text x="300" y="160" text-anchor="middle" font-size="13" fill="#9fe3b0">↑ 先看这一行！最重要</text>
</svg>
<div class="caption">图 B-1　报错从下往上看，最后一行告诉你错误类型和原因</div>
</div>

---

## 第二招：查速查表

知道了看最后一行，接下来只要认识**冒号前那个错误类型的名字**，就能大致猜到哪儿出了问题。下面这张表收了 9 个新手最常遇到的报错。**遇到报错，先记下最后一行开头那个词，回来这里查。**

| 错误类型 | 报错大概长啥样 | 中文意思 | 常见原因 | 怎么修 |
|---|---|---|---|---|
| **SyntaxError** | `SyntaxError: invalid syntax` | 语法错误 | 代码写得电脑读不通：中文括号/引号、少了冒号 `:`、括号没配对 | 把输入法切英文；检查 `( )`、`" "`、`:` 是否齐全、成对 |
| **NameError** | `NameError: name 'x' is not defined` | 名字没定义 | 用了一个还没创建的变量；或函数名、变量名拼错了 | 检查拼写；确认这个名字在**使用之前**已经赋过值 |
| **TypeError** | `TypeError: can only concatenate str (not "int") to str` | 类型不对 | 把不同"种类"的数据硬凑一起，比如文字 `+` 数字 | 用 `str()` 把数字变成文字，或用 `int()` 把文字变数字，再运算 |
| **ValueError** | `ValueError: invalid literal for int() with base 10: 'abc'` | 值不合适 | 类型对但内容不行，比如想把 `"abc"` 变成整数 | 检查要转换的内容是不是真的数字；输入前先验证 |
| **IndexError** | `IndexError: list index out of range` | 下标越界 | 列表里没那么多东西，你却去取第 5 个，可它只有 3 个 | 记住下标从 **0** 开始；确认序号没超过列表长度 |
| **KeyError** | `KeyError: 'name'` | 键不存在 | 字典里没有你要找的那个键（钥匙） | 检查键名拼写；用 `字典.get(键)` 取值可避免报错 |
| **IndentationError** | `IndentationError: expected an indented block` | 缩进错误 | 该缩进的地方没缩进，或缩进乱了（空格和 Tab 混用） | 冒号 `:` 下一行要缩进；全用空格（一般 4 个），别混 Tab |
| **ModuleNotFoundError** | `ModuleNotFoundError: No module named 'xxx'` | 找不到模块 | 你 `import` 的库还没装；或库名拼错 | 在终端 `pip install 库名` 装上；检查名字拼写 |
| **ZeroDivisionError** | `ZeroDivisionError: division by zero` | 除以零了 | 做了 `÷ 0` 的运算，数学上不允许 | 除之前先判断除数是不是 0，是 0 就特殊处理 |

> 小贴士：表里 `int`、`str`、`list` 这些词你现在可能还没全学到，没关系。等学到相应的章节，再回来看这张表会豁然开朗。**这张表是拿来"查"的，不是拿来"背"的**——收藏好，遇到报错翻过来对一对就行。

---

## 三个最常见报错，细讲一下

上面表格压缩得比较狠，这里把新手前几周最常撞到的三个，用例子展开讲。

### SyntaxError（语法错误）——十有八九是符号问题

```python-norun
print("你好"）
```

```
SyntaxError: invalid character '）' (U+FF08)
```

这里的括号 `）` 是**中文全角**的，Python 只认英文半角 `)`。**新手一大半的 SyntaxError 都是中文符号造成的。** 写代码前把输入法切到英文，能躲掉无数坑。

### NameError（名字没定义）——不是拼错，就是用早了

```python-norun
print(age)
age = 13
```

```
NameError: name 'age' is not defined
```

问题是：你在**创建 `age` 之前**就想用它。代码是从上往下运行的，第 1 行执行时 `age` 还不存在。把顺序换过来（先 `age = 13` 再 `print`）就好。另一种常见原因是单纯**拼错了名字**，比如变量叫 `player_name` 你却写成 `playername`。

### IndentationError（缩进错误）——空白也是有意义的

```python-norun
if score > 60:
print("及格")
```

```
IndentationError: expected an indented block
```

Python 靠"缩进"（每行前面的空格）来判断哪些代码属于 `if` 里面。冒号 `:` 的下一行必须往右缩进（通常 4 个空格）。VS Code 一般会帮你自动缩，但如果你手动删过空格，就可能出这个错。

> 小贴士：**缩进只用空格，别用 Tab 键，更别空格和 Tab 混着用**——它俩看起来一样，电脑却分得清清楚楚，混用会莫名其妙报错。VS Code 里按 Tab 键通常会自动变成空格，一般不用操心；真出问题时，检查是不是混用了。

---

## 遇到表里没有的报错怎么办？

编程世界里的报错成千上万，这张表只收了最常见的。碰到没见过的，试试这几招：

1. **只看最后一行**，把冒号后面那句英文，一个词一个词查一下意思，往往就懂大半了。
2. **把最后一行原封不动地复制**，去搜索引擎搜。全世界都有人踩过同一个坑，答案大概率现成。
3. **看行号，回到那一行**，仔细读读，八成能发现哪个符号、哪个名字不对劲。
4. **问 AI 帮手**：把报错信息发给 AI，让它用简单的话解释。（本书后面几章会教你怎么和 AI 对话。）

> 小贴士：养成一个好习惯——**报错出现时，先别急着改代码，先把最后一行读完。** 花 10 秒读懂它，往往比瞎改 10 分钟管用。报错是你的朋友，不是敌人。

---

记住这一节的两句话就够了：**从下往上看，最后一行最重要；报错不是你笨，是电脑在帮你找问题。** 带着这个心态，你会发现调试（找错、改错）甚至有点像破案的乐趣——而你，就是那个越来越厉害的侦探。
