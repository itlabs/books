# 附录 C 读懂编译器与链接器的报错

C++ 的报错以吓人著称。但绝大多数错误其实落在有限的几类里，认得出套路就不慌。这份附录教你**快速定位**：先分清是哪个阶段的错（第 2 章），再对号入座常见模式。

## 第一步：分清编译错误还是链接错误

回忆第 2 章：编译和链接是两个阶段。**看报错来自谁，是定位的第一步**：

<div class="keypoint">
<strong>一眼分辨</strong>
<ul>
<li><strong>编译错误</strong>：报错带<strong>文件名和行号</strong>（<code>main.cpp:12:5: error: ...</code>），来自单个文件内部说不通。</li>
<li><strong>链接错误</strong>：报错里出现 <code>ld</code>、<code>collect2</code>、<code>undefined reference</code>、<code>duplicate symbol</code>，<strong>没有行号</strong>，发生在把多个 <code>.o</code> 拼一起时。</li>
</ul>
搞错阶段会浪费大量时间——链接错误里去源码某一行找语法问题，永远找不到。
</div>

## 编译错误常见模式

### 1. 缺分号 / 括号不匹配

```
error: expected ';' before '}' token
```

**最常见的低级错误。** 关键：编译器报的行**往往是真正出错行的下一行**——因为它读到下一个 token 才发现"前面少了个东西"。看到 `expected ';'`，先检查**上一行**行尾。

### 2. 未声明的名字

```
error: 'foo' was not declared in this scope
error: use of undeclared identifier 'foo'
```

原因通常是：拼写错、忘了 `#include` 对应头文件、或命名空间没写对（如 `cout` 写成没有 `std::`）。

<div class="note">
<strong>忘记 <code>#include</code> 的典型信号</strong>
用了 <code>std::vector</code> 却没 <code>#include &lt;vector&gt;</code>，报错可能是 "'vector' is not a member of 'std'" 或 "'vector' was not declared"。<strong>用了标准库的什么，就要 include 对应的头</strong>：<code>vector</code>→<code>&lt;vector&gt;</code>，<code>cout</code>→<code>&lt;iostream&gt;</code>，<code>string</code>→<code>&lt;string&gt;</code>，等等。
</div>

### 3. 类型不匹配 / 无法转换

```
error: cannot convert 'const char*' to 'int' in initialization
```

字面意思：类型对不上。回忆第 9、10 章——如果是"无法从 X 隐式转换到 Y"，可能是某个构造函数你想让它隐式转换、却被 `explicit` 挡住了（或相反，你没料到会发生隐式转换）。

### 4. 找不到匹配的函数（重载/模板）

```
error: no matching function for call to 'foo(...)'
note: candidate: ... (指出候选和为什么不匹配)
```

**别只看第一行 error，一定要读下面的 `note:`**——它逐个列出候选函数、并说明每个为什么不匹配（参数类型不对、数量不对、const 不匹配等）。这是定位重载/模板问题的关键信息。

### 5. const 相关

```
error: passing 'const X' as 'this' argument discards qualifiers
```

第 9 章的经典错误：你在一个 `const` 对象上调用了**非 const** 成员函数。要么给那个成员函数加 `const`，要么确认你不该改这个对象。

## 链接错误常见模式

### 1. undefined reference（找不到定义）

```
undefined reference to `foo()'
collect2: error: ld returned 1 exit status
```

**最常见的链接错误。** 意思是：某处**声明并使用**了 `foo`，但整个程序里**找不到它的定义**（第 2 章）。排查清单：

<div class="keypoint">
<strong><code>undefined reference</code> 的排查顺序</strong>
<ol>
<li><strong>忘了定义</strong>：只在头文件写了声明，没写函数体。</li>
<li><strong>忘了把 .cpp 加入编译</strong>：定义在 <code>helper.cpp</code> 里，但编译时只写了 <code>g++ main.cpp</code>——漏了 <code>helper.cpp</code>。改成 <code>g++ main.cpp helper.cpp</code>（或用 CMake，附录 A）。</li>
<li><strong>忘了链接库</strong>：用了某个库的函数，但没 <code>-l</code> 链接它（如线程忘了 <code>-pthread</code>）。</li>
<li><strong>签名不一致</strong>：声明和定义的参数/const 不完全一致，链接器认为是两个不同的东西。</li>
</ol>
</div>

### 2. multiple definition（定义重复）

```
multiple definition of `foo()'
```

第 2 章的 **ODR** 被违反了——同一个函数/全局变量被定义了不止一次。最常见的原因：**把普通函数的定义写进了头文件**，然后多个 `.cpp` 都 `#include` 了它。解决：头文件里只放**声明**，定义放到某个 `.cpp`；或者给函数加 `inline`（允许多个翻译单元有相同定义）。

## 模板错误：抓住关键行

模板的报错以"一屏都是"著称（第 11 章）。别被吓到，有套路：

<div class="keypoint">
<strong>读模板报错的技巧</strong>
<ol>
<li><strong>从最后往前读</strong>，或找 <code>required from here</code> / <code>in instantiation of</code>——它标出<strong>你的代码</strong>是从哪一行触发实例化的（那才是你要改的地方，而非中间那堆标准库内部行）。</li>
<li><strong>找真正的失败原因</strong>：一大段里通常有一句 <code>no match for 'operator&lt;'</code> 之类的话，指出模板到底用了类型不支持的操作。</li>
<li><strong>治本：用 concepts</strong>（第 12 章）。给模板加约束后，报错会变成清晰的"类型不满足某约束"，直指调用处——这正是 concepts 最大的实用价值。</li>
</ol>
</div>

## 运行时才炸：未定义行为

有一类最难缠的问题**根本不报错**——编译链接都过，运行时却崩溃、或时对时错。这通常是**未定义行为**：悬垂引用（第 4 章）、越界访问、使用已移动对象（第 6 章）、数据竞争（第 20 章）等。

<div class="warn">
<strong>对付未定义行为：上消毒器</strong>
编译器帮不了你（代码"合法"），但运行时工具能。用 <strong>AddressSanitizer / UBSan</strong>（附录 A）重新编译并运行：
<pre>g++ -std=c++23 -g -fsanitize=address,undefined main.cpp -o app
./app</pre>
它会在程序真正踩雷的那一刻<strong>报告类型和精确位置</strong>（"heap-use-after-free at ..."、"index out of bounds"）。这是排查"编译通过但运行出错"的第一利器。
</div>

## 通用建议

<div class="keypoint">
<strong>面对报错的心法</strong>
<ol>
<li><strong>从第一个错误改起</strong>：C++ 的错误常有连锁反应，一个错误引发一串。先修最上面那个，重新编译，后面很多可能自动消失。</li>
<li><strong>读 <code>note:</code> 行</strong>：它们是编译器给的诊断上下文，往往比 <code>error:</code> 那行更有用。</li>
<li><strong>始终开 <code>-Wall -Wextra</code></strong>：很多 bug 编译器早以 warning 提示了，别忽略警告。</li>
<li><strong>拿不准就丢进 Compiler Explorer</strong>：换个编译器（GCC/Clang）看看，有时另一个的报错更易懂。</li>
</ol>
</div>
