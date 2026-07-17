# 附录 D 继续前进：标准、书目与工具链

读完本书，你已经建立起 C++ 的核心心智模型。但 C++ 是一门庞大的语言，本书是**地基**，不是全部。这份附录给你一张继续前进的地图。

## 本书没深入、但你会遇到的话题

按重要性大致排序，这些是自然的下一步：

<div class="keypoint">
<strong>值得接着学的主题</strong>
<ul>
<li><strong>完美转发与转发引用</strong>（<code>T&amp;&amp;</code> 在模板里的特殊含义、<code>std::forward</code> 的完整故事）——第 11、16 章点到，值得系统学。</li>
<li><strong>模板元编程与类型特征</strong>（<code>&lt;type_traits&gt;</code>、SFINAE 的历史、变量模板）。</li>
<li><strong>移动语义的进阶细节</strong>（引用折叠、值类别在泛型中的传播）。</li>
<li><strong>更多标准库</strong>：<code>std::variant</code> / <code>std::any</code>、<code>std::string_view</code>、<code>std::span</code>、<code>&lt;chrono&gt;</code> 时间库、<code>&lt;filesystem&gt;</code>、格式化 <code>std::format</code>。</li>
<li><strong>协程</strong>（C++20 <code>co_await</code>/<code>co_yield</code>）——异步与惰性生成的新范式。</li>
<li><strong>模块</strong>（C++20 <code>import</code>）——第 2 章提过，头文件的现代替代，工具链正在成熟。</li>
<li><strong>并发进阶</strong>：<code>std::future</code>/<code>std::async</code>、条件变量、无锁数据结构、内存序细节（第 20 章）。</li>
</ul>
</div>

## C++ 标准的演进

C++ 每三年出一个新标准，理解版本脉络能帮你读懂各种代码：

| 标准 | 年份 | 标志性特性 |
|---|---|---|
| C++11 | 2011 | 移动语义、`auto`、lambda、智能指针、`nullptr`、右值引用——**现代 C++ 的起点** |
| C++14 | 2014 | 泛型 lambda、放宽的 `constexpr` |
| C++17 | 2017 | 结构化绑定、`std::optional`/`variant`、`if` 初始化、CTAD、强制拷贝消除 |
| C++20 | 2020 | **concepts、ranges、协程、模块**、`std::jthread`——一次大更新 |
| C++23 | 2023 | `std::expected`、`std::print`、更强的 ranges 与 `constexpr` |

<div class="note">
<strong>"现代 C++" 指什么</strong>
通常指 <strong>C++11 及以后</strong>的风格：值语义 + RAII + 智能指针 + 移动语义为默认，几乎不写裸 <code>new</code>/<code>delete</code>。本书教的就是这套。如果你在网上看到满是裸指针、<code>new</code>/<code>delete</code>、<code>NULL</code> 的代码，那多半是 C++11 之前的老风格——理解它、但别模仿它。
</div>

## 权威学习资源

<div class="keypoint">
<strong>参考与查阅</strong>
<ul>
<li><strong><a href="https://en.cppreference.com">cppreference.com</a></strong>——标准库和语言特性的权威参考，<strong>你会天天用</strong>。每个类型/函数都标注了从哪个标准起可用。</li>
<li><strong><a href="https://godbolt.org">Compiler Explorer</a></strong>——本书用的在线编译器，看汇编、对比编译器、快速试代码的神器。</li>
<li><strong><a href="https://isocpp.org/faq">C++ Core Guidelines</a></strong>（Bjarne Stroustrup 与 Herb Sutter 主持）——现代 C++ 的官方最佳实践大全。</li>
</ul>
</div>

<div class="keypoint">
<strong>值得读的书</strong>
<ul>
<li><strong>《Effective Modern C++》</strong>（Scott Meyers）——讲透 C++11/14 的移动语义、智能指针、lambda 等，本书读者的<strong>理想下一本</strong>。</li>
<li><strong>《A Tour of C++》</strong>（Bjarne Stroustrup，语言之父）——简洁的现代 C++ 全景导览。</li>
<li><strong>《C++ Concurrency in Action》</strong>（Anthony Williams）——想深入第 20 章的并发，就读它。</li>
<li><strong>《C++ Templates: The Complete Guide》</strong>——模板的百科全书，进阶时查阅。</li>
</ul>
</div>

## 保持工具链的现代化

<div class="keypoint">
<strong>让工具帮你写对代码</strong>
<ul>
<li><strong>始终开 <code>-Wall -Wextra</code></strong>，把警告当错误对待（<code>-Werror</code>）。</li>
<li><strong>日常用 AddressSanitizer / UBSan 跑测试</strong>（附录 A、C）——它抓住的 bug 比你想象的多。</li>
<li><strong>用 clang-tidy 做静态检查</strong>，用 clang-format 统一风格。</li>
<li><strong>写单元测试</strong>：GoogleTest、Catch2 是主流框架。</li>
<li><strong>跟上标准</strong>：编译器默认标准往往偏旧，记得显式 <code>-std=c++23</code>（或你能用的最新版）。</li>
</ul>
</div>

## 最后的话

你从"变量就是对象本身"这一个观念出发，一路建立起对内存、生命周期、所有权、编译期、以及编译器行为的完整认识。这套心智模型，是很多只会"用" C++ 的人从未真正拥有的。

C++ 的学习没有终点——语言在演进，你的理解也会随项目加深。但**最难的那道坎——从"托管语言的引用世界"切换到"C++ 的值与生命周期世界"——你已经迈过去了**。剩下的，是在实践中把它变成本能。

去写代码吧。遇到困惑时，随时回到 Compiler Explorer 点开"看汇编"，问问编译器到底做了什么——那往往是最好的老师。

**祝你在 C++ 的世界里，写得清晰，也跑得飞快。**
