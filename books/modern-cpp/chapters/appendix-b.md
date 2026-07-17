# 附录 B 从 Java / Python 到 C++ 术语对照表

这份附录是给你查的——遇到陌生的 C++ 概念，先看它对应你熟悉的 Java / Python 里的什么，再看关键差异。**⚠️ 标记的是"看起来像、其实很不同"的陷阱**，这些最容易踩。

## 变量与内存

| 概念 | Java / Python | C++ | 关键差异 |
|---|---|---|---|
| 变量 | 对象的引用 | ⚠️ 对象本身（值语义） | `b = a` 在 C++ 是**拷贝**，不是共享（第 3 章） |
| 对象位置 | 几乎全在堆 | ⚠️ 默认在栈 | C++ 不写 `new` 也能建对象（第 4 章） |
| 内存回收 | GC 自动 | ⚠️ RAII（析构自动） | 确定时刻销毁，非"未来某时"（第 5 章） |
| 空值 | `null` / `None` | `nullptr`（指针）/ `std::optional`（值） | 引用不能为空（第 8 章） |
| 常量 | `final` | `const` / `constexpr` | `const` 更深、会传染；`constexpr` 是编译期（第 9、13 章） |

## 引用、指针与所有权

| 概念 | Java / Python | C++ | 关键差异 |
|---|---|---|---|
| 传参默认 | 传引用（对象） | ⚠️ 传值（拷贝） | 要改外部对象需显式 `&`（第 3 章） |
| "指向"对象 | 引用（隐式） | 引用 `T&` / 指针 `T*` | 引用不可空不可改绑；指针可（第 8 章） |
| 独占持有堆对象 | —（GC 管） | `std::unique_ptr` | 不可拷贝，只可移动（第 8 章） |
| 共享持有 | 普通引用 | `std::shared_ptr` | 引用计数，归零才释放（第 8 章） |
| 破除循环引用 | —（GC 能处理环） | ⚠️ `std::weak_ptr` | `shared_ptr` 循环会泄漏（第 8 章） |
| 移动/转移所有权 | —（无此概念） | ⚠️ `std::move` | 只是类型转换，"贴可移动标签"（第 6 章） |

## 类与对象

| 概念 | Java / Python | C++ | 关键差异 |
|---|---|---|---|
| 构造函数 | `constructor` / `__init__` | 构造函数 | 用**成员初始化列表**（第 10 章） |
| 析构/清理 | `finalize` / `__del__`（不可靠） | ⚠️ 析构函数 `~T()`（确定） | 离开作用域即调用（第 5 章） |
| 方法默认虚 | ⚠️ 是（Java） | 否，需显式 `virtual` | C++ 默认非虚更快（第 18 章） |
| 抽象方法 | `abstract` / `ABC` | 纯虚函数 `= 0` | （第 18 章） |
| 重写标注 | `@Override` | `override` | 建议每次都写（第 18 章） |
| 禁止拷贝 | —（自行控制） | `= delete` | （第 10 章） |
| 只读方法 | —（约定） | `const` 成员函数 | const 对象只能调 const 方法（第 9 章） |

## 泛型与编译期

| 概念 | Java / Python | C++ | 关键差异 |
|---|---|---|---|
| 泛型 | `List<T>` / 鸭子类型 | 模板 `template<typename T>` | ⚠️ 编译期**单态化**，非类型擦除（第 11 章） |
| 泛型约束 | `<T extends X>` | concepts（`requires`） | 更灵活的鸭子类型契约（第 12 章） |
| 编译期常量 | —（很有限） | `constexpr` / `consteval` | 能编译期跑循环、算表（第 13 章） |
| 编译期断言 | —（运行时 assert） | `static_assert` | 不满足直接编译失败（第 13 章） |

## 标准库

| 概念 | Java | Python | C++ |
|---|---|---|---|
| 动态数组 | `ArrayList` | `list` | `std::vector`（第 14 章） |
| 哈希表 | `HashMap` | `dict` | `std::unordered_map`（第 14 章） |
| 有序表 | `TreeMap` | — | `std::map`（第 14 章） |
| 集合 | `HashSet` | `set` | `std::unordered_set`（第 14 章） |
| 遍历抽象 | `Iterator` | 迭代器协议 | 迭代器（`begin`/`end`，第 14 章） |
| 流式处理 | Stream | 推导式/生成器 | ranges + views（第 15 章） |
| 匿名函数 | `x -> ...` | `lambda` | ⚠️ lambda（捕获要显式，第 16 章） |
| 可能无值 | `Optional<T>` | 返回 `None` | `std::optional<T>`（第 17 章） |
| 成功或错误 | —（多用异常） | — | `std::expected<T,E>`（第 17 章） |

## 错误处理与并发

| 概念 | Java / Python | C++ | 关键差异 |
|---|---|---|---|
| 抛/捕获异常 | `throw` / `try-catch` | `throw` / `try-catch` | 无受检异常；按 `const&` 捕获（第 17 章） |
| 资源清理 | try-with-resources / `with` | ⚠️ RAII（自动） | 每个对象默认行为，非专门语法（第 5 章） |
| 承诺不抛 | —（无） | `noexcept` | 影响移动优化（第 17 章） |
| 线程 | `Thread` | `threading.Thread` | `std::thread` / `std::jthread`（第 20 章） |
| 真并行 | ⚠️ 是 / ⚠️ 否(GIL) | 是（无 GIL） | Python 多线程受 GIL 限制，C++ 不受（第 20 章） |
| 加锁 | `synchronized` / `Lock` | `std::mutex` + `lock_guard` | RAII 自动解锁（第 20 章） |
| 数据竞争 | 有内存模型保护 | ⚠️ 未定义行为 | 后果更严重（第 20 章） |

## 编译模型

| 概念 | Java / Python | C++ | 关键差异 |
|---|---|---|---|
| 模块引入 | `import` | `#include`（/ C++20 `import`） | ⚠️ `#include` 是文本粘贴（第 2 章） |
| 编译单位 | 类 / 模块 | 翻译单元（每个 `.cpp`） | 独立编译再链接（第 2 章） |
| 声明与定义 | 一体 | ⚠️ 分离（头文件/源文件） | 分出编译错误 vs 链接错误（第 2、附录 C 章） |
| 优化 | 运行时 JIT | 编译期（`-O2`） | 结果可在汇编里直接看（第 19 章） |

<div class="keypoint">
<strong>最该记住的五个"看起来像、其实不同"</strong>
<ol>
<li><strong>变量是对象本身</strong>，不是引用——赋值是拷贝（第 3 章）。</li>
<li><strong>对象默认在栈上</strong>、离开作用域确定销毁——RAII 而非 GC（第 4、5 章）。</li>
<li><strong>传参默认拷贝</strong>，共享要显式 <code>&</code>（第 3 章）。</li>
<li><strong>方法默认非虚</strong>，多态要显式 <code>virtual</code>（第 18 章）。</li>
<li><strong>模板是编译期代码生成</strong>，不是运行时泛型（第 11 章）。</li>
</ol>
这五点是从托管语言切换到 C++ 时最大的认知落差，也是本书反复强调的主线。
</div>
