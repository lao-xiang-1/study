---
sr-due: 2026-09-07
sr-interval: 24
sr-ease: 250
---
#code

## 1. 正则表达式与 `re` 模块

### 1.1 什么是正则表达式

正则表达式（Regular Expression，简称 regex）是一种描述**字符串模式**的小型语言。用一个模式串去匹配、搜索、替换文本。

Python 内置 `re` 模块提供正则支持：

```python
import re

re.match(r"u", "u:se:zero")   # 从字符串开头匹配
re.search(r"<-", "h, u <- u")  # 在任意位置搜索第一个匹配
re.findall(r"\d", "a1b2c3")    # 找出所有匹配，返回列表 ['1','2','3']
```

### 1.2 `re` 模块常用函数

| 函数 | 作用 | 返回 |
|------|------|------|
| `re.match(pattern, s)` | 从**开头**匹配 | `Match` 或 `None` |
| `re.search(pattern, s)` | 搜索**第一个**匹配（任意位置） | `Match` 或 `None` |
| `re.findall(pattern, s)` | 找出**所有**匹配 | 字符串列表 |
| `re.finditer(pattern, s)` | 找出所有匹配 | 迭代器，每个是 `Match` |
| `re.sub(pattern, repl, s)` | 替换所有匹配 | 新字符串 |
| `re.compile(pattern)` | 预编译模式（重复使用更高效） | `Pattern` 对象 |

> ⚠️ `re.match` 只在开头匹配，等价于 `re.search` 加了 `^` 锚点。这是最常见的坑之一。

### 1.3 `Match` 对象

匹配成功后返回 `Match` 对象，常用方法：

```python
m = re.match(r"(\w+),\s*(\w+)", "h, u")
m.group()    # 'h, u'      整个匹配
m.group(0)   # 'h, u'      同上
m.group(1)   # 'h'         第 1 个捕获组
m.group(2)   # 'u'         第 2 个捕获组
m.group(1, 2)  # ('h', 'u')  多个组一起取，返回元组
m.groups()   # ('h', 'u')  所有捕获组
m.start()    # 0  匹配起始位置
m.end()      # 4  匹配结束位置
```

### 1.4 `re.escape()` — 转义特殊字符

**作用**：把字符串中所有**正则特殊字符**（`.` `*` `+` `?` `(` `)` `[` `]` `{` `}` `^` `$` `|` `\`）前面加上反斜杠 `\`，让这段字符串被当作**字面文本**匹配，而不是被解析成正则语法。

```python
import re

print(re.escape("a.b"))    # a\.b    点号被转义
print(re.escape("1+1=2"))  # 1\+1=2  加号被转义
print(re.escape("hello"))  # hello   普通字符不变
```

**为什么要转义**：`.` `*` `+` 等在正则里有特殊含义；如果直接把用户输入拼进 pattern，就可能匹配错：

```python
# 想搜字面 "a.b"，但 . 被当成「任意字符」
re.match(r"a.b", "aXb")               # 匹配成功（错！）

# 用 re.escape 转义后，才真正匹配字面 "a.b"
re.match(re.escape("a.b"), "a.b")     # 匹配成功
re.match(re.escape("a.b"), "aXb")     # None（对）
```

**典型用法**：把动态变量 / 用户输入安全地嵌入正则：

```python
keyword = input("要搜索的关键词：")       # 用户可能输入 "C++" 或 "a.b"
pattern = re.compile(re.escape(keyword))  # 把 keyword 当纯文本搜索
re.search(pattern, text)
```

> 💡 Python 3.7+ 的 `re.escape` **只转义真正的特殊字符**，字母、数字、下划线保持原样（所以 `re.escape("hello")` 返回 `"hello"`）。

### 1.5 `re.compile()` 与 `Pattern` 类

`re.compile(pattern, flags=0)` 把正则**预先编译**成 `re.Pattern` 对象，之后反复调用它的方法，避免每次重新解析：

```python
import re

pat = re.compile(r"\d+")   # pat 是 re.Pattern 实例
pat.search("a1b22c333")    # <re.Match object; span=(1, 2), match='1'>
```

`Pattern` 对象的常用方法（与模块函数一一对应）：

| Pattern 方法 | 等价的模块函数 | 作用 |
|------|------|------|
| `pat.match(s)` | `re.match(pattern, s)` | 从**开头**匹配 |
| `pat.fullmatch(s)` | `re.fullmatch(pattern, s)` | **整串**完全匹配 |
| `pat.search(s)` | `re.search(pattern, s)` | 搜索**第一个**匹配 |
| `pat.findall(s)` | `re.findall(pattern, s)` | 所有匹配 → 列表 |
| `pat.finditer(s)` | `re.finditer(pattern, s)` | 所有匹配 → 迭代器 |
| `pat.sub(repl, s)` | `re.sub(pattern, repl, s)` | 替换所有匹配 |
| `pat.split(s)` | `re.split(pattern, s)` | 按模式切分 |

常用属性：

| 属性 | 含义 |
|------|------|
| `pat.pattern` | 原始模式串 |
| `pat.flags` | 编译时传入的 flags |
| `pat.groups` | 捕获组数量 |
| `pat.groupindex` | 命名组名 → 编号的映射 |

### 1.6 `Pattern` 方法 vs 模块函数

**功能完全一样**。模块级函数内部就是 `re.compile(pattern).method(s)` 的简写（自带编译缓存）。区别只有三点：

1. **参数不同**：模块函数第一个参数是 pattern 字符串；`Pattern` 方法已「记住」模式，第一个参数直接是待匹配字符串。
2. **flags 位置不同**：`re.match(p, s, flags=...)` 每次调用都能传；`Pattern` 的 flags 在 `compile()` 时定死，方法里不能再传。
3. **性能**：同一正则重复使用很多次时，先 `compile` 一次比每次 `re.match(p, s)` 更高效。

```python
pat = re.compile(r"\d+", re.I)   # flags 在这里固定
pat.search("ABC123")              # 等价于 re.search(r"\d+", "ABC123", re.I)
```

### 1.7 编译标志（flags）与 `re.X`

flags 在编译时（或模块函数调用时）传入，控制匹配行为：

| flag | 全名 | 作用 |
|------|------|------|
| `re.I` | `re.IGNORECASE` | 忽略大小写 |
| `re.M` | `re.MULTILINE` | `^` `$` 匹配每行首尾 |
| `re.S` | `re.DOTALL` | `.` 也匹配换行符 |
| `re.X` | `re.VERBOSE` | 允许注释和空白（详细模式） |

> ⚠️ flags 是**常量**不是函数，`re.X()` 会报 `TypeError: 'int' object is not callable`。

`re.X`（`re.VERBOSE`）让正则里可以写**注释**和**换行缩进**，方便写长正则：

```python
pat = re.compile(r"""
    \d+    # 数字部分
    \s*    # 可选的空白
    \w+    # 单词部分
""", re.X)

pat.search("abc 123 word")   # 匹配 '123 word'
```

