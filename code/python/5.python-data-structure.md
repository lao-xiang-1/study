# Python 数据结构

Python 内置了丰富且高效的数据结构，掌握它们的特性和常用操作是编写 Pythonic 代码的基础。

---

## 1. 字符串（str）

字符串是**不可变**的字符序列，支持索引和切片。

### 1.1 常用方法

| 方法 | 说明 | 示例 |
|------|------|------|
| `len(s)` | 返回字符串长度 | `len("abc")` → `3` |
| `s.split(sep)` | 按分隔符分割为列表 | `"a,b".split(",")` → `['a', 'b']` |
| `s.join(iter)` | 用字符串连接可迭代对象 | `"-".join(['a', 'b'])` → `"a-b"` |
| `s.strip()` | 去除首尾空白字符 | `"  abc  ".strip()` → `"abc"` |
| `s.startswith(p)` | 是否以某前缀开头 | `"abc".startswith("ab")` → `True` |
| `s.endswith(p)` | 是否以某后缀结尾 | `"abc".endswith("bc")` → `True` |
| `s.replace(old, new)` | 替换子串 | `"aabb".replace("a", "x")` → `"xxbb"` |
| `s.find(sub)` | 查找子串位置，不存在返回 `-1` | `"abc".find("b")` → `1` |
| `s.index(sub)` | 查找子串位置，不存在抛异常 | `"abc".index("b")` → `1` |
| `s.count(sub)` | 统计子串出现次数 | `"ababa".count("a")` → `3` |
| `s.upper()` | 转大写 | `"abc".upper()` → `"ABC"` |
| `s.lower()` | 转小写 | `"ABC".lower()` → `"abc"` |
| `s.isdigit()` | 是否全为数字 | `"123".isdigit()` → `True` |
| `s.isalpha()` | 是否全为字母 | `"abc".isalpha()` → `True` |

### 1.2 切片操作

```python
s = "abcdef"
s[0:3]     # "abc"   — 左闭右开
s[:3]      # "abc"   — 从头开始
s[3:]      # "def"   — 到末尾
s[::2]     # "ace"   — 步长为 2
s[::-1]    # "fedcba" — 反转字符串
```

### 1.3 格式化字符串

```python
# f-string（推荐，Python 3.6+）
name = "Alice"; age = 20
f"{name} is {age} years old"

# format 方法
"{} is {} years old".format(name, age)
"{0} is {1} years old".format(name, age)
"{n} is {a} years old".format(n=name, a=age)

# 格式化数字
f"{3.14159:.2f}"    # "3.14"   — 保留两位小数
f"{1000:,}"         # "1,000"  — 千位分隔
f"{42:08b}"         # "00101010" — 二进制补零
```

### pcx示例：`str.rsplit` / `str.split` —— 字符串分割

```python
# pcx/predictive_coding/_vode.py:122 —— 从右分割（递归处理链式变换）
tform, _t = tform.rsplit(":", 1)
# "u:se:zero"  rsplit(":", 1)  →  ("u:se", "zero")
# "u:se"       rsplit(":", 1)  →  ("u", "se")

# pcx/predictive_coding/_vode.py:265 —— 逗号分割目标
for _target in _targets.split(","):
    _target = _target.strip()      # 去掉空格
```

**`rsplit` vs `split`**：`rsplit(":", 1)` 从**右边**割一刀，`split(":", 1)` 从**左边**割一刀。链式变换 "u:se:zero" 需要从右向左处理（先算出 `u:se` 的结果，再传给 `zero`），所以用 `rsplit`。


---

## 2. 列表（list）

列表是**可变**的有序序列，可以存储任意类型的元素。

### 2.1 常用方法

| 方法 | 说明 | 时间复杂度 |
|------|------|-----------|
| `len(lst)` | 返回列表长度 | O(1) |
| `lst.append(x)` | 末尾添加元素 | O(1) |
| `lst.extend(iter)` | 批量添加元素 | O(k) |
| `lst.insert(i, x)` | 在位置 `i` 插入元素 | O(n) |
| `lst.pop()` | 弹出末尾元素 | O(1) |
| `lst.pop(i)` | 弹出位置 `i` 的元素 | O(n) |
| `lst.remove(x)` | 删除第一个值为 `x` 的元素 | O(n) |
| `lst.index(x)` | 查找 `x` 的索引 | O(n) |
| `lst.count(x)` | 统计 `x` 出现次数 | O(n) |
| `lst.sort()` | 原地排序（升序） | O(n log n) |
| `lst.reverse()` | 原地反转 | O(n) |
| `lst.clear()` | 清空列表 | O(1) |
| `lst.copy()` | 浅拷贝 | O(n) |

### 2.2 列表推导式

```python
# 基本形式
squares = [x**2 for x in range(10)]

# 带条件
evens = [x for x in range(10) if x % 2 == 0]

# 双重循环
pairs = [(x, y) for x in range(3) for y in range(3)]

# 等价于
pairs = []
for x in range(3):
    for y in range(3):
        pairs.append((x, y))
```

### 2.3 排序技巧

```python
# 内置 sorted 返回新列表，原列表不变
sorted([3, 1, 2])                    # [1, 2, 3]
sorted([3, 1, 2], reverse=True)      # [3, 2, 1]

# 按 key 排序
words = ["banana", "pie", "apple"]
sorted(words, key=len)               # ["pie", "apple", "banana"]

# 多级排序（先按长度，再按字母）
sorted(words, key=lambda x: (len(x), x))

# 对象列表排序
students = [("Alice", 85), ("Bob", 90), ("Alice", 80)]
sorted(students, key=lambda s: (-s[1], s[0]))  # 分数降序，姓名升序
```

### 2.4 其他常用操作

```python
# 切片赋值 — 可以增删改
lst = [1, 2, 3, 4]
lst[1:3] = [20, 30]       # [1, 20, 30, 4]
lst[1:3] = []             # [1, 4] — 删除
lst[1:1] = [100]          # [1, 100, 4] — 插入

# 堆栈（Stack）— LIFO
stack = []
stack.append(1)    # 入栈
stack.pop()        # 出栈

# 队列（Queue）— FIFO（但 list.pop(0) 是 O(n)，大数据量建议用 deque）
queue = []
queue.append(1)    # 入队
queue.pop(0)       # 出队（O(n)，不推荐）
```

---

## 3. 元组（tuple）

元组是**不可变**的有序序列，与列表类似但不可修改。

### 3.1 特性与用法

```python
# 创建
t = (1, 2, 3)
t = 1, 2, 3        # 括号可省略
t = (1,)           # 单元素元组，逗号不能省！

# 不可变性
t[0] = 100         # TypeError!

# 可哈希 — 可作为字典的 key 和集合的元素
point = (3, 4)
d = {point: "A"}   # 合法

# 解包
x, y = (3, 4)
a, *rest = (1, 2, 3, 4)
```

### 3.2 具名元组 namedtuple

```python
from collections import namedtuple

Point = namedtuple("Point", ["x", "y"])
p = Point(3, 4)
print(p.x, p.y)    # 3 4
```

---

## 4. 字典（dict）

字典是**键值对**的哈希映射，键必须可哈希，查找、插入、删除平均 O(1)。

### 4.1 常用方法

| 方法                              | 说明                            |
| ------------------------------- | ----------------------------- |
| `dict.items()`                  | 返回 (key, value) 对迭代器          |
| `dict.keys()`                   | 返回所有键                         |
| `dict.values()`                 | 返回所有值                         |
| `dict.get(key, default)`        | 安全取值，不存在时返回默认值                |
| `dict.copy(x)`                  | 浅拷贝字典                         |
| `dict.setdefault(key, default)` | 获取指定key的值（如果键不存在则设置默认值，返回最终值） |

| 方法 | 说明 |
|------|------|
| `len(d)` | 键值对数量 |
| `d[k]` | 获取值，键不存在抛 `KeyError` |
| `d.get(k, default)` | 获取值，不存在返回默认值 |
| `d[k] = v` | 设置键值对 |
| `d.update(other)` | 批量更新 |
| `del d[k]` | 删除键值对 |
| `k in d` | 判断键是否存在 |
| `d.keys()` | 返回所有键的视图 |
| `d.values()` | 返回所有值的视图 |
| `d.items()` | 返回所有键值对的视图 |
| `d.pop(k, default)` | 弹出并返回值 |
| `d.popitem()` | 弹出最后一项（Python 3.7+ 有序） |
| `d.setdefault(k, v)` | 键不存在则设置默认值 |

### 4.2 字典推导式

```python
# 基本形式
squares = {x: x**2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# 从两个列表创建
keys = ["a", "b", "c"]
vals = [1, 2, 3]
d = {k: v for k, v in zip(keys, vals)}

# 过滤
{ k: v for k, v in d.items() if v > 0 }
```

### 4.3 合并字典

```python
# Python 3.9+: | 运算符
d1 = {"a": 1}; d2 = {"b": 2}
d = d1 | d2        # {"a": 1, "b": 2}
d1 |= d2           # 原地更新 d1

# 更早版本
{**d1, **d2}
d1.update(d2)
```

### 4.4 有序性

Python 3.7+ 中字典保持**插入顺序**，`collections.OrderedDict` 通常不再需要。

---

## 5. 集合（set）

集合是**无序**、**不重复**的元素集合，基于哈希表实现。

### 5.1 常用方法

| 方法 | 说明 |
|------|------|
| `len(s)` | 元素个数 |
| `x in s` | 判断成员 |
| `s.add(x)` | 添加元素 |
| `s.remove(x)` | 删除元素，不存在抛异常 |
| `s.discard(x)` | 删除元素，不存在不报错 |
| `s.pop()` | 随机弹出一个元素 |
| `s.clear()` | 清空集合 |

### 5.2 集合运算

```python
a = {1, 2, 3}
b = {3, 4, 5}

a | b     # {1, 2, 3, 4, 5} — 并集（union）
a & b     # {3}              — 交集（intersection）
a - b     # {1, 2}           — 差集（difference）
a ^ b     # {1, 2, 4, 5}     — 对称差集（symmetric_difference）

# 方法形式
a.union(b)
a.intersection(b)
a.difference(b)
a.issubset(b)      # a 是否是 b 的子集
a.issuperset(b)    # a 是否是 b 的超集
a.isdisjoint(b)    # 是否无交集
```

### 5.3 去重与集合推导式

```python
# 快速去重
lst = [1, 2, 2, 3, 3, 3]
unique = list(set(lst))      # [1, 2, 3]（无序）

# 集合推导式
s = {x**2 for x in range(10) if x % 2 == 0}
# {0, 4, 16, 36, 64}
```

### 5.4 frozenset

不可变集合，可哈希，可作字典的键或集合的元素。

```python
fs = frozenset([1, 2, 3])
d = {fs: "immutable set"}
```

---

## 6. collections 模块

### 6.1 deque — 双端队列

```python
from collections import deque

dq = deque([1, 2, 3])
dq.append(4)        # 右端添加
dq.appendleft(0)    # 左端添加
dq.pop()            # 右端弹出
dq.popleft()        # 左端弹出（O(1)，优于 list.pop(0)）

dq.rotate(1)        # 向右轮转
```

### 6.2 Counter — 计数器

```python
from collections import Counter

c = Counter(["a", "b", "a", "c", "a"])
# Counter({'a': 3, 'b': 1, 'c': 1})

c.most_common(2)    # [('a', 3), ('b', 1)] — 最常见的 n 个

# 集合运算
c1 + c2             # 计数相加
c1 - c2             # 计数相减（非负）
c1 & c2             # 取最小值
c1 | c2             # 取最大值
```

### 6.3 defaultdict — 默认字典

```python
from collections import defaultdict

# 自动为不存在的键生成默认值
d = defaultdict(list)
d["a"].append(1)    # 不需要判断 "a" 是否存在
d["a"].append(2)
# defaultdict(<class 'list'>, {'a': [1, 2]})

d2 = defaultdict(int)
d2["x"] += 1        # 默认值为 0

d3 = defaultdict(lambda: "default")
```

### 6.4 OrderedDict — 有序字典

Python 3.7+ 中内置 `dict` 已保证顺序，`OrderedDict` 仅在需要特殊功能时使用：

```python
from collections import OrderedDict

od = OrderedDict()
od.move_to_end("key")        # 将某键移到最后
od.popitem(last=False)       # 弹出最先插入的项
```

---

## 7. 数据结构选择速查

| 场景 | 推荐结构 | 理由 |
|------|---------|------|
| 频繁尾部增删 | `list` | 动态数组，append/pop 均摊 O(1) |
| 频繁头部增删 | `deque` | 双端队列，两端 O(1) |
| 键值查找 | `dict` | 哈希表，平均 O(1) |
| 去重 / 集合运算 | `set` | 哈希集合 |
| 不可变序列 | `tuple` | 可哈希、安全、内存小 |
| 计数统计 | `Counter` | 专用工具，方便高效 |
| 需要默认值的字典 | `defaultdict` | 简化代码，避免 KeyError |
| 固定结构的数据记录 | `namedtuple` / `dataclass` | 可读性好，字段名访问 |

---

## 8. 不可变与可变性总结

| 结构 | 可变？ | 有序？ | 可哈希？ |
|------|--------|--------|---------|
| `str` | 否 | 是 | 是 |
| `list` | 是 | 是 | 否 |
| `tuple` | 否 | 是 | 元素均可哈希则是 |
| `dict` | 是 | 是（3.7+） | 否 |
| `set` | 是 | 否 | 否 |
| `frozenset` | 否 | 否 | 是 |
| `deque` | 是 | 是 | 否 |
| `Counter` | 是 | 否 | 否 |
