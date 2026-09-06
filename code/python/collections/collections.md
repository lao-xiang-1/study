# Python collections 模块

`collections` 是 Python 标准库中的**容器数据类型模块**，在 `list`、`dict`、`tuple`、`set` 等内置容器的基础上，提供了针对特定场景的扩展容器类型。它们大多数是内置容器的**子类**，因此保留了原有接口，同时补充了更便利的行为。

| 类型                                     | 基于      | 用途            |
| -------------------------------------- | ------- | ------------- |
| `namedtuple`                           | `tuple` | 带字段名的元组       |
| `deque`                                | —       | 双端队列（两端 O(1)） |
| `defaultdict`                          | `dict`  | 访问缺失键时自动生成默认值 |
| `Counter`                              | `dict`  | 可哈希对象的计数统计    |
| `OrderedDict`                          | `dict`  | 记录插入顺序，支持重排   |
| `ChainMap`                             | —       | 多个映射的链式视图     |
| `UserDict` / `UserList` / `UserString` | —       | 便于继承的自定义容器基类  |
| `collections.abc`                      | —       | 容器抽象基类（ABC）   |

---

## 1. namedtuple — 具名元组

`namedtuple` 是一个**工厂函数**，生成带字段名的 `tuple` 子类。它既有元组的轻量、不可变、可哈希，又可以用字段名访问，可读性远高于 `tuple[0]` 这种索引写法。

```python
from collections import namedtuple

Point = namedtuple("Point", ["x", "y"])   # 字段名可用列表或空格分隔的字符串
p = Point(3, 4)

p.x, p.y      # 3, 4     — 字段名访问
p[0], p[1]    # 3, 4     — 仍是元组，支持索引/切片/解包
```

### 1.1 常用方法与属性

| 方法 / 属性 | 说明 | 示例 |
|-----------|------|------|
| `._fields` | 字段名元组 | `('x', 'y')` |
| `._asdict()` | 转为普通字典 | `{'x': 3, 'y': 4}` |
| `._replace(**kw)` | 返回替换字段后的**新**元组 | `p._replace(x=10)` |
| `._make(iterable)` | 从可迭代对象构造 | `Point._make([1, 2])` |
| `._field_defaults` | 字段默认值（3.7+） | `{'z': 0}` |

> `_replace` / `_make` 以下划线开头是历史命名习惯，是**公共 API**，可以放心使用。它们返回新对象，不会修改原对象（因为元组不可变）。

### 1.2 默认值与字段重命名

```python
# 默认值：defaults 从右向左填充（3.7+）
Point = namedtuple("Point", ["x", "y", "z"], defaults=[0, 0])
Point(1)          # Point(x=1, y=0, z=0)

# rename=True：自动把非法字段名改成 _0, _1, ...
Data = namedtuple("Data", ["a", "class", "for"], rename=True)
Data._fields      # ('a', '_1', '_2')
```

### 1.3 namedtuple vs dataclass

| | `namedtuple` | `dataclass` |
|---|-------------|-------------|
| 可变性 | 不可变 | 默认可变，`frozen=True` 可冻结 |
| 内存 | 更小（继承 tuple） | 稍大（普通类） |
| 可哈希 | 是 | 仅 `frozen=True` 且字段均可哈希 |
| 继承 / 方法 | 受限 | 完整，可写方法、property |
| 比较运算 | 按值比较（元组语义） | 可配置 |

> 只是「不可变的轻量数据记录」用 `namedtuple`；需要自定义方法、类型校验或可变时用 `@dataclass`。

---

## 2. deque — 双端队列

`deque`（double-ended queue）在**两端**进行追加和弹出都是 **O(1)**，而 `list` 在头部插入/删除是 **O(n)**（需要移动全部元素）。因此需要「队列」时优先用 `deque`。

```python
from collections import deque

dq = deque([1, 2, 3])
dq.append(4)        # 右端追加 → [1, 2, 3, 4]
dq.appendleft(0)    # 左端追加 → [0, 1, 2, 3, 4]
dq.pop()            # 右端弹出 → 4
dq.popleft()        # 左端弹出 → 0
```

### 2.1 常用方法

| 方法 | 说明 | 复杂度 |
|------|------|--------|
| `append(x)` / `appendleft(x)` | 右/左端追加 | O(1) |
| `pop()` / `popleft()` | 右/左端弹出 | O(1) |
| `extend(iter)` / `extendleft(iter)` | 右/左端批量追加 | O(k) |
| `rotate(n=1)` | 轮转：`n>0` 向右，`n<0` 向左 | O(k) |
| `insert(i, x)` | 中间插入 | O(n) |
| `remove(x)` | 删除第一个匹配值 | O(n) |

```python
dq = deque([1, 2, 3, 4, 5])
dq.rotate(1)       # [5, 1, 2, 3, 4]  — 向右转一步
dq.rotate(-2)      # [2, 3, 4, 5, 1]  — 向左转两步

dq.extend([6, 7])        # 右侧扩展
dq.extendleft([-1, 0])   # 左侧扩展（注意顺序会被反转）
```

### 2.2 maxlen — 固定长度队列

构造时传入 `maxlen`，队列满时**自动从另一端丢弃**旧元素，适合「只保留最近 N 项」的场景：

```python
dq = deque(maxlen=3)
for i in range(5):
    dq.append(i)      # 依次得到 [0] [0,1] [0,1,2] [1,2,3] [2,3,4]
dq                    # deque([2, 3, 4], maxlen=3)

dq.maxlen             # 3（只读属性）
```

> `maxlen` 对 `appendleft` 同样生效：左侧满则丢弃右侧。也可用于实现滑动窗口、环形缓冲。

---

## 3. defaultdict — 默认字典

`defaultdict(default_factory)` 在访问**不存在的键**时，调用工厂函数生成默认值并插入，而不是抛出 `KeyError`。

```python
from collections import defaultdict

# 分组：键不存在时自动创建空列表
d = defaultdict(list)
for word in ["apple", "banana", "avocado"]:
    d[word[0]].append(word)
# {'a': ['apple', 'avocado'], 'b': ['banana']}

# 计数：int() 返回 0
c = defaultdict(int)
for ch in "hello":
    c[ch] += 1
# {'h': 1, 'e': 1, 'l': 2, 'o': 1}

# 自定义默认值
d = defaultdict(lambda: "unknown")
d["missing"]       # "unknown"
```

### 3.1 常用默认工厂

| 工厂 | 默认值 | 典型场景 |
|------|--------|---------|
| `list` | `[]` | 按键分组 |
| `set` | `set()` | 去重分组 |
| `int` | `0` | 计数 |
| `lambda: ...` | 任意 | 自定义默认值 |

### 3.2 注意点

- `default_factory` 必须是**无参可调用对象**。
- `d[k]` 访问缺失键会**产生副作用**：把 `k` 插入字典。
- `d.get(k)` **不会**触发默认工厂（`get` 不经过 `__missing__`），返回 `None` 或指定的默认值。

```python
d = defaultdict(list)
d["a"].append(1)   # 自动创建
d.get("b")         # None，且不插入 "b"
"b" in d           # False
```

> 原理：`defaultdict` 重写了 `__missing__`，仅在 `dict.__getitem__` 找不到键时被调用。这也是 `get` 不受影响的原因。

---

## 4. Counter — 计数器

`Counter` 是 `dict` 的子类，用于统计**可哈希对象**的出现次数。与普通字典不同，访问不存在的键返回 **0** 而不是抛异常。

```python
from collections import Counter

c = Counter("abracadabra")
# Counter({'a': 5, 'b': 2, 'r': 2, 'c': 1, 'd': 1})

c["a"]       # 5
c["z"]       # 0（不存在返回 0）
```

### 4.1 常用方法

| 方法 | 说明 |
|------|------|
| `most_common(n=None)` | 返回计数最多的 n 个 `(元素, 计数)` |
| `elements()` | 按计数展开为迭代器（每个元素重复其计数次） |
| `update(iterable)` | 累加计数 |
| `subtract(iterable)` | 减去计数（**允许负数**） |
| `total()` | 所有计数之和（3.10+） |

```python
c.most_common(2)        # [('a', 5), ('b', 2)]
list(c.elements())      # 无序展开
c.total()               # 11
c.subtract("aaa")       # a 的计数减 3，可为负
```

### 4.2 数学运算

`Counter` 支持 `+`、`-`、`&`、`|`，结果中**只保留正计数**：

```python
c1 = Counter(a=3, b=1)
c2 = Counter(a=1, b=2)

c1 + c2    # Counter({'a': 4, 'b': 3})  — 相加
c1 - c2    # Counter({'a': 2})          — 相减（负数丢弃）
c1 & c2    # Counter({'a': 1, 'b': 1})  — 对应位置取 min
c1 | c2    # Counter({'a': 3, 'b': 2})  — 对应位置取 max

+c1        # 去除非正计数；-c1 取负后再去除非正
```

---

## 5. OrderedDict — 有序字典

`OrderedDict` 记录键的**插入顺序**。Python 3.7+ 内置 `dict` 已保证顺序，因此日常场景无需使用；`OrderedDict` 仅在需要**重排**等特殊能力时才有价值。

```python
from collections import OrderedDict

od = OrderedDict()
od["a"] = 1; od["b"] = 2; od["c"] = 3
```

### 5.1 特有方法

| 方法 | 说明 |
|------|------|
| `move_to_end(key, last=True)` | 把键移到末尾（`last=False` 移到开头） |
| `popitem(last=True)` | 弹出末尾项（`last=False` 弹出最旧项） |

```python
od.move_to_end("a")      # a 移到最后：b, c, a
od.popitem(last=False)   # 弹出最旧项 ("b", 2)
```

### 5.2 用 OrderedDict 实现 LRU 缓存

```python
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)      # 标记为最近使用
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)   # 淘汰最久未使用
```

> 实际工程中，函数级缓存直接使用 `functools.lru_cache` / `functools.cache`；上面的 `OrderedDict` 版本用于理解 LRU 原理或需要自定义淘汰策略的场景。

---

## 6. ChainMap — 链式映射

`ChainMap` 把多个字典**链接**成一个逻辑视图，查找时按顺序从前到后搜索，**不复制数据**，只是保持对原字典的引用。典型用途是**配置分层**：命令行参数 > 环境变量 > 默认值。

```python
from collections import ChainMap

defaults = {"color": "red", "size": "M"}
overrides = {"color": "blue"}

cm = ChainMap(overrides, defaults)
cm["color"]    # "blue"  — 先命中 overrides
cm["size"]     # "M"     — 回退到 defaults
```

### 6.1 属性与方法

| 属性 / 方法 | 说明 |
|-----------|------|
| `.maps` | 底层字典的**可变列表**（前面的优先） |
| `.new_child(m=None)` | 返回新 `ChainMap`，在最前面插入新映射 |
| `.parents` | 去掉第一个映射后的新 `ChainMap` |

```python
cm.maps                  # [{'color': 'blue'}, {'color': 'red', 'size': 'M'}]
cm.new_child({"x": 1})   # 前面加一层
cm.parents               # ChainMap({'color': 'red', 'size': 'M'})
```

### 6.2 配置分层的典型写法

```python
import os
import argparse

defaults = {"host": "localhost", "port": 8080}
env = {"host": os.environ.get("HOST")}     # 可能为 None
cli = vars(parser.parse_args())            # 命令行参数

config = ChainMap(cli, env, defaults)
config["host"]   # 按 cli → env → defaults 的顺序取值
```

---

## 7. UserDict / UserList / UserString — 自定义容器基类

想**继承并定制** `dict` / `list` / `str` 时，直接继承内置类型往往不可靠——很多内置方法（如 `update`、`copy`）用 C 实现，**不会调用你覆写的特殊方法**。`UserDict` 等把数据存放在 `.data` 属性中，覆写方法能稳定生效。

```python
from collections import UserDict

class CaseInsensitiveDict(UserDict):
    def __setitem__(self, key, value):
        super().__setitem__(key.lower(), value)   # 键统一转小写

d = CaseInsensitiveDict()
d["Name"] = "Alice"
d["name"]            # "Alice"  — 大小写不敏感
d.data               # {'name': 'Alice'}  — 数据实际存于 .data
```

> 更现代的做法是继承 `collections.abc` 的抽象基类（如 `MutableMapping`）来实现自定义容器；`UserDict` 等则适合快速包装已有行为。

---

## 8. collections.abc — 抽象基类

`collections.abc` 提供一系列**容器抽象基类（ABC）**，用于：

1. **`isinstance` 测试**——判断对象是否满足某类接口；
2. **自定义容器**——继承 ABC 可自动获得一套默认方法（只需实现少数核心方法）。

```python
from collections.abc import Iterable, Mapping, Sequence, Hashable

isinstance([1, 2], Iterable)     # True
isinstance({}, Mapping)          # True
isinstance((1, 2), Sequence)     # True
isinstance(1, Hashable)          # True
```

### 8.1 常见 ABC 层级

```
Iterable
├── Iterator
└── Collection
    ├── Sequence        →  list, tuple, range, str
    ├── Mapping         →  dict
    │   └── MutableMapping
    └── Set             →  set, frozenset
```

### 8.2 用 ABC 自定义容器

只实现 `__getitem__`、`__setitem__`、`__delitem__`、`__iter__`、`__len__` 五个核心方法，`MutableMapping` 会自动补全 `get`、`keys`、`items`、`pop` 等：

```python
from collections.abc import MutableMapping

class MyDict(MutableMapping):
    def __init__(self):
        self._data = {}

    def __getitem__(self, key):       return self._data[key]
    def __setitem__(self, key, val):  self._data[key] = val
    def __delitem__(self, key):       del self._data[key]
    def __iter__(self):               return iter(self._data)
    def __len__(self):                return len(self._data)

d = MyDict()
d["a"] = 1
d.get("a")        # 1  — 由 ABC 自动提供
list(d.items())   # [('a', 1)]
```

---

## 9. 选择速查

| 场景 | 推荐 | 理由 |
|------|------|------|
| 需要队列 / 频繁头部操作 | `deque` | 两端 O(1)，优于 `list.pop(0)` |
| 只保留最近 N 项 | `deque(maxlen=N)` | 自动丢弃旧项 |
| 轻量不可变数据记录 | `namedtuple` | 字段名可读、内存小、可哈希 |
| 按键分组 | `defaultdict(list)` | 免去 `setdefault` 判断 |
| 计数 / 频率统计 | `Counter` | `most_common`、`+ - & \|` 运算 |
| 计数型访问 | `defaultdict(int)` | 缺失键默认为 0 |
| 需要重排键序 / LRU | `OrderedDict` | `move_to_end`、`popitem(last=False)` |
| 多来源配置分层 | `ChainMap` | 视图合并、不复制、按优先级查找 |
| 自定义容器 | `UserDict` 或 `collections.abc` | 覆写方法稳定生效 |
