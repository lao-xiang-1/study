# Python 类特殊方法（Dunder Methods）总结

> 参考来源：[Python 3.14 官方文档 - 数据模型](https://docs.python.org/zh-cn/3.14/reference/datamodel.html#special-method-names)
>
> 整理日期：2026-07-31

特殊方法（又称魔术方法、双下方法）是 Python 中以双下划线开头和结尾的方法（如 `__init__`、`__repr__`），用于让自定义类支持内置函数、运算符和语法（如 `len()`、`+`、`with`、`for` 等）。

## 0. 调用规则总览

1. **特殊方法查找在类型上而非实例上**：解释器隐式调用时，会查找 `type(x).__method__`，而非 `x.__method__`，所以直接在实例上覆盖无效。
2. **`NotImplemented`**：数值与比较方法在运算不支持时应返回 `NotImplemented`（不是抛异常），以便解释器尝试反向操作或回退。
3. **设为 `None`**：把特殊方法显式设为 `None` 表示该操作不可用（如 `__iter__ = None` 让对象不可迭代）。
4. **隐式调用优先**：`for x in obj` 不会调用 `obj.__iter__()`，而是调用 `type(obj).__iter__(obj)`。

---

## 1. 基本定制（Basic Customization）

| 方法 | 签名 | 调用方式 / 用途 | 返回值 |
|------|------|----------------|--------|
| `__new__` | `__new__(cls, ...)` | 创建实例（静态方法，特例）。先于 `__init__` 调用 | 新对象实例（通常是 `cls` 的实例） |
| `__init__` | `__init__(self, ...)` | 实例创建后初始化 | 只能返回 `None`，否则 `TypeError` |
| `__del__` | `__del__(self)` | 实例被销毁时调用（析构器）；不推荐做资源清理 | 无；异常会被忽略并打印警告 |
| `__repr__` | `__repr__(self)` | `repr(obj)`；交互式直接显示。返回"官方"字符串，应尽量为有效 Python 表达式 | 字符串 |
| `__str__` | `__str__(self)` | `str(obj)`、`print(obj)`；`__format__` 默认实现回退到此 | 字符串 |
| `__bytes__` | `__bytes__(self)` | `bytes(obj)`，生成字节串表示 | `bytes` |
| `__format__` | `__format__(self, format_spec)` | `format(obj, spec)`、f-string、`str.format()` | 字符串 |
| `__lt__` | `__lt__(self, other)` | `x < y` | `True`/`False` 或 `NotImplemented` |
| `__le__` | `__le__(self, other)` | `x <= y` | 同上 |
| `__eq__` | `__eq__(self, other)` | `x == y`。默认实现 `x is y` | 同上 |
| `__ne__` | `__ne__(self, other)` | `x != y`。默认委托 `__eq__` 并取反 | 同上 |
| `__gt__` | `__gt__(self, other)` | `x > y` | 同上 |
| `__ge__` | `__ge__(self, other)` | `x >= y` | 同上 |
| `__hash__` | `__hash__(self)` | `hash(obj)`；用于字典键/集合元素。须与 `__eq__` 一致：相等对象哈希必相等 | 整数 |
| `__bool__` | `__bool__(self)` | `bool(obj)`、真值检测。未定义则回退到 `__len__`（非零为真） | `True`/`False` |

> **关键提示**：若重载了 `__eq__` 但未定义 `__hash__`，则 `__hash__` 会被隐式设为 `None`，实例不可哈希。

```python
class Vector:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __repr__(self):
        return f"Vector({self.x!r}, {self.y!r})"

    def __eq__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __bool__(self):
        return bool(self.x or self.y)
```

---

## 2. 自定义属性访问（Customizing Attribute Access）

| 方法 | 签名 | 调用时机 / 用途 | 返回值 |
|------|------|----------------|--------|
| `__getattr__` | `__getattr__(self, name)` | **仅当**默认查找失败（`AttributeError`）时才触发 | 属性值，或引发 `AttributeError` |
| `__getattribute__` | `__getattribute__(self, name)` | **无条件**调用所有属性访问。需谨慎避免无限递归（用 `super()`） | 属性值 |
| `__setattr__` | `__setattr__(self, name, value)` | 属性赋值 `obj.x = v` | 无 |
| `__delattr__` | `__delattr__(self, name)` | `del obj.x` | 无 |
| `__dir__` | `__dir__(self)` | `dir(obj)` | 可迭代对象 |

### 描述器协议（Descriptor Protocol）

| 方法             | 签名                                    | 用途                                              | 返回值 |
| -------------- | ------------------------------------- | ----------------------------------------------- | --- |
| `__get__`      | `__get__(self, instance, owner=None)` | 获取属性。`owner` 为所有者类，`instance` 为实例（类访问时为 `None`） | 属性值 |
| `__set__`      | `__set__(self, instance, value)`      | 设置属性。**定义此方法会变为"数据描述器"**                        | 无   |
| `__delete__`   | `__delete__(self, instance)`          | 删除属性。**定义此方法也变为"数据描述器"**                        | 无   |
| `__set_name__` | `__set_name__(self, owner, name)`     | 所有者类创建时自动调用，告知被赋值的属性名                           | 无   |

> **核心规则**：数据描述器（有 `__set__` 或 `__delete__`）总是优先于实例字典 `__dict__`；非数据描述器（仅有 `__get__`）会被实例字典覆盖。`property` 就是数据描述器的典型实现。

---

## 3. 自定义类创建（Customizing Class Creation）

| 方法 | 签名 | 用途 | 返回值 |
|------|------|------|--------|
| `__init_subclass__` | `__init_subclass__(cls, **kwargs)` | 类方法。所在类被继承（派生子类）时自动调用 | 无 |
| `__class_getitem__` | `__class_getitem__(cls, key)` | 类方法。支持泛型语法 `list[int]`，返回 `GenericAlias` | `GenericAlias` |
| `__mro_entries__` | `__mro_entries__(self, bases)` | 当基类不是 `type` 实例时调用，返回元组替代该基类 | 类的元组 |
| `__instancecheck__` | `__instancecheck__(self, instance)` | 元类方法，重载 `isinstance()` | 布尔 |
| `__subclasscheck__` | `__subclasscheck__(self, subclass)` | 元类方法，重载 `issubclass()` | 布尔 |

```python
class Plugin:
    registry = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Plugin.registry.append(cls)  # 子类自动注册
```

---

## 4. 模拟可调用对象（Emulating Callable Objects）

| 方法 | 签名 | 用途 | 返回值 |
|------|------|------|--------|
| `__call__` | `__call__(self[, args...])` | 使实例可调用：`obj(arg1, arg2)` 等价于 `type(obj).__call__(obj, arg1, arg2)` | 任意 |

```python
class Adder:
    def __init__(self, n): self.n = n
    def __call__(self, x): return x + self.n

add5 = Adder(5)
add5(10)  # 15
```

---

## 5. 模拟容器类型（Emulating Container Types）

| 方法 | 签名 | 用途 | 返回值 |
|------|------|------|--------|
| `__len__` | `__len__(self)` | `len(obj)`；未定义 `__bool__` 时返回 0 视为假 | `>= 0` 的整数 |
| `__length_hint__` | `__length_hint__(self)` | `operator.length_hint()`；性能优化的长度估计 | 整数或 `NotImplemented` |
| `__getitem__` | `__getitem__(self, key)` | `obj[key]`。序列：整数/切片；映射：键 | 值；不当索引引发 `TypeError`/`LookupError` |
| `__setitem__` | `__setitem__(self, key, value)` | `obj[key] = value` | 无 |
| `__delitem__` | `__delitem__(self, key)` | `del obj[key]` | 无 |
| `__missing__` | `__missing__(self, key)` | `dict` 子类在 `__getitem__` 找不到键时调用 | 对应值 |
| `__iter__` | `__iter__(self)` | `iter(obj)`、`for x in obj`。映射应迭代键，序列迭代值 | 新的迭代器对象 |
| `__next__` | `__next__(self)` | 迭代器取下一项；无项时引发 `StopIteration` | 下一项 |
| `__reversed__` | `__reversed__(self)` | `reversed(obj)`；未提供则回退到 `__len__` + `__getitem__` | 逆向迭代器 |
| `__contains__` | `__contains__(self, item)` | `x in obj` / `x not in obj`；映射基于键检测 | 布尔 |

> **迭代协议**：迭代器必须同时实现 `__iter__`（返回 self）和 `__next__`；可迭代对象只需 `__iter__` 返回一个迭代器。

---

## 6. 模拟数字类型（Emulating Numeric Types）

### 6.1 双目算术运算符（正向）

| 方法 | 对应运算 |
|------|----------|
| `__add__(self, other)` | `+` |
| `__sub__(self, other)` | `-` |
| `__mul__(self, other)` | `*` |
| `__matmul__(self, other)` | `@`（矩阵乘法，PEP 465） |
| `__truediv__(self, other)` | `/` |
| `__floordiv__(self, other)` | `//` |
| `__mod__(self, other)` | `%` |
| `__divmod__(self, other)` | `divmod()` |
| `__pow__(self, other[, modulo])` | `**`、`pow()`（支持可选 `modulo`） |
| `__lshift__(self, other)` | `<<` |
| `__rshift__(self, other)` | `>>` |
| `__and__(self, other)` | `&` |
| `__xor__(self, other)` | `^` |
| `__or__(self, other)` | `\|` |

### 6.2 反射（逆向）双目运算符

左操作数返回 `NotImplemented` 时，或右操作数是左操作数类型的子类时调用，方法名前加 `r`：

- `__radd__`、`__rsub__`、`__rmul__`、`__rmatmul__`、`__rtruediv__`、`__rfloordiv__`、`__rmod__`、`__rdivmod__`、`__rpow__`、`__rlshift__`、`__rrshift__`、`__rand__`、`__rxor__`、`__ror__`

> 例如 `x + y`：先尝试 `type(x).__add__(x, y)`，返回 `NotImplemented` 则尝试 `type(y).__radd__(y, x)`。

### 6.3 增强赋值运算符（原地运算）

| 方法 | 对应运算 |
|------|----------|
| `__iadd__(self, other)` | `+=` |
| `__isub__(self, other)` | `-=` |
| `__imul__(self, other)` | `*=` |
| `__imatmul__(self, other)` | `@=` |
| `__itruediv__(self, other)` | `/=` |
| `__ifloordiv__(self, other)` | `//=` |
| `__imod__(self, other)` | `%=` |
| `__ipow__(self, other[, modulo])` | `**=` |
| `__ilshift__(self, other)` | `<<=` |
| `__irshift__(self, other)` | `>>=` |
| `__iand__(self, other)` | `&=` |
| `__ixor__(self, other)` | `^=` |
| `__ior__(self, other)` | `\|=` |

> 应就地修改 `self` 并返回；若未定义或返回 `NotImplemented`，回退到 `__add__` + `__radd__` 等。

### 6.4 单目运算符

| 方法 | 签名 | 对应运算 |
|------|------|----------|
| `__neg__` | `__neg__(self)` | `-x`（取负） |
| `__pos__` | `__pos__(self)` | `+x`（取正） |
| `__abs__` | `__abs__(self)` | `abs(x)` |
| `__invert__` | `__invert__(self)` | `~x`（按位取反） |

### 6.5 类型转换与舍入

| 方法 | 签名 | 调用方式 | 返回值 |
|------|------|---------|--------|
| `__complex__` | `__complex__(self)` | `complex(obj)` | 复数 |
| `__int__` | `__int__(self)` | `int(obj)` | 整数 |
| `__float__` | `__float__(self)` | `float(obj)` | 浮点数 |
| `__index__` | `__index__(self)` | `operator.index()`、切片索引、`bin()`/`hex()`/`oct()`；无损转整数 | 整数 |
| `__round__` | `__round__(self, ndigits=None)` | `round(obj, n)` | 通常与 `self` 同类型；`ndigits=None` 时返回整数 |
| `__trunc__` | `__trunc__(self)` | `math.trunc(obj)` | 整数（向零截断） |
| `__floor__` | `__floor__(self)` | `math.floor(obj)` | 整数（向下取整） |
| `__ceil__` | `__ceil__(self)` | `math.ceil(obj)` | 整数（向上取整） |

> 若未定义 `__int__`、`__float__`、`__complex__`，对应内置函数会回退到 `__index__`。

```python
class Temperature:
    def __init__(self, celsius): self.c = celsius

    def __add__(self, other):
        if isinstance(other, Temperature):
            return Temperature(self.c + other.c)
        return NotImplemented

    def __float__(self):
        return float(self.c)

    def __round__(self, ndigits=None):
        return Temperature(round(self.c, ndigits)) if ndigits else round(self.c)
```

---

## 7. With 语句上下文管理器（Context Managers）

| 方法 | 签名 | 用途 | 返回值 |
|------|------|------|--------|
| `__enter__` | `__enter__(self)` | 进入 `with` 块；返回值绑定到 `as` 子句目标 | 上下文对象 |
| `__exit__` | `__exit__(self, exc_type, exc_value, traceback)` | 退出 `with` 块；三个参数描述导致退出的异常（无异常时均为 `None`） | 返回真值则**抑制**异常，否则异常正常传播 |

> `__exit__` 不应重新引发传入的异常；那是调用方的责任。

```python
class ManagedFile:
    def __init__(self, filename, mode='r'):
        self.filename, self.mode = filename, mode

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        return False  # 不抑制异常


with ManagedFile('data.txt') as f:
    data = f.read()
```

---

## 8. 协程与异步（Coroutines & Async）

### 8.1 可等待对象

| 方法 | 签名 | 调用方式 | 返回值 |
|------|------|---------|--------|
| `__await__` | `__await__(self)` | `await obj` 表达式 | 迭代器（驱动协程执行） |

### 8.2 异步迭代器

| 方法 | 签名 | 调用方式 | 返回值 |
|------|------|---------|--------|
| `__aiter__` | `__aiter__(self)` | `async for x in obj` | 异步迭代器对象（须实现 `__aiter__` + `__anext__`） |
| `__anext__` | `__anext__(self)` | `async for` 每次迭代 | 可等待对象（awaitable）；结束时引发 `StopAsyncIteration` |

### 8.3 异步上下文管理器

| 方法 | 签名 | 调用方式 | 返回值 |
|------|------|---------|--------|
| `__aenter__` | `__aenter__(self)` | 进入 `async with` 块 | 可等待对象；其结果绑定到 `as` 目标 |
| `__aexit__` | `__aexit__(self, exc_type, exc_value, traceback)` | 退出 `async with` 块 | 可等待对象；返回真值则抑制异常 |

```python
class AsyncDBConnection:
    async def __aenter__(self):
        self.conn = await connect()
        return self.conn

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.conn.close()
        return False


async def fetch():
    async with AsyncDBConnection() as db:
        return await db.query('SELECT 1')
```

---

## 9. 拷贝与序列化（Copying & Pickling）

> 这些方法主要定义在 `copy` 和 `pickle` 模块的协议中，非数据模型页面主体内容，但属于常用特殊方法。

### 9.1 拷贝协议（`copy` 模块）

| 方法 | 签名 | 调用方式 | 用途 |
|------|------|---------|------|
| `__copy__` | `__copy__(self)` | `copy.copy(obj)` | 浅拷贝 |
| `__deepcopy__` | `__deepcopy__(self, memo)` | `copy.deepcopy(obj)` | 深拷贝；`memo` 是已拷贝对象字典，避免循环引用 |

### 9.2 Pickle 协议（`pickle` 模块）

| 方法 | 签名 | 用途 |
|------|------|------|
| `__getstate__` | `__getstate__(self)` | 返回序列化状态（默认用 `__dict__`） |
| `__setstate__` | `__setstate__(self, state)` | 反序列化时恢复状态 |
| `__reduce__` | `__reduce__(self)` | 返回重组对象所需的元组 `(callable, args, state, ...)` |
| `__reduce_ex__` | `__reduce_ex__(self, protocol)` | 指定协议版本的 `__reduce__`，默认委托 `__reduce__` |

---

## 10. 类模式匹配与缓冲区（Pattern Matching & Buffer）

| 名称 | 类型 | 用途 |
|------|------|------|
| `__match_args__` | 类变量 | 模式匹配中的位置参数名元组，`match` 语句按位置匹配 |
| `__buffer__` | 方法 | 缓冲区协议（PEP 688），由 `memoryview` 等使用 |

---

## 11. 其他常用类属性

| 名称 | 类型 | 用途 |
|------|------|------|
| `__slots__` | 类变量 | 显式声明实例数据成员，禁止 `__dict__` 和 `__weakref__`（除非显式声明），节省内存、加速查找 |
| `__dict__` | 实例属性 | 存储实例可写属性 |
| `__class__` | 实例属性 | 实例所属的类 |
| `__mro__` | 类属性 | 方法解析顺序元组 |
| `__subclasses__` | 类方法 | 返回直接子类列表（按定义顺序） |
| `__annotations__` | 类/函数属性 | 类型标注字典 |

---

## 12. 速查表：常用内置函数 ↔ 特殊方法

| 内置/语法 | 对应特殊方法 |
|-----------|--------------|
| `len(obj)` | `__len__` |
| `bool(obj)` | `__bool__`（回退 `__len__`） |
| `repr(obj)` / 交互式显示 | `__repr__` |
| `str(obj)` / `print(obj)` | `__str__`（回退 `__repr__`） |
| `format(obj, spec)` / f-string | `__format__` |
| `bytes(obj)` | `__bytes__` |
| `int(obj)` / `float(obj)` / `complex(obj)` | `__int__` / `__float__` / `__complex__`（回退 `__index__`） |
| `abs(obj)` | `__abs__` |
| `round(obj, n)` | `__round__` |
| `hash(obj)` | `__hash__` |
| `==` / `!=` / `<` / `<=` / `>` / `>=` | `__eq__` / `__ne__` / `__lt__` / `__le__` / `__gt__` / `__ge__` |
| `+x` / `-x` / `~x` | `__pos__` / `__neg__` / `__invert__` |
| `x + y` 等 | `__add__` 等（反向 `__radd__`） |
| `x += y` 等 | `__iadd__` 等（回退 `__add__`） |
| `obj[key]` / `obj[k]=v` / `del obj[k]` | `__getitem__` / `__setitem__` / `__delitem__` |
| `x in obj` | `__contains__`（回退 `__iter__`） |
| `for x in obj` / `iter(obj)` | `__iter__`（回退 `__getitem__`） |
| `next(it)` | `__next__` |
| `reversed(obj)` | `__reversed__`（回退 `__len__`+`__getitem__`） |
| `obj(...)` | `__call__` |
| `dir(obj)` | `__dir__` |
| `with obj as x:` | `__enter__` / `__exit__` |
| `await obj` | `__await__` |
| `async for x in obj` | `__aiter__` / `__anext__` |
| `async with obj as x:` | `__aenter__` / `__aexit__` |
| `copy.copy(obj)` / `copy.deepcopy(obj)` | `__copy__` / `__deepcopy__` |
| `pickle.dumps(obj)` / `pickle.loads` | `__getstate__` / `__setstate__` / `__reduce__` |

---

## 13. 实现建议

1. **保持方法一致性**：`__eq__` 与 `__hash__` 必须协同——重写 `__eq__` 时记得定义 `__hash__`（除非确需不可哈希）。
2. **返回 `NotImplemented` 而非抛异常**：数值/比较方法返回 `NotImplemented`，让解释器尝试反向运算，否则会破坏子类互操作。
3. **`__repr__` 应可重建**：理想情况下 `eval(repr(obj)) == obj`，至少要清晰可读。
4. **`__str__` 面向用户**：默认回退到 `__repr__`，定义时关注可读性。
5. **慎用 `__getattribute__`**：会拦截所有属性访问，极易无限递归，必须用 `super().__getattribute__(name)`。
6. **数据描述器优先于实例字典**：自定义描述器时若需"实例可覆盖"，则只定义 `__get__`（非数据描述器）。
7. **优先用 `@contextlib.contextmanager` 或 `contextlib.abstractcontextmanager`**：简单场景可避免手写 `__enter__`/`__exit__`。
8. **`__slots__` 节省内存**：大量实例时显著省内存，但会限制动态属性添加，且影响多重继承。

---

## 14. 参考链接

- 原文：[Python 3.14 数据模型 - 特殊方法名称](https://docs.python.org/zh-cn/3.14/reference/datamodel.html#special-method-names)
- [PEP 343 - with 语句](https://peps.python.org/pep-0343/)
- [PEP 465 - 矩阵乘法运算符 `@`](https://peps.python.org/pep-0465/)
- [PEP 492 - async/await](https://peps.python.org/pep-0492/)
- [PEP 688 - 缓冲区协议](https://peps.python.org/pep-0688/)
