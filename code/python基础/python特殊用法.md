---
sr-due: 2026-10-19
sr-interval: 49
sr-ease: 270
---
#code

## 1. `functools`

### `functools.partial(func, *args, **kwargs)` —— 固定参数

**PCX 中 `partial` 的核心用途是解决 JAX 注册的签名不匹配问题。**

```python
# pcx/core/_module.py:61-63 —— 元类注册 pytree
jax.tree_util.register_pytree_with_keys(
    _cls,
    ...
    unflatten_func=functools.partial(
        _BaseModuleMeta.unflatten_module, cls=_cls
        #  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  原始函数
        #                                    ^^^^^^^^ 固定的关键字参数
    ),
)

# unflatten_module 的签名：
# def unflatten_module(aux_data, children, *, cls) -> "BaseModule":
#     ...

# JAX 调用的是: unflatten_func(aux_data, children)
# 而不是:       unflatten_func(aux_data, children, cls=xxx)

# partial 解决了这个矛盾：cls 被预先绑定，剩余签名恰好匹配 JAX 的调用约定
```

**`partial` 的原理（简化版）**：
```python
def partial(func, *more_args, **more_kwargs):
    def wrapper(*args, **kwargs): # 返回的wrapper只接受func的部分参数
        return func(*args, *more_args, **kwargs, **more_kwargs) # 实际调用的fun传入了更多参数（事先传入partial）
    return wrapper
```

### `functools.reduce(fn, iterable)` —— 累积计算

```python
# pcx/predictive_coding/_energy_module.py:45-46 —— 递归能量求和
def energy(self) -> jax.Array:
    return functools.reduce(
        lambda x, y: x + y,                              # 二元操作：加法
        (m.energy() for m in self.submodules(cls=EnergyModule))
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # 生成器：每个子模块的能量值
    )
```

**`reduce` 的展开等价写法**：

```python
# reduce(lambda x, y: x + y, (a, b, c, d))
# 等价于:
# (((a + b) + c) + d)
```

这里用 `reduce` 替代了 `sum()`，因为返回值是 JAX 数组而非 Python 数字。

---

## 2. `abc` / `contextlib` / `enum`

### `abc.ABCMeta` —— 抽象基类元类

```python
# pcx/core/_module.py:46 —— 元类继承 ABCMeta
class _BaseModuleMeta(abc.ABCMeta):
    def __new__(mcs, name, bases, dct):
        ...

# pcx/core/_parameter.py:24
class _BaseParamMeta(abc.ABCMeta):
    ...
```

PCX 的元类继承 `abc.ABCMeta` 而不是 `type`，这样基类中的 `@abc.abstractmethod` 才能正常工作（阻止直接实例化）。

### `@abc.abstractmethod` —— 强制子类实现

```python
# pcx/core/_parameter.py:84-90
class BaseParam(metaclass=_BaseParamMeta):
    @abc.abstractmethod
    def get(self):
        raise NotImplementedError()     # 子类不实现 → 实例化时报 TypeError

    @abc.abstractmethod
    def set(self, value):
        raise NotImplementedError()

# pcx/functional/_transform.py:98
class _BaseTransform(abc.ABC):  # abc.ABC 是 abc.ABCMeta 的语法糖
    @abc.abstractmethod
    def _t(self, *args, **kwargs):
        return NotImplemented
```

---

## 3. contextlib
### `@contextlib.contextmanager` —— 生成器式上下文管理器

```python
import contextlib

# pcx/utils/_misc.py:16-70
@contextlib.contextmanager
def step(module, status=None, *, clear_params=None):
    # 预处理
    if clear_params[0] is not None:
        module.clear_params(clear_params[0])
    tree_apply(lambda m: m._status.set(status[0]), ..., tree=module)

    yield       # ← 在这里交出控制权，执行 with 块内的代码

    # 后处理（清理）
    if clear_params[1] is not None:
        module.clear_params(clear_params[1])
    ...

# 用法：
with step(model, STATUS.INIT, clear_params=(None, VodeParam.Cache)):
    energy = model.energy()
#   with 块执行完后，自动清除 Cache
```

**等价于手动写 `__enter__` + `__exit__`**：

```python
class step:
    def __enter__(self):
        # yield 之前的代码
    def __exit__(self, *args):
        # yield 之后的代码
```

`@contextmanager` 让这个模式只需要一个函数加两段代码，比手写类简洁得多。

---

## 4. enum

### `IntEnum` —— 整数枚举

```python
from enum import IntEnum
# pcx/core/_module.py:149-152
class MODE(IntEnum):
	NONE = 0
	TRAIN = 1
	EVAL = 2
```

`IntEnum` = `int` + `Enum`。可以当作普通整数使用（`MODE.TRAIN == 1` 是 `True`），但有语义化名称和类型安全（不能随便赋值为 `5`）。

#### 枚举成员是枚举类的实例（Enum 特有）

`IntEnum`（及所有 `Enum` 子类）的成员本身就是枚举类的**实例**，而非普通类属性。这是 `Enum` 特有的行为，源于其元类 `EnumType`：它在创建类时会拦截 `TRAIN = 1` 这样的赋值，把每个条目转换成 `MODE` 的一个实例，只是该实例的 `.value = 1`。

```python
# 已验证
MODE.TRAIN                      # <MODE.TRAIN: 1>，本身就是 MODE 实例
MODE.TRAIN.value                # 1
type(MODE.TRAIN)                # <enum 'MODE'>
isinstance(MODE.TRAIN, MODE)    # True
```

---

## 5. `types.UnionType`

Python 3.10+ 支持用 `|` 写联合类型（如 `int | str`），运行时它的类型是 `types.UnionType`：

```python
from types import UnionType

def check_type(value, expected):
    if isinstance(expected, type | UnionType):
        return isinstance(value, expected)
    else:
        raise TypeError("expected 必须是类型或类型联合")

# 使用
check_type(42, int)           # True
check_type("hi", int | str)   # True
check_type(3.14, int | str)   # False
```

> `int` 和 `str` 都是 `type`类型，而 `int | str` 是 `UnionType` 类型
