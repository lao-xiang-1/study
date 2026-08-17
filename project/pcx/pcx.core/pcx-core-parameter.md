# PCX 核心模块：`_parameter.py` 详解

> 源文件：`pcx/core/_parameter.py`

pcx 受 objax 与 equinox 启发，核心思想是：**把每个 JAX 数组包装进一个 `Param` 对象，使 pcx 能在保留面向对象写法的同时，透过 JAX 变换追踪这些张量**，而不必严格遵守 JAX 的纯函数式范式。

本文件定义了参数类的继承体系与相关工具函数。

---

## 参数类继承关系

```
BaseParam   (抽象基类, 元类 _BaseParamMeta 自动注册为 pytree)
├── DynamicParam         (标记类: _value 被当作"动态"叶子追踪)
│   ├── Param            (单个张量/值)
│   └── ParamDict        (按名存储多个张量)
└── StaticParam          (定义在 _static.py: _value=None, 静态值另存)
```

> `BaseParam` 同时是判断"某对象是否为参数"的依据（`isinstance(x, BaseParam)`），这也是 [_tree.py](pcx-tree.md) 中 ref/unref/extract/inject 定位叶子的方式。

---

## 1. `_BaseParamMeta` — 元类

```python
class _BaseParamMeta(abc.ABCMeta):
    def __new__(mcs, name, bases, dct):
        _cls = super().__new__(mcs, name, bases, dct)
        jax.tree_util.register_pytree_with_keys(_cls, ...)
        return _cls
```

### 展平规则（与 Module 不同！）
参数展平时**只把 `_value` 当动态叶子**，其余属性全部进 `aux_data`（静态）：

| 方法 | children（动态叶子） | aux_data（静态） |
|------|---------------------|------------------|
| `flatten_parameter` | `(param._value,)` | `__dict__` 去掉 `_value` 后的副本 |
| `flatten_parameter_with_keys` | `((GetAttrKey("value"), param._value),)` | 同上 |
| `unflatten_parameter` | 由 `aux_data` + `children[0]` 重建对象（`object.__new__` 绕过 `__init__`） | — |

**意义**：JAX 只追踪 `_value`（张量），而参数的其它元信息不参与追踪、也不会触发重编译。

---

## 2. `BaseParam` — 抽象基类

```python
class BaseParam(metaclass=_BaseParamMeta):
    def __init__(self, value=None):
        self._value = value

    @abc.abstractmethod
    def get(self): ...
    @abc.abstractmethod
    def set(self, value): ...

    def __bool__(self):
        raise TypeError(...)
```

- `__init__`：把传入值存到 `_value`（只有这样的值会被 pcx 当作动态值追踪）。
- `get` / `set`：抽象方法，子类必须实现。
- `__bool__`：**故意抛错**。防止把参数误用在 `if param:` 这类布尔判断中（应使用 `is None` / `is not None`），避免形状含 0 的张量导致难以排查的 bug。

---

## 3. `DynamicParam` — 动态参数标记类

```python
class DynamicParam(BaseParam):
    pass
```

本身无新逻辑，仅作为"其 `_value` 需被 JAX 当作动态叶子追踪"的类型标记。`_tree.py` 中 `tree_extract` / `tree_inject` 的默认 `filter_fn` 就是 `isinstance(x, DynamicParam)`——即默认只提取/注入动态参数，跳过 `StaticParam`。

---

## 4. `Param` — 单个动态值（最常用）

```python
class Param(DynamicParam):
    def get(self) -> jax.Array: return self._value
    def set(self, value) -> "Param":
        self._value = value
        return self
```

### 核心设计：运算符全面重载，让 `Param` 用起来像裸数组
`Param` 重载了几乎所有特殊方法，把操作转发给 `self._value`：

| 类别 | 重载的方法 |
|------|-----------|
| 算术 | `+ - * / // % ** @`（含 `r` 反向、`i` 就地版本），`neg/pos/abs/invert` |
| 比较 | `== != < <= > >=` |
| 位运算 | `& \| ^ << >>`（含反向） |
| 其它 | `__getitem__`、`__round__`、`__array__` |
| 属性代理 | `__getattr__` -> `getattr(self._value, name)` |
| 属性 | `dtype`、`shape`、`ndim` |

> **关键细节**：二元运算的右操作数都先经过 `get(other)` 处理，所以 `param + other` 无论 `other` 是裸数组还是另一个 `Param` 都能正确工作。

```python
w = Param(jnp.array([1.0, 2.0]))
y = w * 2 + 1                  # 无需 w.get()
y2 = w + Param(jnp.array([0.1, 0.2]))
w += 1                         # 就地修改 _value（靠 __iadd__）
```

### `__repr__`
若 `_value` 是 `jax.Array`，打印 `Param([shape], dtype)`；否则打印 `Param(repr(value))`。

> Python 只在**类**上查找特殊方法，因此这些 `__xxx__` 必须显式定义，不能靠 `__getattr__` 兜底。

---

## 5. `ParamDict` — 字典式动态参数

```python
class ParamDict(DynamicParam):
    def __init__(self, value: Dict = None): ...
    def __getitem__(self, key): ...
    def __setitem__(self, key, value): ...
    def __contains__(self, key): ...
    def get(self, key=None, default=None): ...
    def set(self, value): ...
```

把**多个具名张量**收纳进单个参数，对外像字典。适用场景：一个状态参数需同时持有多个子张量（如多种梯度缓存、分组缓冲区）。

- `__setitem__`：若 `_value is None`（被"清空"过），先重置为 `{}` 再写入。
- `get(key, default)`：
  - `key=None` -> 返回整个内部字典
  - `key` 非空 -> 返回 `self._value.get(key, default)`
- `set(value)`：整体替换内部字典。

```python
buf = ParamDict()
buf["momentum"] = jnp.zeros((3,))
buf["velocity"]  = jnp.zeros((3,))
"momentum" in buf     # True
buf.get("momentum")   # 取出对应数组
```

---

## 6. `ParamCache` — 缓存标记哨兵

```python
class ParamCache:
    pass
```

无任何行为的哨兵类。用来**标记**"此参数仅作临时缓存用"。配合 `isinstance` 过滤，可在遍历时区分真正要持久化/优化的参数与一次性缓存参数。

---

## 7. 工具函数 `get` / `set`

### `get(x)` — 解包辅助函数
```python
def get(x):
    return x.get() if isinstance(x, BaseParam) else x
```
若 `x` 是参数则返回其值，否则原样返回。用于"不确定对方是参数还是裸值"的歧义场景，是 `Param` 运算符重载内部反复调用的基础工具。

```python
get(Param(1.0))   # 1.0
get(2.0)          # 2.0
```

### `set(obj, x)` — 赋值辅助函数
```python
def set(obj, x):
    if isinstance(obj, BaseParam):
        obj.set(get(x))      # 主要用法：给参数赋（已解包的）值
    else:
        ...                  # 见下注
    return obj
```
- **主要用法**：`obj` 是参数 -> 调 `obj.set(get(x))`（先解包 `x`），返回 `obj`。
- 文档约定：若 `obj` 不是参数，应"返回新值本身"。

```python
# pcx/utils/_optim.py 中的实际用法
set(g, g * scale_by)                            # g 是 Param -> g.set(g*scale_by)
set(p, eqx.apply_updates(get(p), get(u)))       # 用更新后的值写回 p
```

> ⚠️ **注意**：当前实现的 `else` 分支为 `obj = set(x)`（仅传一个参数的递归调用），而 `set` 的签名要求两个参数，运行时会因缺少参数而报 `TypeError`。这与文档"返回新值本身"的描述不一致。代码库中（如 `_optim.py`）均以参数为第一参数调用，走的是参数分支，故该 bug 在实践中未被触发；但若直接对非参数对象调用 `set`，需留意此问题。

---

## 小结

| 概念 | 作用 |
|------|------|
| `BaseParam` | 所有参数的抽象基类 + pytree 注册入口 |
| `DynamicParam` | "动态值被追踪"的类型标记 |
| `Param` | 单张量包装，重载运算符使其如裸数组般可用 |
| `ParamDict` | 多具名张量的字典式包装 |
| `ParamCache` | 标记临时缓存参数的哨兵 |
| `get` / `set` | 处理"参数 vs 裸值"歧义的工具函数 |
