# PCX 核心模块：`_static.py` 详解

> 源文件：`pcx/core/_static.py`

## 为什么需要静态参数？

`Module` 是 pytree，其属性会被 JAX 展平。但 JAX 只接受 arraylike 的动态叶子——字符串、整数、函数、配置对象等**非数组值**无法直接放进模块，否则传入 JAX/pcx 变换时会报错。

`StaticParam` 解决这一问题：把任意值包成一个"静态参数"。展平时其 `_value=None`（动态部分为空），真正的值存在 `_static_value` 里、作为 `aux_data`（静态）随对象一起携带，**不参与 JAX 追踪**。

```python
# 错误：函数不能直接作为模块属性
class M(pcax.Module):
    def __init__(self, x, f):
        self.x = pcax.Param(x)
        self.f = f            # ✗ 传入 jit 会出错

# 正确：用 static 包装
class M(pcax.Module):
    def __init__(self, x, f):
        self.x = pcax.Param(x)
        self.f = pcax.static(f)   # ✓
```

---

## 1. `StaticParam(BaseParam)`

```python
class StaticParam(BaseParam):
    def __init__(self, value=None):
        super().__init__(None)          # _value 设为 None（动态部分为空）
        self._static_value = value      # 真正值存在这里
    def get(self): return self._static_value
    def set(self, value): self._static_value = value
```

### `get` / `set`
重写基类方法，读写 `_static_value` 而非 `_value`。由于 `flatten_parameter` 只把 `_value` 当动态叶子，而 `_value=None`，所以 `StaticParam` 在 pytree 中不贡献任何动态叶子——它的 `_static_value`（连同其它属性）进 `aux_data`，对 JAX 不可见。

### 透明代理：让 `StaticParam` 用起来像被包装的值本身
重载大量特殊方法，转发到 `_static_value`：

| 方法 | 效果 |
|------|------|
| `__getattr__` | `sp.any_attr` -> `getattr(静态值, any_attr)` |
| `__contains__` | `key in sp` -> `key in 静态值` |
| `__iter__` | `for x in sp` -> 迭代静态值 |
| `__len__` | `len(sp)` -> `len(静态值)` |
| `__getitem__` / `__setitem__` | 下标读写 |
| `__call__` | 若静态值是函数，`sp(args)` 直接调用 |
| `__eq__` | 与（已 `get` 的）另一值比较 |
| `__repr__` | `StaticParam(repr(静态值))` |

```python
m.f = pcx.static(some_fn)
m.f(x, y)          # 等价于 some_fn(x, y)，靠 __call__

cfg = pcx.static({"lr": 0.1})
cfg["lr"]          # 靠 __getitem__
"lr" in cfg        # 靠 __contains__
```

### 关于在变换内修改静态值
pcx 默认**只追踪动态值**，故在 JAX 变换内对静态值的修改不会自动反映到变换外。考虑到 `jit` 要求每次调用的静态/动态结构一致，**不建议**在变换内永久改变静态值——pcx 目前的行为是变换内的改动是临时的。详见源码 DEV NOTE。

---

## 2. `static(x)` — 包装辅助函数

```python
def static(x):
    return x if isinstance(x, StaticParam) else StaticParam(x)
```
- 若 `x` 已是 `StaticParam`，原样返回（幂等，避免重复包装）。
- 否则包成新的 `StaticParam`。

这是用户最常用的入口：`self.f = pcx.static(f)`。

---

## 与参数体系的关系

- `StaticParam` 继承自 `BaseParam`，所以 `isinstance(sp, BaseParam)` 为真——会被 [_tree.py](pcx-tree.md) 的 ref/unref/extract 当作"参数叶子"定位。
- 但它**不是** `DynamicParam`，故默认的 `tree_extract` / `tree_inject`（`filter_fn=isinstance(x, DynamicParam)`）会**跳过**它。这正是期望行为：静态值不该被当成可优化张量提取/注入。

> `_static.py` 依赖 [_parameter.py](pcx-core-parameter.md)（`BaseParam`、`get`）。
