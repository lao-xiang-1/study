# PCX 核心模块：`_module.py` 详解

> 源文件：`pcx/core/_module.py`

本文件定义了 pcx 中所有"模块"的基类。与 equinox 类似，每个 `Module` 都是一个 JAX pytree，可以作为其它模块和参数的容器。其核心思想是：**让带状态的神经网络模块能像普通 Python 对象一样使用，同时仍可被 JAX 变换（jit/grad/...）追踪**。

---

## 1. `_BaseModuleMeta` — 元类：自动注册为 JAX pytree

```python
class _BaseModuleMeta(abc.ABCMeta):
    def __new__(mcs, name, bases, dct):
        _cls = super().__new__(mcs, name, bases, dct)
        jax.tree_util.register_pytree_with_keys(
            _cls,
            flatten_func=_BaseModuleMeta.flatten_module,
            flatten_with_keys=_BaseModuleMeta.flatten_module_with_keys,
            unflatten_func=functools.partial(_BaseModuleMeta.unflatten_module, cls=_cls),
        )
        return _cls
```

### 作用
每当定义一个继承自 `BaseModule` 的类时，元类的 `__new__` 会被调用，自动把该类注册为 JAX pytree。这样模块对象就能被 `jax.jit`、`jax.grad` 等展平/还原。

### 展平规则（关键：把模块当作字典）
- **children（动态叶子）** = `module.__dict__.values()`（所有实例属性的值）
- **aux_data（静态数据）** = `module.__dict__.keys()`（所有实例属性的名字）

| 方法 | 作用 |
|------|------|
| `flatten_module` | 返回 `(values, keys)`，不带路径 |
| `flatten_module_with_keys` | 返回 `(((GetAttrKey(key), value), ...), keys)`，带属性路径，供 `tree_map_with_path` 使用 |
| `unflatten_module` | 用 `object.__new__(cls)` 绕过 `__init__` 创建裸对象，再根据 `keys` + `children` 重建 `__dict__` |

> `unflatten` 用 `object.__new__` 是为了不触发 `__init__`，从而能完全从展平数据重建对象。

---

## 2. `BaseModule` — 所有模块的基类

```python
class BaseModule(metaclass=_BaseModuleMeta):
    def __call__(self): ...
    def __repr__(self) -> str: ...
    def submodules(self, *, cls=None) -> Generator: ...
```

### `__call__`
子类应重写以定义前向逻辑；基类中直接 `raise NotImplementedError`。

### `__repr__`
递归遍历模块，打印所有 `DynamicParam` 叶子（及其 pytree 路径），方便查看模型结构。若没有动态参数，打印 `(empty)`。例如：

```
(Linear):
  .w: Param([3,4], float32)
  .b: Param([4], float32)
```

### `submodules(cls=None)` — 直接子模块生成器
- 返回**直接子节点**中类型匹配 `cls` 的模块（**非递归**，只看一层）。
- `cls=None` 时默认匹配所有 `BaseModule`。
- 实现：先用 `eqx.tree_flatten_one_level` 取第一层子节点，再从中筛出指定类型。

```python
for sub in module.submodules(Module):
    ...
```

---

## 3. `Module(BaseModule)` — 带 train/eval 模式的标准模块

```python
class Module(BaseModule):
    class MODE(IntEnum):
        NONE = 0
        TRAIN = 1
        EVAL = 2
```

### `MODE`
用 `IntEnum` 表示模块状态。`IntEnum` 使模式可比较、可哈希，且能作为静态值参与 JAX 变换。

### `__init__`
`self._mode = static(None)` —— 模式以 `StaticParam` 存储，避免被当作动态参数追踪（模式是结构信息，不是张量）。**子类构造时需调用 `super().__init__()`**。

### `mode(value)` — 递归设置/读取模式
- `value` 非 `None`：通过 `tree_apply` **递归**地把所有子 `Module` 的 `_mode` 设为 `value`，返回 `None`。
- `value` 为 `None`：返回当前模式（只读查询）。

### `train()` / `eval()` — 模式 + equinox 推理模式联动
两者除了把模式设为 `MODE.TRAIN`/`MODE.EVAL`，还会联动设置 equinox 模块的推理标志：
- `train()`：把所有含 `inference` 属性的对象（即 equinox 模块）设为 `inference=False`。
- `eval()`：设为 `inference=True`。

注意此处 `tree_apply` 的 `recursive=False`：因为 `mode()` 已经递归过一遍，equinox inference 只需对每个匹配节点直接应用一次。

### `is_train` / `is_eval` — 只读属性
```python
@property
def is_train(self) -> bool:
    return self._mode.get() == Module.MODE.TRAIN
```

---

## 使用示例

```python
import jax.numpy as jnp
import pcx

class Linear(pcx.Module):
    def __init__(self, din, dout):
        super().__init__()                       # 初始化 _mode
        self.w = pcx.Param(jnp.zeros((din, dout)))
        self.b = pcx.Param(jnp.zeros((dout,)))

    def __call__(self, x):
        return x @ self.w + self.b              # Param 重载了运算符，可直接参与运算

m = Linear(3, 4)
print(repr(m))        # 打印 w、b 的 shape/dtype
m.train()             # 递归设为 TRAIN 模式
m.is_train            # True
```

---

## 依赖关系

`_module.py` 依赖：
- [_parameter.py](pcx-core-parameter.md)（`DynamicParam`，用于 `__repr__` 定位叶子）
- [_static.py](pcx-core-static.md)（`static`，用于 `_mode`）
- [_tree.py](pcx-tree.md)（`tree_apply`，用于递归设置模式）
