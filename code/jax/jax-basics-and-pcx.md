---
sr-due: 2026-08-04
sr-interval: 1
sr-ease: 230
---
#review

官方文档：[Quickstart: How to think in JAX](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html)

## 1. JAX 是什么？

JAX 是一个高性能数值计算库，核心理念是**函数式编程 + 自动微分 + XLA 编译加速**。可以把它理解为"支持 GPU/TPU 加速、自带自动微分的 NumPy"。

```python
import jax
import jax.numpy as jnp  # API 与 NumPy 几乎一致

x = jnp.array([1.0, 2.0, 3.0])
y = jnp.sin(x)  # 用法完全同 numpy
```

---

## 2. JAX 三大核心机制

### 2.1 `jax.jit` — 即时编译

将 Python 函数编译为 XLA 优化的底层代码，大幅加速执行。

```python
@jax.jit
def f(x):
    return x @ x.T

x = jnp.ones((1000, 1000))
f(x)  # 第一次调用会编译（较慢），之后调用极快
```

**pcx 中的对应**：[`Jit`](../pcx/functional/_transform.py#L240) — 包装 `jax.jit`，自动处理参数追踪：

```python
# pcx 用法（参数通过 kwargs 传递，自动追踪更新）
@Jit(...)
def update(x, *, model):
    ...
```

### 2.2 `jax.grad` / `jax.value_and_grad` — 自动微分

自动计算函数关于任意参数的梯度。

```python
@jax.value_and_grad
def loss_fn(w, x):
    return (w @ x).sum()

w = jnp.array([1.0, 2.0, 3.0])
x = jnp.array([4.0, 5.0, 6.0])
loss, grads = loss_fn(w, x)  # 同时返回 loss 值和梯度
```

**pcx 中的对应**：[`ValueAndGrad`](../pcx/functional/_transform.py#L265) — 包装 `jax.value_and_grad`，增加 `kwargs_mask` 机制来控制哪些参数被微分：

```python
# pcx 中，mask 作用在 Param（叶子）级别，而非 jax.Array 级别
model = [Param(jnp.array([1.0])), Param(jnp.array([2.0]))]
mask  = [True, False]  # 仅对第一个参数求梯度

@ValueAndGrad(kwargs_mask=mask, has_aux=True)
def loss_fn(x, y, *, model):
    ...
```

**为什么需要在参数级别 mask？** 因为一个 `Param` 在 JAX 眼中是一个叶子节点——它的 `_value` 是动态的，其余属性是静态的。mask 替换整个 `Param` 对象，而非深入 `_value` 内部。这意味着**不同行为应放在不同 `Param` 中**（参见 [_parameter.py](../pcx/core/_parameter.py#L296) 的 `ParamDict`）。

### 2.3 `jax.vmap` — 自动向量化

自动沿 batch 维度扩展函数，无需手写循环。

```python
@jax.vmap
def f(x):
    return x * 2

x = jnp.arange(12).reshape(3, 4)
f(x)  # 自动沿第一维批量执行，等价于 jnp.array([f(x[i]) for i in range(3)])
```

**pcx 中的对应**：[`Vmap`](../pcx/functional/_transform.py#L345) — 包装 `jax.vmap`，自动处理 RKG 的分裂/合并：

```python
@Vmap(kwargs_mask=mask)  # 指定哪些 kwargs 参数参与向量化
def eval(x, *, model):
    ...
```

---

## 3. PyTree 系统 — JAX 的"数据结构协议"

PyTree 是 JAX 最核心的概念之一：**任何嵌套的 Python 容器（list, tuple, dict, 自定义类）只要能按规则"拍平"为叶子数组的列表，就是一个 pytree**。

### 3.1 内置 pytree 操作

```python
# tree_flatten: 将树展平为 (叶子列表, 结构信息)
leaves, treedef = jtu.tree_flatten({"a": 1, "b": [2, 3]})
# leaves  = (1, 2, 3)
# treedef = PyTreeDef({'a': *, 'b': [*, *]})

# tree_unflatten: 根据结构信息重建树
jtu.tree_unflatten(treedef, (10, 20, 30))
# {'a': 10, 'b': [20, 30]}

# tree_map: 对每个叶子应用函数，保留树结构
jtu.tree_map(lambda x: x * 2, {"a": 1, "b": [2, 3]})
# {'a': 2, 'b': [4, 6]}

# tree_leaves: 只取叶子列表
jtu.tree_leaves({"a": 1, "b": [2, 3]})
# [1, 2, 3]
```

### 3.2 `is_leaf` 参数

控制遍历深度——哪些节点应被视为"叶子"，不继续展开：

```python
# 默认行为：展开所有容器
jtu.tree_leaves([1, [2, 3]])   # [1, 2, 3]

# 指定叶子条件：将 list 视为叶子，不展开
jtu.tree_leaves([1, [2, 3]], is_leaf=lambda x: isinstance(x, list))
# [1, [2, 3]]
```

**这是 pcx 与 JAX 集成的关键接口**。pcx 中所有 `tree_map`/`tree_leaves` 调用都会传入：

```python
is_leaf=lambda x: isinstance(x, BaseParam)
```

这样 `Param` 对象被视为叶子节点，JAX 不会深入其内部属性。

### 3.3 自定义 pytree 注册

JAX 允许任何类通过 `register_pytree_with_keys` 注册为 pytree：

```python
jax.tree_util.register_pytree_with_keys(
    MyClass,
    flatten_func=...,          # 如何展平
    flatten_with_keys=...,     # 带路径信息的展平
    unflatten_func=...,        # 如何重建
)
```

pcx 的两个核心注册：

| 类 | 注册位置 | 展平方式 |
|---|---|---|
| `BaseParam` | [_parameter.py:33](../pcx/core/_parameter.py#L33) | `_value` 为动态叶子，其余属性为静态辅助数据 |
| `BaseModule` | [_module.py:57](../pcx/core/_module.py#L57) | `__dict__` 的所有值按 key 展平 |

---

## 4. JAX 的函数式约束与 pcx 的解决之道

### 4.1 问题：JAX 要求纯函数

JAX 变换要求函数**纯**——不能有副作用。但深度学习模块状态随时变化（如 BatchNorm 的 running mean）：

```python
# 这在 JAX 中不行——batch_norm 内部有状态更新
@jax.jit
def forward(x, model):
    return model(x)  # model 的状态变了，但 JAX 不知道！
```

### 4.2 pcx 的方案：参数追踪

pcx 的核心思路——**用 `Param` 包装 `jax.Array`，让 pcx 变换自动追踪并写回**：

```
┌─────────────────────────────────────────────────────┐
│            pcx 变换工作流                              │
│                                                       │
│  1. 提取：tree_extract(model) → 取出所有 Param 的值    │
│  2. 转换：交给 JAX 变换处理（jit/grad/vmap）           │
│  3. 注入：tree_inject(model, values) → 写回更新后的值   │
└─────────────────────────────────────────────────────┘
```

具体实现在 [_BaseTransform](../pcx/functional/_transform.py#L92)：

```python
def __call__(self, *args, _is_root=True, **kwargs):
    if _is_root:
        kwargs = tree_ref(kwargs)        # 第 1 步：pydag → pytree
    _r, _kwargs = self._t(*args, **kwargs)  # 第 2 步：执行 JAX 变换
    tree_inject(kwargs, params=_kwargs, is_pytree=True)  # 第 3 步：写回
    return _r
```

### 4.3 完整对比

| | 原始 JAX | pcax |
|---|---|---|
| 状态管理 | 手动传入/传出状态 | `Param` 包装，变换自动追踪 |
| 参数传递 | 全部作为位置参数 | 位置参数→"纯"数据；kwargs→含 Param 的模型 |
| 参数共享 | 手动处理（展平后再索引） | `tree_ref`/`tree_unref` 自动处理 pydag |
| 函数签名 | `fn(params, x)` | `fn(x, *, model)` |
| 使用梯度 | `opt.update(params, grads)` | `tree_inject(model, values=new_values)` |

### 4.4 实际用法示例

```python
import pcx
from pcx.functional import Jit, ValueAndGrad

# 定义模型（参数使用 pcx.Param）
class LinearModel(pcx.Module):
    def __init__(self):
        self.w = pcx.Param(jnp.zeros((10, 1)))
        self.b = pcx.Param(jnp.zeros((1,)))

    def __call__(self, x):
        return x @ self.w.get() + self.b.get()

model = LinearModel()

# x 是"纯"数据，model 放在 kwargs 中自动追踪
@ValueAndGrad(has_aux=True)
def loss_fn(x, y, *, model):
    y_hat = model(x)
    return ((y_hat - y) ** 2).mean()

# 调用——model 的参数自动更新
loss, grads = loss_fn(x_batch, y_batch, model=model)
```

> **核心心智模型**：位置参数 → JAX 原生世界（纯数据），关键字参数 → pcx 追踪世界（有状态参数）。

---

## 5. `jax.lax` 控制流

JAX 不允许对动态值使用 Python 原生控制流（`if`/`for`/`while`），因为它们在 trace 时就被固化了：

```python
@jax.jit
def f(x):
    if x > 0:  # ❌ 错误！x 是 tracer，无法判断
        return x
    return -x
```

必须使用 `jax.lax` 中的函数式替代：

| Python | jax.lax 替代 | pcx 封装 |
|---|---|---|
| `for` 循环（累积状态） | `jax.lax.scan` | [`Scan`](../pcx/functional/_flow.py#L41) |
| `while` 循环 | `jax.lax.while_loop` | [`WhileLoop`](../pcx/functional/_flow.py#L79) |
| `if-else` | `jax.lax.cond` | [`Cond`](../pcx/functional/_flow.py#L120) |
| `switch-case` | `jax.lax.switch` | [`Switch`](../pcx/functional/_flow.py#L158) |

pcx 的封装优势：**自动追踪 kwargs 中的参数**，避免手动拼装 `carry` 元组：

```python
# 原始 jax.lax.scan
def step(carry, x):
    params, state = carry
    # ... 计算 ...
    return (new_params, new_state), output

# pcx Scan——无需 compact carry，参数自动追踪
def step(x, state, *, model):
    # model 中的参数自动更新
    return (new_state,), output

Scan(step, xs=data)(initial_state)
```

---

## 6. pcx 中的 JAX 技术栈总览

```
                    ┌──────────┐
                    │  用户代码  │
                    └────┬─────┘
                         │
              ┌──────────▼──────────┐
              │  pcx 变换层           │
              │  Jit / Vmap /        │
              │  ValueAndGrad /      │
              │  Scan / Cond ...     │
              │                      │
              │  职责：参数追踪        │
              │  + ref/unref         │
              │  + extract/inject    │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  pcx 核心层           │
              │  Param / Module /    │
              │  StaticParam /       │
              │  _tree / _random     │
              │                      │
              │  职责：pytree 注册     │
              │  + pydag 支持         │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  JAX 原生层           │
              │  jax.jit / jax.grad  │
              │  jax.vmap / jax.lax  │
              │  jtu.tree_*          │
              │                      │
              │  职责：编译 + 微分     │
              │  + 向量化 + pytree    │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Equinox             │
              │  eqx.nn / eqx.Module │
              │  eqx.tree_*          │
              │                      │
              │  职责：NN 层实现      │
              │  + 增强的 pytree 工具  │
              └──────────────────────┘
```

---

## 7. 两个关键辅助机制

### 7.1 `StaticParam` — 携带非数组数据

JAX pytree 要求叶子节点是 `jax.Array` 兼容的类型，但实际模块常需要携带函数、字符串、配置字典等。`StaticParam`（[../pcx/core/_static.py](../pcx/core/_static.py)）利用 pytree 的"静态辅助数据"机制解决了这个矛盾：

```python
class StaticParam(BaseParam):
    def __init__(self, value=None):
        self._static_value = value  # 实际值存入静态辅助数据
        super().__init__(None)       # _value = None → JAX 视为无动态内容
```

在注册的 `flatten_parameter` 中，`_value` 之外的属性都放入静态辅助数据，而 `StaticParam` 的所有实际内容都在 `_static_value` 中，因而 JAX 变换完全不会触碰它。

**使用场景**：
- `_BaseParamRef(n)` — 存储整数索引，继承自 `StaticParam`
- `Module._mode` — 存储 train/eval 模式标识
- `Layer` 中所有非数组的 equinox 模块属性

### 7.2 `RKG` — 全局随机数管理

JAX 的随机数生成是**显式**的——必须手动传递和分裂 key。pcx 提供了全局 `RKG`（Random Key Generator）来自动化这个流程（[../pcx/core/_random.py](../pcx/core/_random.py)）：

```python
# 自动注入到每个 pcx 变换的 kwargs 中
kwargs["__RKG"] = RKG

# Vmap 中自动沿 batch 维度分裂
kwargs["__RKG"].key.set(kwargs["__RKG"].key.split(_vaxis_dim))

# 变换结束后合并回来
kwargs["__RKG"].key.set(kwargs["__RKG"].key[0])
```

用户代码中使用 `rkg()` 即可获取 split 后的子 key，无需手动传递。

---

## 8. Mask 系统 — 精细控制变换行为

`ValueAndGrad` 和 `Vmap` 需要知道**哪些参数参与变换**。pcx 提供了可组合的 Mask API（[../pcx/utils/_mask.py](../pcx/utils/_mask.py)）：

```python
from pcx.utils import M, M_is, M_has, M_hasnot

# M: 等同于 True/False 字面量
mask = [True, False]     # 第 0 个参数求梯度，第 1 个不求

# M_is: 类型匹配
mask = M_is(LayerParam)  # 仅 LayerParam 参与

# M_has: 属性匹配
mask = M_has("trainable")  # 仅含有 trainable 属性的参数

# 逻辑组合
mask = M_is(LayerParam) & M_has("trainable")  # 且
mask = M_hasnot("frozen")                     # 非
```

**关键设计决策**：Mask 作用于 **`Param` 叶子级别**而非 `jax.Array` 级别（参见 `ValueAndGrad` 的 NOTES）。这意味着：
- `[True, False]` 是对整个 `Param` 对象的替换，不会深入到其内部 `_value`
- 同一 `Param` 内无法区分不同 `jax.Array` 的行为 → 不同行为应放在不同 `Param` 中

---

## 9. `tree_apply` — 在 pytree 上执行副作用

除了四大核心函数外，`tree_apply`（[../pcx/core/_tree.py:85](../pcx/core/_tree.py#L85)）也是一个重要工具——对 pytree 中匹配条件的节点执行有副作用的函数：

```python
# 递归设置所有 Module 子模块的 train 模式
tree_apply(
    lambda m: m._mode.set(Module.MODE.TRAIN),
    lambda x: isinstance(x, Module),
    self,
)
```

**注意**：如果 pytree 中有重复引用（pydag），该函数会在同一节点上执行多次，设计时需考虑此行为。

---

## 10. 关键设计模式总结

| 模式 | 机制 | 解决的问题 |
|---|---|---|
| **元类自动注册** | `_BaseParamMeta` / `_BaseModuleMeta` 调用 `register_pytree_with_keys` | 所有 Param/Module 子类自动成为 pytree 节点 |
| **ref/unref 三明治** | 进入变换前 `tree_ref(kwargs)`，调用用户函数时 `tree_unref` | pydag ↔ pytree 透明转换 |
| **extract/inject 追踪** | 变换中 `tree_extract` 取值，变换后 `tree_inject` 写回 | 有状态参数在纯函数变换中保持状态 |
| **值包装器** | `Param` 代理所有算术运算到 `jax.Array` | 让 Param 对象像原生数组一样使用 |
| **静态哨兵** | `StaticParam._value = None`，真实值在静态辅助数据 | 非数组数据安全穿越 JAX 变换 |
| **RKG 自动注入** | 每个变换自动注入/管理 `__RKG` | 用户无需手动传递随机数 key |
| **Mask 组合** | `M_is` / `M_has` / 逻辑操作符 | 声明式地控制哪些参数参与变换 |

---

## 11. 关键源文件索引

| 文件 | 内容 |
|---|---|
| [../pcx/core/_parameter.py](../pcx/core/_parameter.py) | `Param`, `ParamDict` — JAX Array 的状态包装器 |
| [../pcx/core/_module.py](../pcx/core/_module.py) | `Module` — 基于 pytree 的模块基类 |
| [../pcx/core/_tree.py](../pcx/core/_tree.py) | `tree_ref/unref`, `tree_extract/inject` — pydag ↔ pytree 转换 |
| [../pcx/functional/_transform.py](../pcx/functional/_transform.py) | `Jit`, `ValueAndGrad`, `Vmap` — JAX 变换的 pcx 包装 |
| [../pcx/functional/_flow.py](../pcx/functional/_flow.py) | `Scan`, `WhileLoop`, `Cond`, `Switch` — `jax.lax` 控制流的 pcx 包装 |
| [../pcx/nn/_layer.py](../pcx/nn/_layer.py) | `Layer`, `Linear`, `Conv`, `Dropout`, `LayerNorm` 等 — equinox 层的 pcx 适配 |
| [../pcx/nn/_stateful.py](../pcx/nn/_stateful.py) | `StatefulLayer`, `BatchNorm` — equinox 有状态层的适配 |
| [../pcx/nn/_shared.py](../pcx/nn/_shared.py) | `Shared` — 参数共享机制（利用 pydag 引用） |
| [../pcx/core/_static.py](../pcx/core/_static.py) | `StaticParam` — 非数组数据的安全 pytree 节点 |
| [../pcx/core/_random.py](../pcx/core/_random.py) | `RKG` — 全局随机数密钥管理器 |
| [../pcx/utils/_mask.py](../pcx/utils/_mask.py) | `M`, `M_is`, `M_has`, `M_hasnot` — 可组合的变换 mask |
| [../pcx/utils/_optim.py](../pcx/utils/_optim.py) | `Optimizer` — 基于 `eqx.apply_updates` 的梯度更新 |
| [../pcx/utils/_serialisation.py](../pcx/utils/_serialisation.py) | 模型序列化（使用 `jtu.tree_flatten_with_path`） |
| [../pcx/predictive_coding/_energy.py](../pcx/predictive_coding/_energy.py) | 预测编码能量函数（`jax.numpy` / `jax.nn`） |
