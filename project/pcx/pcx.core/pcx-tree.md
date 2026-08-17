# PCX Tree 函数详解

`pcx/core/_tree.py` 提供了一组工具函数，用于管理基于 `BaseParam` 的有状态 pytree/pydag。核心挑战在于：**JAX 要求所有输入/输出为 pytree（树），但深度学习模型常包含共享参数，形成 pydag（有向无环图）**——同一个参数对象被多处引用。这四个函数协同解决了 pydag ↔ pytree 的转换以及参数值的提取/注入。

---

## 1. `tree_ref` — 将 pydag 转换为 pytree

### 作用

将重复出现的 `BaseParam` 引用替换为显式整数索引（`_BaseParamRef`），把 pydag "拍平"为 pytree。

### 工作原理

- 首次遇到某个 `BaseParam` 实例 → 保留原样
- 再次遇到同一实例 → 替换为 `_BaseParamRef(n)`，其中 `n` 是该参数在所有唯一参数中的序号

```python
p = BaseParam(...)
dag = {"a": p, "b": p}        # p 被引用两次 → pydag

refed = tree_ref(dag)
# refed = {"a": p, "b": _BaseParamRef(0)}
# _BaseParamRef(0) → 第 0 号唯一参数，即 p
```

### 源码关键点

- 使用 `_cache()` 闭包跟踪已见过的对象（基于 `id()`），同时记录首次出现的顺序
- `is_leaf=lambda x: isinstance(x, BaseParam)` — 将 `BaseParam` 视为叶子节点，不深入其内部
- 对已是 `_BaseParamRef` 的节点会嵌套包装，支持多次 ref/unref

---

## 2. `tree_unref` — 将 pytree 还原为 pydag

### 作用

`tree_ref` 的逆操作——将 `_BaseParamRef` 还原为实际的对象引用，重建原始的 pydag。

### 典型用法

```python
def f(pytree):
    pydag = tree_unref(pytree)   # 进入 JAX 变换前恢复 pydag
    # ... 在 pydag 上做操作 ...
    return tree_ref(pydag)        # 返回前转回 pytree

p = pydag(...)
t = a_jax_transformation(f)
p = tree_unref(t(tree_ref(p)))
```

> 注意：当使用 pcax 自动参数追踪时（即在 pcax 变换的 `kwargs` 中传递参数），这个过程会自动完成。

### 重要限制

| 规则 | 说明 |
|------|------|
| 结构一致性 | ref 和 unref 之间的 pytree/pydag 结构必须保持不变 |
| 嵌套顺序 | 多次 ref → 必须逆序 unref（后 ref 先 unref） |
| 禁止跨结构 | 不能从一个大结构中随意取出子结构后单独 unref |

---

## 3. `tree_extract` — 从 pydag 中提取参数值

### 作用

从 pydag 的 `BaseParam` 叶子节点中按序遍历提取值，返回**有序序列**。

### 签名

```python
def tree_extract(
    pydag: PyTree,
    *rest,                    # 可附加多个同结构 pytree
    extract_fn: Callable = lambda x: x,   # 提取转换函数
    filter_fn: Callable = lambda x: isinstance(x, DynamicParam),  # 叶子过滤器
    is_pytree: bool = False,  # 是否已是 pytree（当前仅支持 True）
) -> Sequence[Any]:
```

### 基本用法

```python
# 提取模型中所有 DynamicParam 的值
model = MyModel(...)
values = tree_extract(model)
# values = (value_of_param_0, value_of_param_1, ..., value_of_param_n)
```

### 高级用法

```python
# 多 pytree 同步提取 + 自定义提取函数
grad_values = tree_extract(
    model, grads,
    extract_fn=lambda p, g: (p.get(), g),  # 同时提取参数值和对应梯度
)
```

---

## 4. `tree_inject` — 将值注入回 pydag

### 作用

`tree_extract` 的逆操作——将值序列按相同顺序注入回 pydag 的 `BaseParam` 叶子节点。

### 签名

```python
def tree_inject(
    pydag: PyTree,
    *,
    values: Sequence[Any] = None,   # 要注入的值序列
    params: PyTree = None,           # 或从参数 pytree 中提取值
    inject_fn: Callable = lambda n, v: n.set(v),  # 注入函数
    filter_fn: Callable = lambda x: isinstance(x, DynamicParam),
    is_pytree: bool = False,
    strict: bool = True,             # 严格模式：值数量必须与叶子数匹配
) -> PyTree:
```

### 基本用法

```python
# 典型配合：提取 → 修改 → 注入
values = list(tree_extract(model))   # 提取值
values = [v * 2 for v in values]     # 修改值（如梯度更新）
tree_inject(model, values=values)     # 注入回去
```

### `params` vs `values`

- `params`：直接传入参数 pytree，自动提取 `.get()` 后注入
- `values`：传入已准备好的值序列

两者互斥，不能同时指定。

### `strict` 模式

| `strict=True` | `strict=False` |
|---|---|
| 值数量必须与叶子数精确匹配 | 允许值少于叶子数（多余的叶子不被注入） |
| 用于捕获结构不一致的错误 | 用于部分更新场景 |

---

## 完整工作流示例

```python
# 1. 将 pydag 转为 pytree（使其可通过 JAX 变换）
pytree = tree_ref(model)

# 2. 提取所有可训练参数的值
old_values = tree_extract(pytree, is_pytree=True)

# 3. 应用 JAX 变换（如梯度下降）
@jax.jit
def update(pytree, grads):
    values = tree_extract(pytree, is_pytree=True)
    new_values = [v - 0.01 * g for v, g in zip(values, grads)]
    return tree_inject(pytree, values=new_values, is_pytree=True)

# 4. 更新模型
pytree = update(pytree, computed_grads)

# 5. 解除 ref，恢复原始 pydag 结构
model = tree_unref(pytree)
```

---

## 函数关系总览

```
         tree_ref          tree_extract
pydag ──────────► pytree ──────────────► values (有序序列)
  ▲                  │                        │
  │                  │                        │ tree_inject
  │                  ▼                        ▼
  └────────────  pytree ◄────────────────── values
         tree_unref
```
