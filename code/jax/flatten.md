## "展平"（flatten）的含义与触发时机

### 什么是展平

展平就是把一个**嵌套的、有结构的对象**拆成两个东西：

```
原始对象:                   展平后:
                        ┌─ children（叶子值）: (1.0, 2.0, "hello", 3.0)
MyModel(               │
  weight = (1.0, 2.0),─┤  ← 所有"终端值"被拉平到一个无嵌套元组里
  name = "hello",      │
  bias = 3.0           │
)                      └─ aux_data（结构信息）: 记录每片叶子原本的位置
                           "展平后怎么还原回去"的说明书
```

用 `dict` 类比最直观：

```python
import jax

# dict 是 JAX 内置支持的 pytree
d = {"a": 1, "b": (2, 3)}

leaves, tree_def = jax.tree.flatten(d)
print(leaves)    # [1, 2, 3]           ← 无嵌套，全是叶子
print(tree_def)  # PyTreeDef({'a': *, 'b': (*, *)})  ← 结构说明书
```

对于 `BaseModule`，`__dict__` 就是那个被展平的字典：

```python
class Obj(BaseModule):
    def __init__(self):
        self.weight = np.array([1.0, 2.0])   # 叶子（会被微分）
        self.bias = np.array([3.0])           # 叶子（会被微分）
        self.name = "my_layer"               # 叶子（不会被微分，因为是 str）

obj = Obj()

# flatten_module 做的事：
children = tuple(obj.__dict__.values())
# → (array([1.,2.]), array([3.]), "my_layer")

aux_data = tuple(obj.__dict__.keys())
# → ("weight", "bias", "name")
```

---

### 什么时候触发展平

**几乎所有 JAX 变换在遇到 pytree 类型的参数时都会自动展平。** 你不需要手动调用。

#### 场景 1：`jax.grad` — 计算梯度

```python
def loss_fn(model):                     # model 是一个 BaseModule
    return model.weight.sum() + model.bias.sum()

grads = jax.grad(loss_fn)(model)
# JAX 内部自动做:
#   1. flatten(model) → leaves=(weight, bias, name), struct
#   2. 对 leaves 中可微的部分求导 → (∂L/∂weight, ∂L/∂bias)
#   3. unflatten(struct, new_leaves) → 新 BaseModule，属性值是梯度
```

#### 场景 2：`jax.jit` — 即时编译

```python
@jax.jit
def train_step(model, x):
    ...  # 前向传播

train_step(model, batch)
# JAX 内部自动做:
#   1. flatten(model) → 把模型拆成纯数组
#   2. trace 这些数组（静态信息如 str 不会被 trace）
#   3. 编译成 XLA 计算图
#   4. 每次调用时 unflatten 出新的 model
```

#### 场景 3：`jax.vmap` — 向量化映射

```python
models = [Model(...) for _ in range(10)]   # 10 个模型实例

# 如果可以堆叠（同一个类），vmap 会批量处理
# 内部先 flatten 每个模型，再对 leaf arrays 做 vmap
```

---

### 为什么要展平

JAX 的函数式设计要求**所有变换只操作扁平数组**：

```
用户代码的自然表示:              JAX 内部需要的表示:

MyModel(                        [array([1.,2.]),     ← 只有这些能被微分
  weight: array([1.,2.]),        array([3.]),
  bias:   array([3.]),           "my_layer"]         ← 这个算静态信息
  name:   "my_layer"
)                                +
                                 {"weight", "bias", "name"}  ← 结构元信息
        │                              │
        └── flatten ──────────────────►├── JAX 变换（grad/jit/vmap）
                                        │
        ◄── unflatten ─────────────────┘
```

**核心逻辑**：JAX 的 `grad` 只能对数组求导，`jit` 只能 trace 数组——它们不知道怎么处理自定义 Python 对象。展平 + 还原这个机制让 JAX **只跟数组打交道**，但用户**只需要操作对象**。

---

### 展平 vs 深度学习框架的参数管理

| | PyTorch | JAX + this project   |
| ------- | ------------------------------ | ------------------------------ |
| 如何获取参数  | `model.parameters()` 显式迭代      | `flatten` 自动提取所有 jnp.array 属性  |
| 如何更新参数  | `opt.step()` 原地修改 `param.data` | `unflatten` 用新数组创建新模型实例        |
| 注册方式    | 继承 `nn.Module`                 | 继承 `BaseModule`（元类自动注册 pytree） |

一句话总结：**展平是 JAX 函数式编程范式下的"序列化"——把自定义对象拆成纯数组列表，让 JAX 变换能处理；变换完再装回去。**

## 具体的展平操作（由用户定义）
**是的，对。** 对于自定义 class，JAX 只有一个入口：通过 `register_pytree_with_keys`（或老 API `register_pytree_node`）显式注册。不存在其他方式。

---

### JAX 展平自定义类型的全部途径

```
register_pytree_node(cls, flatten_func, unflatten_func)       ← 老 API，无 key 信息
register_pytree_with_keys(cls, flatten_func,                   ← 新 API，项目用的
                          flatten_with_keys, unflatten_func)
register_pytree_node_class(cls)                                ← 装饰器语法糖，底层调上面
```

三者本质相同：**你必须显式告诉 JAX "这个类型怎么拆、怎么装"**。Python 没有 `__flatten__` 这样的魔术方法让 JAX 自动发现——自定义类的展平逻辑 100% 由注册时传入的那几个函数定义。

---

### 这个项目里的证据

```python
# _module.py —— BaseModule 的展平 = 把 __dict__ 当字典处理
register_pytree_with_keys(_cls,
    flatten_func=...         → 返回 (tuple(__dict__.values()), tuple(__dict__.keys()))
    unflatten_func=...       → object.__new__ + 写回 __dict__
)

# _parameter.py —— BaseParam 的展平 = 只把 _value 当叶子，其余属性静态
register_pytree_with_keys(_cls,
    flatten_func=...         → 返回 ((param._value,), rest_of_attrs)
    unflatten_func=...       → object.__new__ + 恢复完整 __dict__
)
```

同一个 API，传入不同的函数 → 完全不同的展平行为。没有隐式约定，一切由注册函数决定。

---

### 如果不注册会发生什么

```python
class NotRegistered:
    def __init__(self):
        self.x = jnp.array([1.0, 2.0])

obj = NotRegistered()
jax.tree.flatten(obj)   # ❌ TypeError: Cannot handle type <class 'NotRegistered'>
```

JAX 不认识这个类型，直接报错。而 `BaseModule` 的元类让每个子类**在诞生时就自动注册好了**，所以用户永远不需要手动调 `register_pytree_with_keys`