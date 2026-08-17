# Vode 类的继承体系与依赖对象详解

> 深入分析 `Vode` 类的完整继承链和所有引用的外部对象，聚焦具体代码实现方法与设计目的。

## 2. 继承链第一层：`_BaseModuleMeta` 元类

**文件**：[pcx/core/_module.py:46-94](pcx/core/_module.py#L46-L94)

### 2.1 设计目的

在 JAX 中，任何经过 `jit`、`vmap`、`grad` 等变换的对象都必须是 **pytree**（可被展平和还原的树结构）。`_BaseModuleMeta` 让**所有继承 `BaseModule` 的类自动注册为 JAX pytree**，无需任何手动操作。

### 2.2 实现细节

```python
class _BaseModuleMeta(abc.ABCMeta):
    def __new__(mcs, name, bases, dct):
        _cls = super().__new__(mcs, name, bases, dct)

        # 核心：将新创建的类注册到 JAX 的 pytree 系统
        jax.tree_util.register_pytree_with_keys(
            _cls,
            flatten_func=_BaseModuleMeta.flatten_module,
            flatten_with_keys=_BaseModuleMeta.flatten_module_with_keys,
            unflatten_func=functools.partial(
                _BaseModuleMeta.unflatten_module, cls=_cls
            ),
        )
        return _cls
```

**关键机制**：

1. **`__new__` 在类定义时触发**：当 Python 解释器读到 `class Vode(EnergyModule):` 时，会沿着继承链找到 `_BaseModuleMeta`，调用其 `__new__`。此时 `Vode` 这个类还正在创建中，`__new__` 拦截了这个过程。

2. **`register_pytree_with_keys` 注册三个函数**：
   - `flatten_func`：将 Module 实例展平为 `(值元组, 键元组)`，其中值来自 `module.__dict__.values()`，键来自 `module.__dict__.keys()`
   - `flatten_with_keys`：同上但用 `GetAttrKey` 包装键（保留属性路径信息）
   - `unflatten_func`：从展平的数据重建 Module 实例

3. **`functools.partial` 固定 `cls` 参数**：`unflatten_module` 的签名是 `(aux_data, children, *, cls)`，但 JAX 的 unflatten 协议只传 `(aux_data, children)`。`partial` 解决了这个签名不匹配问题——提前把 `cls=_cls` 绑定进去。

### 2.3 展平与重建的具体实现

```python
@staticmethod
def flatten_module(module: "BaseModule") -> Tuple[Tuple[Any, ...], Tuple[str, ...]]:
    # 展平：值 = __dict__ 的所有 value，辅助数据 = __dict__ 的所有 key
    return tuple(module.__dict__.values()), tuple(module.__dict__.keys())

@staticmethod
def unflatten_module(
    aux_data: Tuple[str, ...], children: Tuple[Any, ...], cls: Type["BaseModule"]
) -> "BaseModule":
    _module = object.__new__(cls)        # 绕过 __init__，只分配内存
    _module.__dict__ = dict(zip(aux_data, children))  # 用键值对重建 __dict__
    return _module
```

**设计要点**：
- `object.__new__(cls)` 只分配内存，不调用 `__init__`——这避免了在重建时重复初始化
- `__dict__` 的直接赋值意味着 Module 本质上被当作"带名字的字典"来序列化
- 属性名被保留为辅助数据，这使得重建后的对象结构完全一致

### 2.4 对 Vode 的影响

因为 `_BaseModuleMeta` 的存在，任何 `Vode` 实例都可以：
```python
# 被 JAX 变换接受
jax.jit(lambda v: v.energy())(vode_instance)

# 被展平查看内部状态
leaves, structure = jax.tree_util.tree_flatten(vode_instance)
# leaves: (h的值, cache的值, energy_fn的值, ruleset, shape的值, _status的值, _mode的值)
```

---

## 3. 继承链第二层：`BaseModule`

**文件**：[pcx/core/_module.py:97-138](pcx/core/_module.py#L97-L138)

### `submodules()` —— 遍历直接子模块

```python
def submodules(self, *, cls: Type[T] | None = None) -> Generator[T, None, None]:
    cls = cls or BaseModule
    _leaves, _ = eqx.tree_flatten_one_level(self)

    yield from filter(
        lambda x: isinstance(x, cls),
        jtu.tree_leaves(_leaves, is_leaf=lambda x: isinstance(x, cls)),
    )
```

**实现分析**：

1. **只遍历一层**：`eqx.tree_flatten_one_level(self)` 只展平第一层，不会递归进入子模块。这意味着 `submodules()` 返回的是**直接子级**，不是所有后代。

2. **`yield from` + `filter`**：使用生成器惰性求值，避免一次性构建列表。当网络很大时节省内存。

3. **`is_leaf` 参数**：`jtu.tree_leaves` 的 `is_leaf=lambda x: isinstance(x, cls)` 让匹配类型的模块被视为叶子节点，不再进一步展开。这样即使一个 Module 内部还嵌套了同类型模块，也只返回直接匹配的那一层。

4. **`*,` 强制关键字参数**：`cls` 必须用关键字传递（`submodules(cls=Vode)`），避免位置混淆。参考：[[python基础/2.asterisk-usage# 3. `*,` —— 强制关键字参数]]

**在 Vode 中的使用**：这个能力被 `EnergyModule.energy()` 继承和使用——它通过 `submodules(cls=EnergyModule)` 遍历子 Vode 并递归求和能量。Vode 自身不直接调用 `submodules`。

---

## 4. 继承链第三层：`Module`

**文件**：[pcx/core/_module.py:144-213](pcx/core/_module.py#L144-L213)

### 4.1 设计目的

`Module` 在 `BaseModule` 的基础上增加了深度学习模块的**训练/评估模式管理**。

### 4.2 `MODE` 枚举

```python
class MODE(IntEnum):
    NONE = 0
    TRAIN = 1
    EVAL = 2
```

`IntEnum` 既可以当整数比较（`== 1`），也有语义化的名称（`MODE.TRAIN`）。`NONE` 表示未设置状态。

### 4.3 `mode()` —— 递归设置模式

```python
def mode(self, value: MODE | None) -> MODE | None:
    if value is None:
        return self._mode.get()        # 读取模式 → 查询模式
    else:
        tree_apply(
            lambda m: m._mode.set(value),
            lambda x: isinstance(x, Module),
            self
        )
        return                          # 设置模式 → 递归应用到所有子 Module
```

**设计细节**：
- 同一个方法既是 getter 也是 setter：`mode(None)` 返回当前模式，`mode(MODE.TRAIN)` 设置模式
- `tree_apply` 递归遍历整个 pytree，找到每一个 `Module` 实例并设置其 `_mode`
- `_mode` 是 `StaticParam` 包装的（值为 `None` 初始状态），所以需要 `.set()` / `.get()` 访问

### 4.4 `train()` 和 `eval()` —— 便捷方法

```python
def train(self) -> None:
    self.mode(Module.MODE.TRAIN)
    # 同时设置 equinox 模块的 inference 标志
    tree_apply(
        lambda eqx_m: eqx_m.inference.set(False),
        lambda x: hasattr(x, "inference"),
        self, False,
    )

def eval(self) -> None:
    self.mode(Module.MODE.EVAL)
    tree_apply(
        lambda eqx_m: eqx_m.inference.set(True),
        lambda x: hasattr(x, "inference"),
        self, False,
    )
```

**设计目的**：除了设置 PCX 自己的模式标志，还需要同步设置 equinox 层的 `inference` 属性（控制 BatchNorm、Dropout 等行为）。

---

## 5. 继承链第四层：`EnergyModule`

**文件**：[pcx/predictive_coding/_energy_module.py:28-71](pcx/predictive_coding/_energy_module.py#L28-L71)

### 5.1 设计目的

`EnergyModule` 在 `Module` 的基础上增加了预测编码特有的两个概念：**状态（status）** 和 **能量（energy）**。

### 5.2 `_status` —— 用 `StaticParam` 包装的状态字符串

```python
class EnergyModule(Module):
    def __init__(self) -> None:
        super().__init__()
        self._status = static(None)   # None 是初始状态，表示"无状态"
```

**为什么用 `static()` 包装？** Module 是 JAX pytree，不能直接包含字符串这种非 JAX 类型。`static()` 将其包装为 `StaticParam`，JAX 展平时会将其放在辅助数据（aux_data）中而不是动态值中。

### 5.3 `status` —— `@property` 封装的访问接口

```python
@property
def status(self) -> Any:
    return self._status.get()

@status.setter
def status(self, status: Any):
    self._status.set(status)
```

**设计目的**：封装 `StaticParam` 的 `.get()` / `.set()` 调用，让外部代码可以自然地写 `vode.status = "init"` 和 `print(vode.status)`。

### 5.4 `energy()` —— 递归能量求和

```python
def energy(self) -> jax.Array:
    return functools.reduce(
        lambda x, y: x + y,
        (m.energy() for m in self.submodules(cls=EnergyModule))
    )
```

**实现分析**：

1. **递归遍历**：`self.submodules(cls=EnergyModule)` 找出所有直接子 `EnergyModule`（即子 Vode 或子预测编码网络）
2. **`functools.reduce` + 生成器**：惰性求和，避免构建中间列表
3. **每个子模块调用自己的 `energy()`**：对于 Vode，调用的是 Vode 重写的 `energy()`（计算单个 Vode 的能量）；对于普通 EnergyModule（即预测编码网络），调用这个递归版本（求和所有子模块的能量）
4. **最终效果**：在顶层网络上调用 `energy()` 会递归求和所有 Vode 节点的能量，得到整个预测编码网络的总能量

### 5.5 `clear_params()` —— 清除参数

**设计目的**：在预测编码推理中，每次迭代前需要清除上一次的缓存（如 Vode 的 `cache`）。`filter` 可以是类型（如 `VodeParam.Cache`）或自定义函数。`tree_apply` 遍历 pytree 找到匹配的参数并设置为 `None`。

---

## 6. 继承链第五层：`Vode`

**文件**：[pcx/predictive_coding/_vode.py:160-354](pcx/predictive_coding/_vode.py#L160-L354)

### 6.1 设计目的
每个 Vode 有自己的状态 `h`（节点的**隐变量**），接收传入的激活 `u`（上层的预测）。
通过能量函数计算 `h` 与 `u` 之间的误差，并通过预测编码的推理过程最小化这个误差。

### 6.2 `__init__` —— 构造函数的参数设计

```python
def __init__(
    self,
    energy_fn: Callable[["Vode", RandomKeyGenerator], jax.Array] = se_energy,
    ruleset: dict = {},
    tforms: dict = {},
    param_type: type[VodeParam] = VodeParam,
    *param_args,
    **param_kwargs,
):
```

| 参数 | 类型 | 默认值 | 目的 |
|------|------|--------|------|
| `energy_fn` | `Callable[[Vode, RKG], Array]` | `se_energy` | 计算节点能量的函数 |
| `ruleset` | `dict` | `{}` | 自定义规则（会被合并到默认规则） |
| `tforms` | `dict` | `{}` | 自定义变换函数 |
| `param_type` | `type[VodeParam]` | `VodeParam` | `h` 的参数类型（可被子类替换） |
| `*param_args` | — | — | 透传给 `param_type` 构造函数的位置参数 |
| `**param_kwargs` | — | — | 透传给 `param_type` 构造函数的关键字参数 |

**`*param_args, **param_kwargs` 透传模式**：Vode 不需要知道 `VodeParam.__init__` 接受什么参数。如果将来 `VodeParam` 的签名改变（例如增加 `shape` 参数），Vode 的代码无需修改。这体现了"依赖倒置"原则——Vode 依赖抽象（`param_type` 类型），不依赖具体（`VodeParam` 的构造细节）。

### 6.4 `__call__` —— 标准调用接口

```python
def __call__(
    self, u: jax.Array | None, rkg=..., output="h", **kwargs
) -> jax.Array | Any:
    if u is not None:
        self.set("u", u, rkg)         # 通过规则集设置输入

    for _k, _v in kwargs.items():
        self.set(_k, _v, rkg)         # 通过规则集设置额外的键值

    if (h := self.get("u")) is not None:
        self.shape.set(h.shape)       # 记录输入形状（区分 vmap 上下文）

    if output is None:
        return self                    # 返回自身（支持链式调用）
    else:
        return self.get(output, rkg=rkg)  # 通过规则集获取输出
```

**设计关键**：`set` 和 `get` 都经过 `Ruleset` 处理，而不是直接读写属性。这意味着规则可以拦截、变换、重路由任何数据的存取。

### 6.5 `set()` —— 经规则引擎处理的值设置

```python
def set(self, key: str, value, rkg=RKG) -> "Vode":
    _rule_pattern = f"(.*(?<!\\s))\\s*<-\\s*({key}.*)"

    rules = tuple(self.ruleset.filter(self.status, _rule_pattern))
    for _targets, _tform in rules:
        _value = self.ruleset.apply_set_transformation(
            self, _tform, _tform.split(":", 1)[0], value, rkg
        )
        for _target in _targets.split(","):
            _target = _target.strip()
            if hasattr(self, _target) and isinstance(
                (_param := getattr(self, _target)), Param
            ):
                _param.set(_value)           # 目标是注册的参数 → 直接设置
            else:
                self.cache[_target] = _value  # 目标不是参数 → 存入缓存

    if len(rules) == 0:
        # 没有匹配的规则 → 默认行为：如果 key 是属性中的 Param 就设置，否则存缓存
        if hasattr(self, key) and isinstance((_param := getattr(self, key)), Param):
            _param.set(value)
        else:
            self.cache[key] = value

    return self  # 返回自身，支持链式调用：vode.set("u", x).set("prior", y)
```

**规则匹配流程**：
1. 用正则 `(target)<-(key.*)` 匹配 `self.status` 对应的所有规则（如 `"h, u <- u:se"` → target=`"h, u"`, tform=`"u:se"`）
2. 如果有多条匹配规则，**全部依次执行**（输入规则允许多条）
3. 对每个目标（逗号分隔），检查是否是 `Param` 属性 → 是则 `.set()`，否则存入 `cache`
4. 如果没有规则匹配，执行默认行为

对于默认的`Vode`（无特殊ruleset）， `set("u", value)` 的行为 取决于 `self.status` 的值：

|`self.status`|`set("u", value)` 的效果|
|---|---|
|`"init"` (`STATUS.INIT`)|`self.h = value` **且** `self.cache["u"] = value`|
|其他（如 `None`）|仅 `self.cache["u"] = value`|

### 6.6 `get()` —— 经规则引擎处理的值获取

```python
def get(self, key: str, default=None, rkg=RKG) -> jax.Array | Any | None:
    _rule_pattern = f"({key})\\s*->\\s*(.*)"

    _rules = tuple(self.ruleset.filter(self.status, _rule_pattern))

    if len(_rules) == 0:
        # 无规则 → 默认行为
        if hasattr(self, key) and isinstance((_param := getattr(self, key)), Param):
            return _param.get()
        else:
            return self.cache.get(key, default)
    else:
        if len(_rules) > 1:
            print(f"WARNING: Multiple output rules matched for key '{key}'...")
        (_target, _tform) = _rules[0]    # ← 输出规则只取第一条！

        _value = self.ruleset.apply_get_transformation(self, _tform, _target, rkg=rkg)

        if ":" in _tform:
            self.cache[_tform] = _value    # 缓存变换结果，避免重复计算

        return _value
```

**输入规则与输出规则的关键区别**：
- 输入规则（`target <- key`）：**多条全部执行**
- 输出规则（`key -> target`）：**只执行第一条**（多余的打印警告）

### 6.7 `energy()` —— Vode 的能量计算

```python
def energy(self, rkg=RKG) -> jax.Array:
    if "E" not in self.cache:
        if self.energy_fn.get() is not None:
            _E = self.energy_fn(self, rkg=rkg)

            if self.h.shape == self.shape:
                # h.shape == self.shape → 在 vmapped 上下文中
                # h 没有被加批处理维度，说明 vmap 在外部
                _E = _E.sum()             # 求和为标量
            else:
                # h.shape != self.shape → 在普通上下文中
                # h 有批处理维度，保留每个样本的能量
                _E = jax.numpy.reshape(_E, (self.h.shape[0], -1)).sum(axis=1)

            self.cache["E"] = _E
        else:
            return 0.0

    return self.cache["E"]
```

**vmap 检测机制**：`self.shape` 记录的是 `__call__` 时 `u` 的 shape（不带 batch 维度）。如果 `energy()` 中 `h.shape == self.shape`，说明外部有 `vmap` 在运行（h 没有被加 batch 维），此时对 batch 内的所有元素求和；否则 h 有一个 batch 维度，需要保留每个样本的能量。

---

## 7. 核心依赖（一）：参数系统

Vode 依赖的参数类型构成了一个层级体系。PCX 的参数系统是为了解决 JAX 函数式编程的限制而设计的。

### 7.1 参数系统的设计理念

JAX 是函数式的——变换（`jit`, `vmap`, `grad`）要求输入/输出是纯 pytree。PyTree 只能包含数组和基本类型。但深度学习需要**可变状态**（参数更新、缓存、随机状态）。PCX 的解决方案是：

> 用一个对象包装值，值部分是动态的（随 JAX 变换变化），其他属性是静态的（保持不变）。

### 7.2 参数层级结构

```
_BaseParamMeta(abc.ABCMeta)         ← 元类：自动 JAX pytree 注册
    └── BaseParam                   ← 抽象基类：定义 get()/set() 接口
            ├── DynamicParam        ← 标记类：表示"这是动态参数"
            │       ├── Param       ← Vode 属性的实际类型
            │       └── ParamDict   ← 字典式参数（VodeParam.Cache 的父类之一）
            └── StaticParam         ← 静态参数：存储非 JAX 值（函数、字符串等）
                    └── _BaseParamRef  ← PyDAG 转 PyTree 的引用标记

ParamCache                           ← 标记类：标识缓存参数（VodeParam.Cache 的父类之一）
```

### 7.3 `BaseParam` —— 抽象基类

```python
class BaseParam(metaclass=_BaseParamMeta):
    def __init__(self, value=None):
        self._value = value        # 实际存储的值

    @abc.abstractmethod
    def get(self):                 # 子类必须实现：如何读取值
        raise NotImplementedError()

    @abc.abstractmethod
    def set(self, value):          # 子类必须实现：如何写入值
        raise NotImplementedError()

    def __bool__(self):
        raise TypeError("...")     # 禁止 bool 转换，防止意外错误
```

**`__bool__` 的设计**：Python 中 `if obj:` 会调用 `__bool__`。对 Param 来说这容易出错——用户可能想检查"值是否为 None"却意外检查了"Param 对象是否为 None"（后者总是 True）。直接抛异常强制用户写 `if param.get() is None:` 而不是 `if param:`。

### 7.4 `Param` —— 动态参数（Vode 属性所用的类型）

```python
class Param(DynamicParam):
    def get(self) -> jax.Array:
        return self._value

    def set(self, value) -> "Param":
        self._value = value
        return self
```

`Param` 重载了所有算术运算符（`+`, `-`, `*`, `/`, `@`, `**` 等）、比较运算符（`==`, `<`, `>` 等）和位运算符（`&`, `|`, `^` 等），使得**Param 实例可以直接参与数学运算**：

```python
# 在 se_energy 中：
e = vode.get("h") - vode.get("u")   # .get() 返回裸数组，直接减法
return 0.5 * (e * e)                 # 直接参与算术
```

`Param` 还通过 `__getattr__` 代理 `.shape`, `.dtype`, `.ndim` 到内部的 `_value`：

```python
def __getattr__(self, __name):
    return getattr(self._value, __name)  # param.shape → param._value.shape
```

**对 Vode 的意义**：`self.h` 是一个 `VodeParam`（继承自 `Param`），所以：
- `self.h.set(new_value)` 设置新值
- `self.h.get()` 获取裸 JAX 数组
- `self.h.shape` 直接访问数组的形状（用于 vmap 检测）

### 7.5 `VodeParam` —— Vode 专用的参数类型

```python
class VodeParam(Param):
    class Cache(ParamDict, ParamCache):    # 嵌套类！双重继承
        def __init__(self, params: Dict[str, jax.Array] = None):
            super().__init__(params)

    def __init__(self, value=None):
        super().__init__(value)
```

**设计要点**：

1. **`VodeParam` 继承 `Param`**：获得所有运算符重载和 `__getattr__` 代理。没有添加任何新方法，其作用纯粹是**类型标识**——`isinstance(x, VodeParam)` 可以区分 "Vode 的状态值" 和 "其他 Param"。

2. **`Cache` 嵌套类双重继承 `ParamDict + ParamCache`**：
   - `ParamDict`：提供字典式接口（`cache["key"] = value`, `cache.get("key", default)`, `"key" in cache`）
   - `ParamCache`：一个空标记类（sentinel），用于 `clear_params` 时识别缓存参数。`ParamCache` 的代码就是 `class ParamCache: pass`
   - 双重继承让 `Cache` 既是功能完整的字典参数，又能被筛选器识别为"缓存类型"

3. **Cache 的使用场景**：`self.cache["u"]` 存储传入激活，`self.cache["E"]` 存储能量值，`self.cache["u:se"]` 存储变换后的中间结果。

4. **`VodeParam.Cache()` 无需参数**：因为运行时是通过 `self.cache["key"] = value` 动态添加的，初始为空字典由 `ParamDict.__setitem__` 自动创建。

---

## 8. 核心依赖（二）：`StaticParam` 与 `static()`

**文件**：[pcx/core/_static.py](pcx/core/_static.py)

### 8.1 设计目的

JAX pytree 不能包含非数组类型的叶子节点。但 Vode 需要存储：
- 能量函数（`energy_fn`）：一个 Python 函数
- 规则集（`ruleset`）：一个 `Ruleset` 对象（本身是 Module）
- 形状（`shape`）：一个 Python tuple 或 None
- 状态（`_status`）：一个字符串或 None
- 模式（`_mode`）：一个 `IntEnum` 或 None

`StaticParam` 让这些值被 JAX 视为"静态数据"（展平时放在辅助数据中，不参与梯度计算）。

### 8.2 实现机制

```python
class StaticParam(BaseParam):
    def __init__(self, value=None):
        super().__init__(None)           # _value 始终是 None！
        self._static_value = value       # 真正的值存在这里

    def get(self) -> Any:
        return self._static_value        # 重写：返回 _static_value 而非 _value

    def set(self, value):
        self._static_value = value       # 重写：设置 _static_value 而非 _value
```

**关键设计**：`_value` 设为 `None`（动态部分为空），真正的值放在 `_static_value`。在 JAX 展平时：
- 动态部分（`_value = None`）进入 children tuples
- 静态部分（`_static_value` 和 `__dict__` 中除 `_value` 外的所有属性）进入 aux_data

这使得 `StaticParam` 在 JAX 看来是一个"不变的静态值"。

### 8.3 方法代理

`StaticParam` 重载了大量魔术方法，将操作代理到被包装的值：

```python
def __call__(self, *args, **kwds):       # 函数调用代理
    return self._static_value(*args, **kwds)
    # 效果：self.energy_fn(self, rkg=rkg) 等价于 se_energy(self, rkg=rkg)

def __getattr__(self, __name):           # 属性访问代理
    return getattr(self._static_value, __name)

def __getitem__(self, __idx):           # 索引代理（ruleset.rules 是 dict 用 StaticParam 包装）
    return self._static_value.__getitem__(__idx)

def __contains__(self, __key):           # in 操作代理
    return __key in self._static_value

def __iter__(self):                      # 迭代代理
    return iter(self._static_value)

def __len__(self):                       # len 代理
    return len(self._static_value)
```

**对 Vode 的意义**：
- `self.energy_fn(self, rkg=rkg)` — `energy_fn` 是 `StaticParam`，但 `__call__` 让它可以被调用
- `self.ruleset.rules.items()` — `ruleset` 是 `StaticParam`，但 `__getattr__` 让 `.rules` 可以访问
- `self.shape.set(h.shape)` — `shape` 是 `StaticParam`，可以正常 `.set()` / `.get()`

### 8.4 `static()` —— 智能包装函数

```python
def static(x: Any | StaticParam) -> StaticParam:
    return x if isinstance(x, StaticParam) else StaticParam(x)
```

如果传入的已经是 `StaticParam`，直接返回（避免重复包装）；否则创建新的 `StaticParam`。这是幂等性设计。

---

## 9. 核心依赖（三）：`Ruleset`

**文件**：[pcx/predictive_coding/_vode.py:46-157](pcx/predictive_coding/_vode.py#L46-L157)

### 9.1 设计目的

`Ruleset` 是 Vode 的**声明式规则引擎**。它定义了"在什么状态下，输入什么键，应该输出什么值到什么地方"。这让 Vode 的行为完全可配置，无需修改代码。

### 9.2 数据结构

```python
class Ruleset(BaseModule):       # 继承 BaseModule 而非 Module（不需要 train/eval 模式）
    def __init__(self, rules: Dict[str, Sequence[str]], tforms: dict = {}):
        self.rules = static(rules)    # {regex_pattern: [rule_strings]}
        self.tforms = static(tforms)  # {tform_name: transformation_function}
```

**`rules` 的结构示例**：
```python
{
    "init":  ("h, u <- u",),                       # INIT 状态下，u 写入 h 和 u
    ".*":    ("h <- u:se", "h -> u:zero"),          # 任何状态下...
}
```

**`tforms` 的结构示例**：
```python
{
    "se": lambda node, key, value, rkg: 0.5 * (value - node.get("u")) ** 2,
    "zero": lambda node, key, value, rkg: jnp.zeros_like(value),
}
```

### 9.3 `filter()` —— 规则匹配

`Ruleset` 通过字符串规则来定义 Vode 的输入/输出行为：

```
输入规则：target <- key:transformation   （如 "h, u <- u:se:zero"）
输出规则：key   -> target:transformation （如 "h -> u:se"）
```

- `target`：要写入的参数名，可逗号分隔多个（`h, u`）。
- `key`：被查询/写入的源键名（如 `u`）。
- `transformation`：变换链，`:` 分隔（如 `se:zero` 表示先 `se` 再 `zero`）。

`Ruleset.filter` 用正则把这些规则解析成 `(target, tform)` 元组：

```python
# pcx/predictive_coding/_vode.py 中的 Ruleset.filter
def filter(self, status, rule_pattern):

    for _pattern, _rules in self.rules.items():
        if re.match(_pattern, status) is None:   # ① 先取出self.rules中符合当前status的
            continue
            
        for _rule in _rules:
            if _match := re.match(rule_pattern, _rule):  # ② 再匹配规则
                yield _match.group(1, 2)   # ③ 取出两个捕获组
```

**两层正则匹配**：
1. 外层：`_pattern` 匹配当前 `status`（如 `"init"` 匹配规则组的 key `"init"`）
2. 内层：`rule_pattern` 匹配每条规则字符串（如 `"(.*)<-(u.*)"` 匹配 `"h, u <- u:se"`）
3. `_match.group(1, 2)` 返回 `("h, u", "u:se")`

### 9.4 `apply_set_transformation()` —— 递归应用变换

```python
def apply_set_transformation(self, node, tform, key, value, rkg):
    if ":" in tform:
        tform, _t = tform.rsplit(":", 1)         # "u:se:zero" → tform="u:se", _t="zero"

        value = self.tforms[_t](                  # 从右向左递归
            node, key,
            self.apply_set_transformation(node, tform, key, value, rkg),
            rkg,
        )
    return value
```

**递归处理链式变换**：`"u:se:zero"` 表示先应用 `se` 变换再应用 `zero` 变换。使用 `rsplit(":", 1)` 从右向左递归——先处理 `"u:se"`（得到 se 的结果），再将其传入 `zero`。

### 9.5 `apply_get_transformation()` —— 获取时的变换

```python
def apply_get_transformation(self, node, tform, key, rkg):
    _value = node.get(tform, None)           # 1. 先尝试直接获取（可能已在缓存中）

    if _value is None and ":" in tform:      # 2. 未命中且是链式变换
        tform, _t = tform.rsplit(":", 1)

        _value = self.tforms[_t](            # 3. 递归应用变换
            node, key,
            self.apply_get_transformation(node, tform, key, rkg),
            rkg,
        )

    return _value
```

**与 set 版的关键区别**：get 版先尝试 `node.get(tform, None)`——如果结果已经缓存在 `self.cache[tform]` 中，直接返回，避免重新计算。这实现了**基于缓存的惰性求值**。

---

## 10. 核心依赖（四）：能量函数

**文件**：[pcx/predictive_coding/_energy.py](pcx/predictive_coding/_energy.py)

### 10.1 设计目的

能量函数定义了 Vode 的**概率解释**——Vode 的状态 `h` 应该与传入的预测 `u` 有多接近。不同的能量函数对应不同的概率分布。

### 10.2 `se_energy` —— 平方误差（高斯分布）

```python
def se_energy(vode, rkg=RKG):
    """Squared error energy function derived from a Gaussian distribution."""
    e = vode.get("h") - vode.get("u")   # h 和 u 之间的差异
    return 0.5 * (e * e)                 # ½ * 差异的平方（负对数高斯似然）
```

**概率解释**：假设 `h | u ~ N(u, I)`，负对数似然 = `0.5 * ||h - u||² + const`。这是 Vode 的**默认能量函数**。

**在能量最小化中**：`dE/dh = h - u` → 梯度下降将 h 拉向 u。如果 u = 0（没有传入激活），Vode 保持当前状态不变。

### 10.3 `ce_energy` —— 交叉熵（分类分布）

```python
def ce_energy(vode, rkg=RKG):
    """Cross entropy energy function derived from a categorical distribution."""
    return -(vode.get("h") * jax.nn.log_softmax(vode.get("u")))
```

**概率解释**：假设 `h | u ~ Categorical(softmax(u))`，用于分类任务。`h` 通常是 one-hot 标签，`u` 是 logits。

### 10.4 `zero_energy` —— 无约束

```python
def zero_energy(vode, rkg=RKG):
    """Used to unconstrain the value of a vode from its prior distribution."""
    return jax.numpy.zeros((1,))
```

**使用场景**：当 Vode 的值不应受任何外部约束时（例如网络的输出层，它只需要匹配目标而不需要有内部先验）。

### 10.5 能量函数的签名约定

所有能量函数遵循相同的签名 `(vode, rkg) -> jax.Array`：
- `vode`：Vode 实例，提供 `get("h")` 和 `get("u")` 访问
- `rkg`：随机密钥生成器（大多数能量函数不需要随机性，但签名保持一致）
- 返回值：一个 JAX 数组（可能有多维，由 `Vode.energy()` 按 batch 维度做 reshape/sum）

---

## 11. 核心依赖（五）：`RandomKeyGenerator` / `RKG`

**文件**：[pcx/core/_random.py](pcx/core/_random.py)

### 11.1 设计目的

JAX 的随机数生成是**纯函数式**的——需要显式传递和更新 PRNG 密钥。这让代码变得繁琐。PCX 的 `RandomKeyGenerator` 将 PRNG 状态封装为可变对象，提供命令式接口。

### 11.2 实现

```python
class RKGState(Param):                        # PRNG 状态本身也是 Param
    def __init__(self, seed: int):
        super().__init__(jax.random.PRNGKey(seed))

    def split(self, n: int):
        values = jax.random.split(self.get(), n + 1)
        self.set(values[0])                   # 更新内部状态（消耗了 1 个 key）
        return values[1:]                      # 返回 n 个新 key

class RandomKeyGenerator(BaseModule):
    def __init__(self, seed: int = 0):
        super().__init__()
        self.key = RKGState(seed)

    def __call__(self, n: int = 1):
        _k = self.key.split(n)
        return _k[0] if n == 1 else _k         # 便捷：n=1 时返回单个 key

# 全局单例
RKG = RandomKeyGenerator(time.time_ns())       # 用纳秒级时间戳作为种子
```

**对 Vode 的意义**：`Vode.__call__` 的 `rkg` 参数默认值为 `RKG`（全局单例），所以用户不需要手动管理 PRNG 密钥。当需要在 Vode 内部使用随机性时（例如自定义能量函数中的随机采样），直接用 `rkg()` 获取新密钥。

---

## 12. 核心依赖（六）：`tree_apply`

**文件**：[pcx/core/_tree.py:85-130](pcx/core/_tree.py#L85-L130)

### 12.1 设计目的

遍历 PyTree（或 PyDAG），找到匹配过滤条件的节点，对其执行副作用操作（不改变树结构，只改变节点内容）。

### 12.2 实现

```python
def tree_apply(fn, filter_fn, tree, recursive=True):
    def _wrap_fn(x):
        if r := filter_fn(x):
            fn(x)            # 副作用：直接修改节点
        return r             # 返回 True 表示"这是叶子，不要继续展开"

    leaves = jtu.tree_leaves(tree, is_leaf=_wrap_fn)

    if recursive:
        for leaf in leaves:
            for x in eqx.tree_flatten_one_level(leaf)[0]:
                if x is not tree:
                    tree_apply(fn, filter_fn, tree=x)   # 递归进入
```

### 12.3 对 Vode 的意义

Vode 本身不在自己的方法中调用 `tree_apply`。但它被 `EnergyModule.clear_params()` 使用，而 Vode 的缓存清除依赖这个机制：

```python
# 在 step() 上下文管理器中：
module.clear_params(VodeParam.Cache)
# → 递归遍历所有子模块
# → 找到所有 isinstance(x, VodeParam.Cache) 的参数
# → 执行 x.set(None) → 清除缓存
```

此外，`Module.train()` / `Module.eval()` 也使用 `tree_apply` 递归设置模式和 equinox 的 inference 标志。

---

## 13. 核心依赖（七）：`get()` 和 `set()` 工具函数

**文件**：[pcx/core/_parameter.py:337-365](pcx/core/_parameter.py#L337-L365)

### 13.1 设计目的

在处理混合类型（裸值 + BaseParam）时，需要一个统一的方式提取/设置值。

### 13.2 实现

```python
def get(x: Any | BaseParam) -> Any:
    """如果 x 是 BaseParam，返回 x.get()；否则返回 x 本身。"""
    if isinstance(x, BaseParam):
        return x.get()
    else:
        return x

def set(obj: Any, x: Any | BaseParam) -> Any | BaseParam:
    """如果 obj 是 BaseParam，调用 obj.set(get(x))；否则直接赋值。"""
    if isinstance(obj, BaseParam):
        obj.set(get(x))
    else:
        obj = set(x)
    return obj
```

### 13.3 使用场景

在 `Param` 的运算符重载中大量使用 `get()`：

```python
def __add__(self, __other):
    return self._value.__add__(get(__other))
    #                        ^^^
    # __other 可能是 Param 实例（需要 get()）或裸数字（直接使用）
```

Vode 自身不直接使用 `get()`/`set()`（因为 Vode 清楚地知道自己操作的每个属性是否是 Param），但它间接受益于 `Param` 内部使用这些函数实现的运算符重载。

---

## 14. 核心依赖（八）：`STATUS`

**文件**：[pcx/predictive_coding/_vode.py:34-43](pcx/predictive_coding/_vode.py#L34-L43)

### 14.1 设计目的

`STATUS` 是一个简单的命名空间类，提供语义化的状态常量。

```python
class STATUS:
    NONE = None      # 无状态（默认）
    ALL = ".*"       # 匹配所有状态（用于规则的正则模式）
    INIT = "init"    # 初始化状态（前向初始化）
```

它不是 `Enum`（因为值不固定——用户可以随时用任意字符串作为状态），只是一个**推荐常量**的集合。用户完全可以写 `vode.status = "my_custom_status"`，只要规则集中有对应的正则模式即可。

---

## 15. 完整调用流程示例

以下示例综合展示了 Vode 的各层继承和依赖如何协同工作：

```python
import pcx
import jax
import jax.numpy as jnp

# 1. 创建一个 Vode（默认配置）
vode = pcx.Vode()
#    → _BaseModuleMeta.__new__ 自动注册为 JAX pytree
#    → BaseModule.__init__ （空，无操作）
#    → Module.__init__ 设置 _mode = static(None)
#    → EnergyModule.__init__ 设置 _status = static(None)
#    → Vode.__init__:
#        self.h = VodeParam()                    # 初始值为 None
#        self.cache = VodeParam.Cache()          # 空字典
#        self.energy_fn = static(se_energy)      # 包装为静态参数
#        self.ruleset = Ruleset({"init": ("h, u <- u",)})  # 默认规则
#        self.shape = static(None)

# 2. 前向传播：设置输入激活
x = jnp.ones((32, 128))  # batch=32, dim=128
vode.status = "init"      # → EnergyModule.status.setter
                          # → self._status.set("init")

h = vode(x)               # → Vode.__call__
                          # → self.set("u", x)
                          #   → ruleset.filter("init", "... <- u...")
                          #   → 匹配 "h, u <- u"
                          #   → self.h.set(x)    # 前向初始化 h
                          #   → self.cache["u"] = x
                          # → self.get("h")
                          #   → h 是 Param → 返回 self.h.get()

# 3. 计算能量
vode.status = ".*"         # 切换到通配状态

E = vode.energy()          # → Vode.energy()
                           # → self.energy_fn(self, rkg=RKG)
                           #   → se_energy(vode, rkg)
                           #   → e = vode.get("h") - vode.get("u")
                           #   → return 0.5 * (e * e)
                           # → reshape + sum → (32,) 形状

# 4. 网络级能量求和（递归）
class PCN(pcx.EnergyModule):
    def __init__(self):
        super().__init__()
        self.vode1 = pcx.Vode()
        self.vode2 = pcx.Vode()

pcn = PCN()
total_E = pcn.energy()     # → EnergyModule.energy()
                           # → sum(m.energy() for m in submodules(cls=EnergyModule))
                           # → vode1.energy() + vode2.energy()
```

---

## 16. 总结

### 16.1 继承链职责分工

| 层级 | 类名 | 核心职责 | 关键方法/属性 |
|------|------|---------|-------------|
| 元类 | `_BaseModuleMeta` | 自动 JAX pytree 注册 | `__new__`, `flatten_module`, `unflatten_module` |
| 1 | `BaseModule` | 最小公共基础 | `submodules()`, `__repr__` |
| 2 | `Module` | 训练/评估模式 | `MODE`, `mode()`, `train()`, `eval()`, `is_train`, `is_eval` |
| 3 | `EnergyModule` | 状态 + 能量递归 | `status`, `energy()` (递归), `clear_params()` |
| 4 | `Vode` | 预测编码节点 | `h`, `cache`, `ruleset`, `energy_fn`, `__call__`, `set()`, `get()`, `energy()` (单节点) |

### 16.2 核心设计模式

| 模式 | 体现位置 | 目的 |
|------|---------|------|
| **元类自动注册** | `_BaseModuleMeta` | 零样板代码，所有 Module 自动成为 pytree |
| **透传参数** (`*args, **kwargs`) | `Vode.__init__` → `param_type(...)` | Vode 不依赖 VodeParam 的具体签名 |
| **依赖注入** | `param_type=`, `energy_fn=`, `ruleset=` | 所有行为都可替换 |
| **声明式规则** | `Ruleset` + 正则匹配 | 数据流行为由配置决定，非硬编码 |
| **静态包装** | `static()` / `StaticParam` | 让非 JAX 类型存在于 pytree 中 |
| **属性代理** | `__getattr__`, `@property` | 封装内部实现，对外暴露简洁接口 |
| **惰性计算** | `energy()` 的 `"E" in cache` 检查 | 避免重复计算 |
| **链式调用** | `set()` 返回 `self` | `vode.set("u", x).set("prior", p)` |
