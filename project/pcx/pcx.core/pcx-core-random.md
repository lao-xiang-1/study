# PCX 核心模块：`_random.py` 详解

> 源文件：`pcx/core/_random.py`

## 背景

JAX 采用显式随机数：每次随机操作都要传入一个 `PRNGKey`，并由调用者负责分裂/管理 key 的状态。这保证了可复现性与可并行性，但写起来繁琐。

pcx 提供一个**有状态**的随机数生成器模块 `RandomKeyGenerator`（全局实例别名 `RKG`），内部维护 key，对外暴露简单的"取 n 个 key"接口，且作为 `BaseModule` 可被 pcx 追踪。

---

## 1. `RKGState(Param)` — 随机状态参数

```python
class RKGState(Param):
    def __init__(self, seed: int):
        super().__init__(jax.random.PRNGKey(seed))
    def seed(self, seed: int): ...
    def split(self, n: int): ...
```

继承自 `Param`，因此其持有的 `PRNGKey` 是一个被追踪的动态值。

### `__init__(seed)`
用种子生成初始 `PRNGKey`，存为 `_value`。

### `seed(seed)`
重置为新的 `PRNGKey(seed)`。

### `split(n)` — 生成 n 个 key 并推进内部状态
```python
values = jax.random.split(self.get(), n + 1)
self.set(values[0])      # 留下 1 个作为新状态
return values[1:]        # 返回 n 个给调用方
```
关键：`split(n+1)` 多分裂一个，把第 0 个写回自身作为"下一个状态"，其余 n 个返回。这样**每次调用后状态都前进**，避免重复使用同一 key。

---

## 2. `RandomKeyGenerator(BaseModule)` — 随机数生成器模块

```python
class RandomKeyGenerator(BaseModule):
    def __init__(self, seed: int = 0):
        super().__init__()
        self.key = RKGState(seed)
    def seed(self, seed: int = 0): ...
    def __call__(self, n: int = 1): ...
```

### `__init__`
持有一个 `RKGState` 作为唯一状态（`self.key`）。因继承 `BaseModule`，整个生成器是 pytree，可通过 pcx 变换传递且状态被追踪。

### `seed(seed=0)`
重置内部状态（委托给 `RKGState.seed`）。

### `__call__(n=1)` — 生成 n 个 key
```python
_k = self.key.split(n)
return _k[0] if n == 1 else _k
```
便利设计：`n=1` 时直接返回单个 key（而非长度为 1 的元组），省去 `rkg(1)[0]` 的拆包；`n>1` 返回元组。

```python
key = RKG()                 # 取 1 个 key
k1, k2, k3 = RKG(3)         # 取 3 个 key
```

---

## 3. `RKG` — 全局默认生成器

```python
RKG = RandomKeyGenerator(time.time_ns())
```
- 模块加载时即创建一个**全局**实例，种子取当前纳秒时间，保证每次运行不同。
- 用户可随时 `RKG.seed(seed)` 改成固定种子以复现。
- 从 `pcx.core` 导出时：`RandomKeyGenerator` 指向**类**，`RKG` 指向该类的**全局实例**。

> ⚠️ **注意命名**：`RandomKeyGenerator` 是**类**，`RKG` 是该类的**全局实例**。日常使用直接调 `RKG()` 取 key；若需自建独立生成器才用 `RandomKeyGenerator(seed)`。

---

## 使用示例

```python
from pcx.core import RKG
import jax

k = RKG()                       # 单个 key
x = jax.random.normal(k, (3,))

RKG.seed(42)                    # 固定种子，便于复现
k1, k2 = RKG(2)
```

---

## 依赖关系

`_random.py` 依赖 [_parameter.py](pcx-core-parameter.md)（`Param`）与 [_module.py](pcx-core-module.md)（`BaseModule`）。
