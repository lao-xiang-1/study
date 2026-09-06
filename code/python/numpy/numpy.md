# NumPy

## np.linspace（线性等分序列）

`np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)`

在 `start` 到 `stop` 之间生成**等间距**的 `num` 个数（默认 50 个），常用于生成自变量取值、坐标轴刻度、绘图采样点等。

### 基本用法

```python
import numpy as np

np.linspace(0, 1, 5)
# array([0.  , 0.25, 0.5 , 0.75, 1.  ])
```

### 与 np.arange 的区别

| 函数 | 依据 | 是否含终点 |
| :--- | :--- | :--- |
| `np.linspace` | 指定**个数** `num`，步长自动计算 | 默认包含 |
| `np.arange` | 指定**步长** `step` | 不包含 |

> 需要精确控制"取多少个点"时用 `linspace`；需要精确控制"步长多大"时用 `arange`。

### 常用参数

- `start` / `stop`：区间起点、终点
- `num`：生成点的个数，默认 `50`
- `endpoint`：是否包含 `stop`，默认 `True`
  - `endpoint=True` 时步长 $h = \dfrac{stop - start}{num - 1}$
  - `endpoint=False` 时步长 $h = \dfrac{stop - start}{num}$
- `retstep`：若为 `True`，同时返回步长 `(array, step)`
- `dtype`：输出数组的数据类型
- `axis`：沿哪个轴生成序列（用于多维数组）

### 示例

```python
# 不包含终点
np.linspace(0, 1, 5, endpoint=False)
# array([0. , 0.2, 0.4, 0.6, 0.8])

# 返回步长
x, dx = np.linspace(0, 1, 5, retstep=True)
# x  = array([0.  , 0.25, 0.5 , 0.75, 1.  ])
# dx = 0.25
```

> 注意拼写：是 `linspace`（linear space），不是 `linespace`。
