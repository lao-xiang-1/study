---
sr-due: 2026-09-05
sr-interval: 3
sr-ease: 250
---
#code 

# pandas 十分钟入门

来源：[10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html)（pandas 3.0.5）

## 导入

```python
import numpy as np
import pandas as pd
```

## 基本数据结构

pandas 提供两类核心数据结构：

- **Series**：一维带标签数组（labeled array），可存放任意类型数据（整数、字符串、Python 对象等）
- **DataFrame**：二维数据结构，类似二维数组或表格（行列式）

## 创建对象

**Series**：传入值列表，自动生成默认 `RangeIndex`：

```python
s = pd.Series([1, 3, 5, np.nan, 6, 8])
```

**DataFrame（NumPy 数组 + 日期索引）**：用 `pd.date_range()` 生成日期索引，再配合带标签的列：

```python
dates = pd.date_range("20130101", periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))
```

**DataFrame（字典）**：键为列标签，值为列数据：

```python
df2 = pd.DataFrame({
    "A": 1.0,
    "B": pd.Timestamp("20130102"),
    "C": pd.Series(1, index=list(range(4)), dtype="float32"),
    "D": np.array([3] * 4, dtype="int32"),
    "E": pd.Categorical(["test", "train", "test", "train"]),
    "F": "foo",
})
```

> 各列可以有不同的 dtype（`float64`、`datetime64[us]`、`float32`、`int32`、`category`、`str` 等）。用 `df2.dtypes` 查看。IPython 中列名支持 Tab 补全。

## 查看数据

| 方法 | 作用 |
|:---|:---|
| `df.head()` / `df.tail(n)` | 查看前 / 后若干行 |
| `df.index` / `df.columns` | 查看索引 / 列标签 |
| `df.to_numpy()` | 转成 NumPy 数组（不含行列标签） |
| `df.describe()` | 快速统计摘要（count、mean、std、min、四分位数、max） |
| `df.T` | 转置 |
| `df.sort_index(axis=1, ascending=False)` | 按轴（索引/列名）排序 |
| `df.sort_values(by="B")` | 按某列的值排序 |

> **NumPy 数组整体只有一个 dtype，而 DataFrame 每列各有一个 dtype。** `to_numpy()` 会寻找能容纳所有列 dtype 的 NumPy dtype；若公共类型为 `object`，则会触发数据复制。

## 选择（Selection）

> 交互式工作时标准 Python / NumPy 写法直观方便；但生产代码推荐使用优化过的访问方法：`at()`、`iat()`、`loc()`、`iloc()`。

### Getitem（`[]`）

- `df["A"]`：单列标签 → 返回 `Series`
- `df.A`：列名只含字母、数字、下划线时，可用属性写法
- `df[["B", "A"]]`：列表 -> 选多列，并按传入的顺序排列
- `df[0:3]`：切片 → 按行选
- `df["20130102":"20130104"]`：标签切片按行选

### 按标签选择（label）

- `df.loc[dates[0]]` 即 `df.loc["2013-01-01"]`：选某行
- `df.loc[:, ["A", "B"]]`：所有行 + 指定列
- `df.loc["20130102":"20130104", ["A", "B"]]`：标签切片**两端都包含**
- `df.loc[dates[0], "A"]`：单个行列标签 → 标量
- `df.at[dates[0], "A"]`：快速访问标量（等价）

### 按位置选择（position）

- `df.iloc[3]`：按整数位置选行
- `df.iloc[3:5, 0:2]`：整数切片（类似 NumPy/Python，**左闭右开**）
- `df.iloc[[1, 2, 4], [0, 2]]`：整数位置列表
- `df.iloc[1, 1]`：取单个值
- `df.iat[1, 1]`：快速访问标量（等价）

### 布尔索引

```python
df[df["A"] > 0]          # 选 A 列大于 0 的行
df[df > 0]               # 满足条件保留原值，否则为 NaN
df2[df2["E"].isin(["two", "four"])]   # isin() 过滤
```

### 赋值（Setting）

```python
df["F"] = s1                  # 新列自动按索引对齐
df.at[dates[0], "A"] = 0      # 按标签赋值
df.iat[0, 1] = 0              # 按位置赋值
df.loc[:, "D"] = np.array([5] * len(df))   # NumPy 数组赋值
df2[df2 > 0] = -df2           # where 式赋值（对满足条件的元素取反）
```

## 缺失数据

- 对 NumPy 数据类型，`np.nan` 表示缺失值，默认**不参与计算**
- `reindex()` 可在指定轴上增/删/改索引，返回副本
- `dropna(how="any")`：删除含缺失值的行
- `fillna(value=5)`：填充缺失值
- `pd.isna(df1)`：得到 NaN 位置的布尔掩码

```python
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ["E"])
df1.dropna(how="any")
df1.fillna(value=5)
pd.isna(df1)
```

## 运算（Operations）

### 统计（Stats）

运算一般**自动排除缺失值**：

```python
df.mean()          # 每列均值
df.mean(axis=1)    # 每行均值
```

> 与索引/列不同的 Series 或 DataFrame 运算时，结果会按**索引/列标签的并集对齐**，并自动在指定维度广播，未对齐的标签填 `np.nan`。例：`df.sub(s, axis="index")`。

### 自定义函数

- `df.agg(func)`：应用归约（reduce）型函数
- `df.transform(func)`：应用广播（broadcast）型函数

```python
df.agg(lambda x: np.mean(x) * 5.6)
df.transform(lambda x: x * 101.2)
```

### 值计数（Value Counts）

```python
s.value_counts()   # 统计每个值出现的次数
```

### 字符串方法

`Series.str` 属性提供一整套向量化字符串处理方法：

```python
s.str.lower()   # 逐个元素转小写
```

## 合并（Merge）

### Concat

按行拼接对象：

```python
pieces = [df[:3], df[3:7], df[7:]]
pd.concat(pieces)
```

> **给 DataFrame 加列相对较快；但加行需要复制，代价较高。** 建议把预先构建好的记录列表一次性传给 `DataFrame` 构造器，而不是逐行 append。

### Join（SQL 风格）

`merge()` 按指定列做 SQL 风格连接：

```python
pd.merge(left, right, on="key")
```

## 分组（Grouping）

「分组」指以下三步：**拆分（Splitting）→ 应用（Applying）→ 合并（Combining）**。

```python
df.groupby("A")[["C", "D"]].sum()   # 按单列分组后对选中列求和
df.groupby(["A", "B"]).sum()        # 多列分组 → 形成 MultiIndex
```

## 重塑（Reshaping）

### Stack / Unstack

- `stack()`：「压缩」DataFrame 列中的一层，得到 MultiIndex 的 Series
- `unstack()`：`stack()` 的逆操作，默认展开**最后一层**（`unstack(0)` 展开第一层）

```python
stacked = df2.stack()
stacked.unstack()     # 默认展开最后一层
stacked.unstack(1)
stacked.unstack(0)
```

### 透视表（Pivot tables）

`pivot_table()` 指定 `values`、`index`、`columns`：

```python
pd.pivot_table(df, values="D", index=["A", "B"], columns=["C"])
```

## 时间序列

- `resample()`：频率转换时的重采样（如秒级数据转 5 分钟级）

```python
rng = pd.date_range("1/1/2012", periods=100, freq="s")
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
ts.resample("5Min").sum()
```

- `tz_localize()`：给时间序列标注时区
- `tz_convert()`：转换到另一时区

```python
ts_utc = ts.tz_localize("UTC")
ts_utc.tz_convert("US/Eastern")
```

- 非固定时长偏移：`BusinessDay`

```python
rng + pd.offsets.BusinessDay(5)
```

## 分类数据（Categoricals）

- `astype("category")`：转为分类类型
- `cat.rename_categories()`：重命名分类
- `cat.set_categories()`：重排并同时补充缺失的分类（默认返回新 Series）

```python
df["grade"] = df["raw_grade"].astype("category")
df["grade"] = df["grade"].cat.rename_categories(new_categories)
df["grade"] = df["grade"].cat.set_categories([...])
```

- 分类排序按**分类顺序**而非字典序：`df.sort_values(by="grade")`
- 按分类列分组时 `observed=False` 会显示空分类：

```python
df.groupby("grade", observed=False).size()
```

## 绘图（Plotting）

约定引用 matplotlib：

```python
import matplotlib.pyplot as plt

ts = pd.Series(np.random.randn(1000), index=pd.date_range("1/1/2000", periods=1000))
ts = ts.cumsum()
ts.plot()
```

`df.plot()` 会绘制所有列。在 Jupyter 中绘图自动显示；否则用 `matplotlib.pyplot.show()` 显示或 `savefig()` 保存文件。

## 导入导出数据

| 格式 | 写入 | 读取 |
|:---|:---|:---|
| CSV | `df.to_csv("foo.csv")` | `pd.read_csv("foo.csv")` |
| Parquet | `df.to_parquet("foo.parquet")` | `pd.read_parquet("foo.parquet")` |
| Excel | `df.to_excel("foo.xlsx", sheet_name="Sheet1")` | `pd.read_excel("foo.xlsx", "Sheet1", index_col=None, na_values=["NA"])` |

## 常见坑（Gotchas）

对 Series / DataFrame 直接做布尔判断会报错：

```python
if pd.Series([False, True, False]):
    ...
# ValueError: The truth value of a Series is ambiguous.
# Use a.empty, a.bool(), a.item(), a.any() or a.all().
```

> 应使用 `a.empty`、`a.bool()`、`a.item()`、`a.any()`、`a.all()` 等明确判断。
