---
sr-due: 2026-09-17
sr-interval: 3
sr-ease: 250
---
#code 

# PyTorch 广播机制（Broadcasting）

广播（Broadcasting）是 PyTorch / NumPy 在**逐元素运算**（`+ - * /`、比较等）中的自动形状对齐规则：当两个张量形状不同时，不是直接报错，而是先把小的那个"**虚拟地撑大**"到和大的形状一致，再做逐元素运算。

> 本文由 RoPE 里的 `q * cos.unsqueeze(1)` 引出的问题而来，动机见 [[project/transformer-from-scratch/PE.md]]。

---

## 1. 一句话理解

```python
a = torch.tensor([[1, 2, 3],       # [2, 3]
                  [4, 5, 6]])
b = torch.tensor([[10, 20, 30]])   # [1, 3]

a * b
# tensor([[10,  40,  90],
#         [40, 100, 180]])
```

`b` 只有 1 行，被自动"复制"成 2 行后再逐元素相乘——这就是广播。

---

## 2. 三条规则

从**最后一维往左**逐维对齐比较：

1. 两维**相等** → 直接参与运算；
2. 一边为 **1**（或**缺维**，缺维视作 1）→ 该维被撑大到另一边的大小；
3. 其余情况（如 3 对 5，且都不为 1）→ 无法广播，抛 `RuntimeError`。

**关键点**：缺失的**前导维度**当作 1 处理。这正是"4 维张量能和 3 维张量相乘"的原因——不是看维数相同，而是看右对齐后每一维是否满足规则。

---

## 3. RoPE 实例：`q * sin.unsqueeze(1)`

```python
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    ...
```

以 MiniMind 的布局为例：

```python
q   : [batch, seq, heads, d] = [  2, 512,  8, 64]
sin : [seq, d]               = [     512,     64]
sin.unsqueeze(1)             = [     512,  1, 64]   # 在 heads 处插入 1
```

右对齐逐维比较：

| 维度 | q | `sin.unsqueeze(1)` | 判定 |
|---|---|---|---|
| dim 0（batch） | 2 | 缺（视作 1） | 撑到 2 ✓ |
| dim 1（seq） | 512 | 512 | 相等 ✓ |
| dim 2（heads） | 8 | 1 | 撑到 8 ✓ |
| dim 3（head_dim） | 64 | 64 | 相等 ✓ |

结果形状 `[2, 512, 8, 64]`。语义上就是"**每个 token 位置 m 的 q 分量，乘上该位置的 sin(m·θ)**"——batch 和 heads 这两维 sin 里没有（或为 1），自动广播覆盖，所以**同一套频率表能作用到所有样本、所有头**。

---

## 4. 具体数值演示

```python
q   = torch.tensor([[1, 2, 3, 4],       # [2, 4]
                    [5, 6, 7, 8]])
sin = torch.tensor([[10, 20, 10, 20]])  # [1, 4]

q * sin
# sin 先被广播成：
#   [[10, 20, 10, 20],
#    [10, 20, 10, 20]]
# 再逐元素相乘：
#   [[10, 40, 30, 80],
#    [50, 120, 70, 160]]
```

---

## 5. 为什么"虚拟复制"不占内存：stride = 0

广播**不会真的在内存里复制数据**，而是给被广播的那一维分配 `stride = 0`——即那维所有元素都指向同一块内存。这是 view（视图）而非 copy，几乎零开销。

三个易混操作的区别：

| 操作 | 是否复制数据 | 用途 |
|---|---|---|
| `unsqueeze(dim)` | 否（view） | 在指定位置插入尺寸 1 的维，配合广播 |
| `expand(...)` | 否（view） | 显式把尺寸 1 的维展开到目标大小 |
| `repeat(...)` | **是** | 真的复制数据，得到连续内存 |

```python
x = torch.randn(512, 1, 64)
y = x.expand(2, 512, 8, 64)   # 不占额外内存，stride 含 0
z = x.repeat(2, 1, 8, 1)      # 真的复制，占额外内存
```

---

## 6. 广播失败的情况

不是"任何不同形状都能乘"，必须满足规则：

```python
q   = torch.randn(2, 512, 8, 64)
sin = torch.randn(512, 3, 64)     # 中间是 3，不是 1

q * sin
# RuntimeError: The size of tensor a (8) must match the size of tensor b (3)
# at non-singleton dimension 2
```

右对齐后 `heads=8` 对 `3`，两者既不等也不为 1，无法广播。

---

## 7. 常见用法速查

| 场景 | 写法 | 广播结果 |
|---|---|---|
| 加标量 | `tensor + 2` | 每个元素都加 2 |
| 每通道加减 | `x[2,3,32,32] + bias[3,1,1]` | 每个通道加不同 bias |
| 行向量乘 | `x[2,3] * row[1,3]` | row 撑到 2 行 |
| 列向量乘 | `x[2,3] * col[3]`（1 维） | col 右对齐到最后一维 |
| 手动加维 | `x.unsqueeze(dim)` | 在目标维插 1，等待广播 |
| 显式展开 | `x.expand(shape)` | 得到 strided view，不复制 |

---

## 8. 要点速查

| 问题 | 答案 |
|---|---|
| 广播是什么 | 逐元素运算中，小张量被自动"虚拟复制"到大张量形状 |
| 对齐方向 | 从**最后一维往左**逐维比较 |
| 可广播条件 | 每维相等，或一方为 1 / 缺维 |
| 4 维 × 3 维为何能算 | 缺维视作 1，右对齐后逐维满足即可 |
| 是否复制数据 | 否，stride=0 的 view，零开销 |
| 与 NumPy 的关系 | 规则完全一致，两边通用 |
| 想真正复制 | 用 `repeat()`，而非广播 |
| RoPE 里为什么用 `unsqueeze(1)` | 在 heads 维插 1，让 cos/sin 广播到所有头 |
