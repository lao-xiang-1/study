---
sr-due: 2026-09-18
sr-interval: 3
sr-ease: 250
---
#code 

使 自注意力机制 具有**感知位置**的能力。如果没有位置编码，“猫吃鱼”和“鱼吃猫”在模型看来是一样的。

## 1. 绝对位置编码

这种编码方式简单粗暴：根据位置信息生成一组向量，直接加到原数据上

### 数学公式
$$
PE_{(pos, i)} = 
\begin{cases}
\sin\left(\dfrac{pos}{10000^{2i/d_{\text{model}}}}\right) & \text{if } i \bmod 2 = 0 \\
\cos\left(\dfrac{pos}{10000^{2(i-1)/d_{\text{model}}}}\right) & \text{otherwise}
\end{cases}
$$

- $i$ 表示隐藏层的维数
- $pos$ 表示向量在序列中的位置

记 $10000^{2i/d_{\text{model}}}, 10000^{2(i-1)/d_{\text{model}}} = \theta$，其与数学上角频率和频率的联系：
$$\omega = \dfrac{1}{\theta}, \quad f = \dfrac{1}{2\pi\theta}$$
- 低维->高频，高维->低频
- 相邻维度的 $\theta$ 相同（很好理解：$i$ 取0和1时，$\theta$ 相同。取2和3也相同）

因此 $d_{\text{model}}$ 个维度两两配对，实际只需生成 $d_{\text{model}}/2$ 个不同的角度——这也对应下面代码中 `torch.arange(0, emb_dim, 2)` 步长为 2 的写法。

#### 函数图
>假设总维度 $d_{model}=32$ ，序列长度（pos的最大值）为64。画出 $PE_{(pos, i)}$ 的值：

![](assets/Position%20Encodding.png)

- 当 $i$ 相同时，PE 相当于 $pos$ 的三角函数值
- 同pos相邻维度的区别只有一个用sin，一个用cos（频率一样）
- 当 $i$ 较大时，角度约等于0，PE几乎是在 0 和 1 之间交替变换
- 图中可以很明显看出一条类似指数增长的曲线，这其实很容易在数学上表示：
	这条曲线上的点颜色相近，所以直接看作数值相同。从0开始，sin和cos第一次数值相同是在 $\dfrac{\pi}{4}$ 的位置。在 i 为偶数时，令系数=$\dfrac{\pi}{4}$，即：$$
	\begin{align}
	& \dfrac{pos}{10000^{2i/d_{\text{model}}}} = \dfrac{\pi}{4} \\
	& pos = \dfrac{\pi}{4}10000^{2i/d_{\text{model}}}
	\end{align}
	$$
	可以看到pos和 i 呈指数形式。因为相邻维度的频率相同，所以 i 为奇数时也一样。


#### 深入理解：为什么高维频率低

维度序号 $i$ 越大，指数 $2i/d_{\text{model}}$ 越大，波长 $\lambda_i = 2\pi \cdot 10000^{2i/d_{\text{model}}}$ 越长、频率越低——对位置越「钝」。这是刻意的**多尺度（multi-scale）**设计：

- **低维（$i$ 小）→ 高频、波长短**：对位置敏感，负责区分相邻、很近的位置差异。
- **高维（$i$ 大）→ 低频、波长长**：对位置迟钝，负责刻画很远、全局的位置趋势。

两者配合，模型既能判断「5 和 6 挨着」这种细粒度关系，也能判断「第 100 个 token 在第 200 个之前」这种粗粒度关系。

| 维度对 $i$（$d_{\text{model}}=512$） | 指数 $2i/d_{\text{model}}$ | 波长 | 作用 |
| :--- | :--- | :--- | :--- |
| $0$ | $0$ | $\approx 6.28$ | 相邻几个位置内就变化，捕捉相邻关系 |
| $128$ | $0.5$ | $\approx 628$ | 中等距离 |
| $255$ | $\approx 0.996$ | $\approx 6\times10^4$ | 超过典型序列长度，几乎不变，捕捉全局趋势 |

> 波长从 $2\pi$ 到 $2\pi\times10000$ 呈几何级数铺开，保证每个尺度都有覆盖、没有盲区。若全部用同一频率：全高频则长距离信息冗余、无法刻画全局；全低频则区分不了相邻位置。

> 补充：正弦编码还有一个性质——任意偏移 $k$ 后的 $PE_{pos+k}$ 都能写成 $PE_{pos}$ 的线性组合（由 $\sin(\theta+\phi)=\sin\theta\cos\phi+\cos\theta\sin\phi$ 推出），因此模型能方便地学到相对位置。这与「频率扫描」是两件独立的事。

---

### 代码
```python
# 生成位置编码（PE）
def position_encode(max_len, emb_dim):
    pe = torch.zeros(max_len, emb_dim)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
    pos_sin = torch.sin(position * div_term)
    pos_cos = torch.cos(position * div_term)
    pe[:, 0::2] = pos_sin
    pe[:, 1::2] = pos_cos
    return pe

# 直接把pe加到原数据上
# input_embeddings 是 tokenizer 最后计算出的词嵌入向量
x = input_embeddings + pe
```


## 2.旋转位置编码

### 数学原理
[Rope数学原理](attachment/Rope数学原理.md)
频率的计算方法其实和绝对位置编码相同（和[数学公式](#数学公式)一样），只是这里应用了旋转矩阵

### 代码实现（MiniMind 版本）

#### 预计算频率表

```python
def precompute_freqs_cis(dim, end, rope_base=1e6, rope_scaling=None):
    freqs = torch.exp(torch.arange(0, dim, 2)[: dim//2].float() * (-math.log(rope_base) / dim))
    ...
    t = torch.arange(end) # pos
    freqs = torch.outer(t, freqs).float() # pos * freqs, shape = [seq, dim]
    # 复制后拼接
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin
```

**参数说明：**

1. 必传参数（没有默认值）
  - `dim`：**每个注意力头的维度**（head_dim），如 `hidden_size // num_attention_heads` = 512/8 = 64，**不是** hidden_size。它决定频率向量的长度 `dim/2`。
  - `end`：**位置表的最大长度**（`max_position_embeddings`），即 `t = torch.arange(end)` 的长度，也是返回 cos/sin 表第一维的大小。

2. 可选参数（有默认值）
  - `rope_base`：**频率基（base）**，默认 `1e6`（MiniMind；原论文用 1e4）。决定频率衰减速度 $\theta_i = \text{base}^{-2i/d}$；越大 → 低频波长越长 → 可分辨的位置范围越大。
  - `rope_scaling`：**长度外推参数**（YaRN），默认 `None` 表示不启用。传入 dict（含 `beta_fast`/`beta_slow`/`factor` 等键）时按维度对频率缩放，用于超训练长度的推理外推。

**返回：** `(freqs_cos, freqs_sin)`，形状均为 `[end, dim]`——频率只有 `dim/2` 个，`cat([cos, cos])` 复制拼接后才变 `dim`。两张表以 buffer 形式注册，forward 时按位置切片取用，不重复计算三角函数。

#### 应用旋转

```python
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
	# 把 x 后半取负放前面：[x₀,x₁,x₂,x₃] → [-x₂,-x₃,x₀,x₁]
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed
```

**参数说明：**

1. 必传参数（没有默认值）
  - `q`：**查询向量**，形状 `[batch, seq, heads, head_dim]`。旋转后返回同形状的 `q_embed`。
  - `k`：**键向量**，形状同 `q`。旋转后返回同形状的 `k_embed`。
  - `cos` / `sin`：**预计算的三角函数表**，形状 `[seq, head_dim]`，来自 `precompute_freqs_cis` 的返回值。

2. 可选参数（有默认值）
  - `unsqueeze_dim`：**在哪一维插入 1 以广播到所有头**。默认 `1`，对应 `[batch, seq, heads, d]` 布局——在 `heads` 处插 1，让 cos/sin 覆盖所有注意力头（每头同一套频率）。

**返回：** `(q_embed, k_embed)`，形状分别与 `q`、`k` 完全一致（RoPE 是保形变换，不改变任何一维）。

##### 形状与广播
>补充知识：[broadcasting](../../code/python/torch/broadcasting.md)，[杂项](../../code/python/torch/杂项.md)

当两个张量相乘时，python会尝试进行广播，从**最后一维往左对齐**，逐维比较：

1. 两边该维**相等** → 直接用；
2. 一边该维是 **1**（或**根本没有这一维**，视作 1）→ 这一维被"撑大"到另一边的大小；
3. 否则（比如 3 对 5，都不为 1）→ 无法广播，直接 `RuntimeError`。

以 `q * sin` 为例，由于 `sin` 少了一个 `heads` 维度，所以需要进行扩展（`sin.unsqueeze(1`）以填充一个维度，结果如下：
```
q    : [batch, seq, heads, d]      = [  4, 512,  8, 64]
sin  : [seq, 1, d]                 = [     512,  1, 64]    # unsqueeze(1) 之后
```
当两个向量相乘（`*`）时，sin会自动变形为 `[4, 512, 8, 64]`，然后两者逐元素相乘。这意味着每个注意力头用同一套频率。

##### 维度配对方法
>前三维 `batch, seq, heads` 操作相同，只考虑最后一维`dim`

设 $d=4$，两种配对方式：

| 变体                              | 旋转对              | 出处                               |
| ------------------------------- | ---------------- | -------------------------------- |
| 交错式（interleaved）                | (x₀,x₁), (x₂,x₃) | RoFormer 原论文                     |
| **半分式（half / non-interleaved）** | (x₀,x₂), (x₁,x₃) | Llama / GPT-NeoX，**MiniMind 采用** |
数学上两者等价——只是同一组权重的两种排列，训练出的模型互不兼容，但各自正确。

对于当前代码使用的配对方法，假设原始数据是 `[x₀, x₁, x₂, x₃]`，那么应用旋转后的向量：

| 索引  | 值                   |
| --- | ------------------- |
| 0   | `x₀cosθ₀ − x₂sinθ₀` |
| 1   | `x₁cosθ₁ − x₃sinθ₁` |
| 2   | `x₂cosθ₀ + x₀sinθ₀` |
| 3   | `x₃cosθ₁ + x₁sinθ₁` |

相当于对 (x₀,x₂), (x₁,x₃) 使用旋转矩阵（各分量对用各自的频率）。

#### 调用方法

```python
# MiniMindModel.__init__：建表
freqs_cos, freqs_sin = precompute_freqs_cis(dim=hidden_size // num_attention_heads,
                                             end=max_position_embeddings, ...)

# Attention.forward：旋转 q/k（注意只旋 q、k，不旋 v）
cos, sin = position_embeddings
xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
```

**只旋转 q、k，不旋转 v**：位置信息只需进入 $QK^\top$ 内积参与打分；v 携带的是内容语义，无需位置化。

**预计算 + 查表**：cos/sin 按 `max_position_embeddings` 全长算好存成 buffer，每次 forward 直接切片取用，不重复计算三角函数。

---

## YaRN (Yet another RoPE extensioN)
>基于RoPE，扩展序列长度

问题：训练长度 2048，推理要用 32768。位置 2048~32768 的相位组合训练时从未见过，注意力分数崩坏。

朴素方案是**位置插值（PI）**：所有频率统一除以 factor，等效把位置轴压缩 16 倍——但这对高频维度是伤害：相邻 token 的相位差被压到 1/16，模型失去分辨相邻位置的能力。

**YaRN 的改进：按频率分维度区别对待**

### 数学原理

YaRN 用每个维度的**波长** $\lambda_i$ 作为判据：
$$\lambda_i = 2\pi\theta_i = 2\pi\cdot\text{base}^{2i/d}$$
定义"在训练长度内转的圈数"：
$$r_i = \frac{L_{\text{train}}}{\lambda_i}$$

圈数多 = 高频，圈数少 = 低频。YaRN 的规则是：

| 圈数 | 处理 | ramp $\gamma_i$ |
|---|---|---|
| $r_i > \beta_{\text{fast}}$（高频） | **外推**：频率不变 | $0$ |
| $\beta_{\text{slow}} < r_i < \beta_{\text{fast}}$ | 线性过渡 | $0\to1$ |
| $r_i < \beta_{\text{slow}}$（低频） | **插值**：频率 $\div s$ | $1$ |

其中 ramp 可以用以下公式表示：
$$\gamma_i = \operatorname{clamp}\!\left(\frac{i - \text{low}}{\text{high} - \text{low}},\ 0,\ 1\right)$$

每个维度的新频率：
$$\theta_i' = \theta_i\Big[(1-\gamma_i) + \frac{\gamma_i}{s}\Big]$$

也就是 $\gamma=0$ 时 $\theta_i'=\theta_i$（外推），$\gamma=1$ 时 $\theta_i'=\theta_i/s$（插值），中间线性混合。

### 代码实现

```python
# 如果输入序列比训练数据长
if end / orig_max > 1.0:
    inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
    low  = max(math.floor(inv_dim(beta_fast)), 0)
    high = min(math.ceil(inv_dim(beta_slow)),  dim // 2 - 1)
    ramp = torch.clamp((torch.arange(dim // 2).float() - low) / max(high - low, 0.001), 0, 1)
    freqs = freqs * (1 - ramp + ramp / factor)
```

- `inv_dim(b)`：反解"波长恰好等于 b 个位置"的维度索引（由 $\lambda_i = 2\pi/\theta_i = b$ 解出 $i$）；
- **索引 ≤ low（高频，波长 < beta_fast=32）**：ramp=0，频率不变 → 保持精细局部分辨能力（外推）；
- **索引 ≥ high（低频，波长 > beta_slow=1×训练长度量级）**：ramp=1，频率 ÷ 16 → 插值，把没见过的相位压回已见范围；
- 中间线性过渡。

即每个维度 $f'(i) = f(i)\left[(1-\gamma) + \gamma/s\right]$，γ 是 ramp。

配置（config 中）：

```python
rope_scaling = {
    "beta_fast": 32, "beta_slow": 1, "factor": 16,
    "original_max_position_embeddings": 2048,
    "attention_factor": 1.0,   # YaRN 的注意力温度，这里不调
    "type": "yarn"
} if inference_rope_scaling else None   # 默认 False
```

注意：训练与推理的 scaling 设置需一致，否则频率表分布突变，效果通常要微调恢复。
