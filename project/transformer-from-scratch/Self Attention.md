---
sr-due: 2026-10-07
sr-interval: 1
sr-ease: 230
---
#code 

# 3. Self Attention 笔记

> 对应 notebook：`transformers_from_scratch.ipynb` 第 61–104 个 cell
> 一句话：自注意力让每个 token 动态地从序列中其他 token 那里"取信息"，从而得到**上下文相关**的表示。

---

## 1. 核心问题与直觉

要回答的问题：**序列中第 i 个元素相对于同一序列中其他元素有多相关？**

传统做法 vs 注意力：

| | 做法 | 问题 |
|---|---|---|
| 传统 encoder | 学一个全局的源序列表示 | 与上下文无关，信息被压缩 |
| 注意力 | 为每个 token 学**上下文相关**的表示，生成目标 token 时动态组合最相关的源表示 | 计算量随序列长度平方增长 |

### Q / K / V 的语义

| 向量 | 含义 | 类比 |
|---|---|---|
| **Query** | 这个 token 在"找什么" | 搜索关键词 |
| **Key** | 这个 token "提供什么"、什么时候该被注意 | 文档标签 |
| **Value** | token 的实际内容，用于加权求和 | 文档正文 |

### 公式

$$
\operatorname{Attention}(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

---

## 2. 直观演示：注意力如何产生抽象概念（cell 64–81）

用 fasttext 预训练词向量（300 维）+ 位置编码做实验：

- 输入 `"cows and huts and fields"`，想表达"乡村"这个概念
- 手写**非向量化**的 `simple_attention`，逐 token 两两算分再加权求和
- 画注意力热力图，观察 token 之间的相互影响
- **验证**：把混合后的平均向量与 `city` / `village` 算余弦相似度

**结论**：结果更接近 `village` 而不是 `city` —— 说明注意力把底层词义混合出了更高层的抽象概念。

> 这一段的价值在于"看见"注意力在做什么，而不是直接背公式。

---

## 3. 向量化实现（cell 82–91）

实际工程写法，把上面的循环换成矩阵运算：

```python
'''
q, k, v - [..., seq_len_q, d_k], [..., seq_len_k, d_k],
                  [..., seq_len_k, d_v].
'''
k = x @ w1_key.T          # 投影出 Key
q = x @ w1_query.T        # 投影出 Query
v = x @ w1_value.T        # 投影出 Value

attn_logits = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))  # [..., seq_len_q, seq_len_k]
attention   = F.softmax(attn_logits, dim=-1)                    # 沿"被注意的 token"维归一化
values      = attention @ v                                     # 加权求和
```

- 在自注意力中，`seq_len_q = seq_len_k`，在交叉注意力中才不相等
- `dim`, `d_k`, `d_v` 可以不相等

---

## 4. 封装 + Mask（cell 92–94）

```python
def scaled_dot_product_attention(q, k, v, mask=None):
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(q.size()[-1])
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)   # 关键一步
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention
```

Mask 的原理：把不该看的位置 logit 压到极小值，softmax 后权重 ≈ 0。

- **Padding mask**：屏蔽补齐出来的无意义 token（这里的 `attention_mask`）
- **Causal mask**：`torch.tril(torch.ones(max_len, max_len))` 下三角矩阵，每个位置**只能看左边（含自己）**，用于 GPT 类语言模型和时序模型，让模型学"因果"关系

---

## 5. 多头注意力（cell 98–104）

**动机**：同一 token 可以有多种关系 —— 一个头看语法结构，另一个头看语义关联。

做法：生成多组 Q/K/V，各自跑一遍同样的注意力流程，最后合并。

### `MultiheadAttention` 模块的关键步骤

1. `qkv_proj = nn.Linear(input_dim, 3*emb_dim)` —— **一次投影**同时得到 Q/K/V（比三个 Linear 高效）
2. `reshape(batch, seq, num_heads, 3*head_dim)` → `permute(0, 2, 1, 3)` 得到 `[Batch, Head, SeqLen, head_dim]`
3. `qkv.chunk(3, dim=-1)` 拆出 q、k、v
4. 调用 `scaled_dot_product_attention`；`mask is None` 时默认使用 causal mask
5. `permute` 回来 + `reshape` 合并多头 → `o_proj` 输出投影，融合各头结论

形状变换是最容易出错的地方，记牢：

```
[Batch, Seq, emb_dim]
  → Linear → [Batch, Seq, 3*emb_dim]
  → reshape+permute → [Batch, Head, Seq, head_dim]
  → chunk → q, k, v 各 [Batch, Head, Seq, head_dim]
  → attention @ v → [Batch, Head, Seq, head_dim]
  → permute+reshape → [Batch, Seq, emb_dim]
  → o_proj → [Batch, Seq, emb_dim]
```

约束：`emb_dim % num_heads == 0`（否则抛断言错误）。

---

## 6. 全节流程总结

```
QKV 投影 → 缩放点积打分 → 掩码 → softmax → 加权求和 → 多头并行 → 输出投影
```

最终产物：可复用的 `MultiheadAttention` 模块，供第 4 节的 Transformer Block 直接调用。

---

## 7. 复习自测

- [ ] 为什么 QK^T 要除以 √d_k？（防止点积随维度增大而过大，导致 softmax 梯度消失）
- [ ] softmax 为什么沿 `dim=-1`？（在"被注意的所有 token"这一维上归一化，使权重和为 1）
- [ ] padding mask 和 causal mask 的区别与适用场景？
- [ ] 多头注意力为什么比单头强？计算量会变成 h 倍吗？（不会，通常 head_dim = emb_dim / h）
- [ ] `qkv_proj` 用一个 Linear 和用三个 Linear 的区别？
