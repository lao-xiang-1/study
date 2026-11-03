---
viewed: 2026-11-02
---

>只介绍基本的方法，不考虑工程实现。

#flashcards
## 数学基础
?
### Q / K / V 的语义

| 向量        | 含义                       |
| --------- | ------------------------ |
| **Query** | 这个 token 在"找什么"          |
| **Key**   | 这个 token "提供什么"、什么时候该被注意 |
| **Value** | token 的实际内容，用于加权求和       |

### 公式

$$
\operatorname{Attention}(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

- `Q.shape`：`(seq_len, dim)`
- `K.shape`：`(seq_len, dim)`
- `V.shape`：`(seq_len, seq_len)`

### scores
$$\operatorname{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)$$
- `shape`：`(seq_len, seq_len)`
- 表示每个 token 和其他 token 的相关程度。

### 伪代码

```python
def self_attention(x):
    # x: (seq_len, dim) —— 序列中每个 token 的表示
    Q = x @ W_q            # 每个 token 生成 Query：我在"找什么"
    K = x @ W_k            # 每个 token 生成 Key：  我"提供什么"
    V = x @ W_v            # 每个 token 生成 Value：我的实际内容

    scores  = Q @ K.T / sqrt(d_k)         # (seq_len, seq_len) 两两相关度
    weights = softmax(scores, axis=-1)    # 每个 token 对所有 token 的注意力权重，每行和为 1
    out     = weights @ V                 # 按权重加权求和 → 上下文相关的表示
    return out                            # (seq_len, dim)
```

- `Q @ K.T`：用点积衡量每个 token 与其他 token 的"匹配程度"
- `/ sqrt(d_k)`：缩放，防止点积随维度增大而过大，导致 softmax 梯度消失
- `softmax(axis=-1)`：沿"被注意的所有 token"这一维归一化，使权重非负且和为 1
- `weights @ V`：按注意力权重对所有 token 的 Value 做加权求和
<!--SR:!2026-11-06,3,250-->

---

## 多头注意力
假设每个 token 有 512维的数据。我们取

公式对比：
$$\text{单头}: \text{out} = \underbrace{\text{softmax}\!\left(\frac{XW_QW_K^\top X^\top}{\sqrt{d}}\right)}_{\text{1 个 } n\times n} XW_V W_O$$

$$\text{多头}: \text{out} = \sum_{i=1}^{h} \underbrace{\text{softmax}\!\left(\frac{XW_Q^i (W_K^i)^\top X^\top}{\sqrt{d_h}}\right)}_{\text{h 个不同的 } n\times n} XW_V^i W_O^i$$
- 如果没有softmax的话，两者就是等价的。


## 整体架构

### 三种主流变体

| 变体 | 代表 | 注意力 | 典型任务 |
| :--- | :--- | :--- | :--- |
| Encoder-only | BERT | 双向（无 mask） | 理解：分类、抽取 |
| Encoder-Decoder | 原版 Transformer、T5 | 编码器双向 + 解码器因果 | 序列到序列：翻译、摘要 |
| **Decoder-only** | GPT、LLaMA、Qwen | **因果（causal mask）** | **生成：自回归预测下一个 token** |

> 当前 LLM 几乎全是 decoder-only：结构最统一、易扩展，天然贴合「预测下一个 token」这一目标。

### 数据流总览

### Transformer Block

每一层都是同一个结构，现代实现普遍用 **Pre-Norm**（归一化放在子层**之前**）：

```
x ──┬──────────────────────────────────►(+)──┬──────────────────────────►(+)
    │                                        │
    └─► Norm ─► Masked Self-Attention ───────┘
                                             └─► Norm ─► Feed-Forward ─────┘
```

```python
def block(x):                                    # x: [B, S, d]
    # ── 子层 1：Masked Self-Attention + 残差 ──
    h = norm(x)                                  # [B, S, d]
    q, k, v = h @ W_q, h @ W_k, h @ W_v          # 各 [B, S, d]
    q, k, v = split_heads(q, k, v)               # 各 [B, h, S, d_k]
    scores = q @ k.transpose(-1, -2) / sqrt(d_k) # [B, h, S, S]
    scores = scores.masked_fill(causal, -inf)    # [B, h, S, S]  因果掩码
    out = softmax(scores, -1) @ v                # [B, h, S, d_k]
    x = x + merge_heads(out) @ W_o               # [B, S, d]

    # ── 子层 2：Feed-Forward + 残差 ──
    x = x + feed_forward(norm(x))                # [B, S, d]  内部 d → d_ff → d
    return x                                     # [B, S, d]
```

- **Masked Self-Attention**：让每个 token 只从**左侧**上下文取信息（因果 mask，见下）
- **Feed-Forward（FFN）**：逐位置的两层 MLP，负责"加工"信息；通常先把维度放大若干倍再压回（现代多用 SwiGLU）
- **残差连接**：`x + Sublayer(Norm(x))`，保证梯度通畅、可堆叠很深
- **Pre-Norm**：比 Post-Norm（`Norm(x + Sublayer(x))`）更易训练，是现代默认

### 因果掩码（Causal Mask）

decoder-only 的核心约束：**位置 $i$ 只能看到 $\le i$ 的位置**，不能"偷看"未来。

$$
\text{scores}_{ij} =
\begin{cases}
q_i \cdot k_j & j \le i \\
-\infty & j > i
\end{cases}
$$

即对注意力分数矩阵的**上三角（不含对角线）**置 $-\infty$，softmax 后这些位置权重为 0。

```python
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
scores = scores.masked_fill(mask, float("-inf"))
```

这保证了**训练时的并行**（一次前向同时算出所有位置的预测）与**推理时的自回归**（逐 token 生成）在数学上等价。
