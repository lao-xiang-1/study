# 注意力机制笔记（MiniMind）

> 只关注注意力本身：多头结构 → GQA → KV cache → 两条计算路径。
> 位置编码（RoPE）的数学推导见 [rope_notes.md](./rope_notes.md)，本文只引用不展开。

对应代码：[model/model_minimind.py](../model/model_minimind.py)
- `repeat_kv` — [L142](../model/model_minimind.py#L142)
- `Attention` — [L152](../model/model_minimind.py#L152)
- `MiniMindBlock` — [L354](../model/model_minimind.py#L354)

---

## 1. 注意力在模型中的位置

`MiniMindBlock` 是标准的 **Pre-LN** 结构：归一化放在子层**之前**，残差绕过子层直接相加。

```
x ──┬──────────────────────────────────────►(+)──┬──────────────────────►(+)
    │                                            │
    └─► RMSNorm ─► Attention ─► Dropout ─────────┘
                                                 └─► RMSNorm ─► MLP ─► Dropout ─┘
```

```python
# MiniMindBlock.forward
residual = hidden_states
hidden_states, present_key_value = self.self_attn(
    self.input_layernorm(hidden_states), position_embeddings,
    past_key_value, use_cache, attention_mask)
hidden_states += residual                                    # 注意力残差
hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
```

注意力的职责可以一句话概括：**让每个 token 从上下文里"取"信息**。MLP 则是"加工"信息，只在单个位置上做。两者交替堆叠。

---

## 2. 基础公式与为什么要缩放

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

三个矩阵的角色分工：

| 矩阵 | 角色 | 直觉 |
|---|---|---|
| Q (query) | 我在找什么 | 当前 token 发出的"检索请求" |
| K (key) | 我是什么 | 每个 token 的"索引标签" |
| V (value) | 我提供什么 | 每个 token 的"内容" |

$QK^\top$ 是**相似度打分**，softmax 归一化成权重，再对 V 做加权平均。所以注意力的输出永远是 V 的**凸组合**——这是它的表达能力边界，也是它稳定的原因。

**缩放因子 $\sqrt{d_k}$ 不能省。** 设 $q, k$ 各分量独立、均值 0 方差 1，则

$$\text{Var}\left(\sum_{i=1}^{d_k} q_i k_i\right) = d_k$$

内积的方差随维度线性增长。head_dim=64 时 logits 标准差约 8，softmax 会严重饱和成近似 one-hot：**梯度趋近于 0，训练停滞**。除以 $\sqrt{d_k}$ 把方差拉回 1。

代码里对应 [L201](../model/model_minimind.py#L201)：

```python
scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
```

---

## 3. 多头与 GQA

### 3.1 从单头到多头

把 $d$ 维空间切成 $h$ 个 $d_k = d/h$ 维子空间，每个子空间独立做一次注意力，最后拼接。**参数量与计算量不变**，但模型可以在不同的子空间里关注不同类型的关系（一个头盯语法、一个头盯指代……）。

MiniMind 的配置：`hidden_size=512`，`num_attention_heads=8` → `head_dim=64`。

### 3.2 MHA → MQA → GQA

区别只在 **K、V 用几个头**：

| 变体 | `num_key_value_heads` | KV cache 相对大小 | 说明 |
|---|---|---|---|
| MHA | = 8 | 100% | 每个 Q 头配一组独立 KV |
| **GQA** | **2（MiniMind）** | **25%** | Q 头分组，组内共享 KV |
| MQA | 1 | 12.5% | 所有 Q 头共享一组 KV |

推理时的瓶颈不是算力而是**显存带宽**：每生成一个 token，都要把整个 KV cache 从显存读一遍。GQA 把 KV 头从 8 降到 2，读取量直接砍到 1/4，质量损失却很小——所以现在几乎所有开源模型（Llama 3、Qwen、Mistral）都用 GQA。

### 3.3 维度与参数量（MiniMind 实际值）

```python
self.n_rep  = 8 // 2 = 4          # 每个 KV 头被 4 个 Q 头共享
self.head_dim = 512 // 8 = 64
```

| 投影 | 形状 | 参数量 |
|---|---|---|
| `q_proj` | 512 → 8×64 = 512 | 262,144 |
| `k_proj` | 512 → 2×64 = 128 | 65,536 |
| `v_proj` | 512 → 2×64 = 128 | 65,536 |
| `o_proj` | 8×64 = 512 → 512 | 262,144 |
| 合计 | | **655,360 / 层**（8 层约 5.2M） |

`q_proj` 输入输出都是 512 是巧合（因为 `head_dim = hidden_size / num_heads`），不是通用规律。所有投影 `bias=False`：归一化层已经提供了可学习的偏置，线性层再叠一个收益很小。

---

## 4. forward 逐段拆解

### 4.1 投影 + 拆头

```python
bsz, seq_len, _ = x.shape
xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
xq = xq.view(bsz, seq_len, self.n_local_heads,    self.head_dim)   # [b, s, 8, 64]
xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)   # [b, s, 2, 64]
xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
```

`view` 只是重新解释内存布局，**零开销**。此时 head 维被放到倒数第二维（`[b, s, h, d]`），是为了后面 RoPE 的 `unsqueeze(1)` 广播能对上——cos/sin 是 `[s, d]`，广播成 `[s, 1, d]` 后正好乘到 h 维上。

### 4.2 施加 RoPE

```python
cos, sin = position_embeddings
xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
```

**只旋转 q、k，不旋转 v。** 位置信息只需要进入 $QK^\top$ 内积参与打分；v 携带的是纯内容语义，位置化没有意义。详见 [rope_notes.md](./rope_notes.md)。

### 4.3 KV cache

```python
if past_key_value is not None:
    xk = torch.cat([past_key_value[0], xk], dim=1)   # dim=1 是序列维
    xv = torch.cat([past_key_value[1], xv], dim=1)
past_kv = (xk, xv) if use_cache else None
```

自回归生成时，每步只新增 1 个 token，但注意力需要看到**全部历史**。若不缓存，就得把整段前缀重算一遍 —— 复杂度从 $O(n)$ 退化成 $O(n^2)$。KV cache 就是把历史 K、V 存下来，每步只算新 token 的。

两个关键细节：

1. **拼接发生在 `repeat_kv` 之前**，存的是 **2 个 KV 头**而非 8 个。GQA 的显存收益正是在这里落地的——如果先 repeat 再存，等于白做 GQA。
2. **只缓存 K、V，不缓存 Q**。因为历史 token 的 query 在后续步骤中不再被需要；每个新 token 只用自己这一行 Q 去查询整个历史。

> 缓存布局是 `[b, s, kv_heads, d]`（转置之前），与 `apply_rotary_pos_emb` 的输出布局一致，避免每步做一次转置。

### 4.4 repeat_kv：把 2 个头"撑"成 8 个

```python
def repeat_kv(x, n_rep):
    bs, slen, num_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (x[:, :, :, None, :]
            .expand(bs, slen, num_kv_heads, n_rep, head_dim)
            .reshape(bs, slen, num_kv_heads * n_rep, head_dim))
```

用 `expand` + `reshape` 而非 `repeat_interleave`：先在 head 维后面插一个长度 `n_rep` 的虚拟维，广播展开后再 reshape 合并。结果与 `repeat_interleave(x, dim=2, repeats=n_rep)` 完全等价——**每个 KV 头连续重复 4 次**，于是 Q 头 $[4i, 4i+3]$ 共享第 $i$ 组 KV。

`expand` 本身不复制内存（返回的是 stride=0 的视图），真正的物化发生在 `reshape` 时。所以这一步仍然有拷贝开销，只是省掉了显式的中间张量。

映射关系（8 个 Q 头 ↔ 2 个 KV 头）：

```
Q head:  0  1  2  3 | 4  5  6  7
KV head: 0  0  0  0 | 1  1  1  1
```

### 4.5 转置到 SDPA 布局

```python
xq, xk, xv = (xq.transpose(1, 2),
              repeat_kv(xk, self.n_rep).transpose(1, 2),
              repeat_kv(xv, self.n_rep).transpose(1, 2))
```

`[b, s, h, d]` → `[b, h, s, d]`。PyTorch 的 `scaled_dot_product_attention` 要求 head 维在前（批 × 头 × 序列 × 维度）。

### 4.6 两条计算路径

```python
if self.flash and (seq_len > 1) and (past_key_value is None) \
        and (attention_mask is None or torch.all(attention_mask == 1)):
    output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=..., is_causal=True)
else:
    ...  # 手写路径
```

**走 flash 路径的四个条件**：

| 条件 | 原因 |
|---|---|
| `self.flash` | PyTorch ≥ 2.0 且 config 允许 |
| `seq_len > 1` | 单 token 解码没有并行收益 |
| `past_key_value is None` | 带 cache 时序列长度不齐，causal mask 需特殊处理 |
| mask 全 1 | 有真实 padding mask 时需显式传入，故退回手写 |

实际上这条分支就是 **训练 / prefill 走 flash，逐 token 解码走手写**。`is_causal=True` 让 SDPA 内部自己生成因果 mask，不需要外部构造 $O(n^2)$ 的 mask 矩阵——这正是 FlashAttention 省显存的核心。

### 4.7 手写路径：mask 的 `-seq_len:` 技巧

```python
scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
scores[:, :, :, -seq_len:] += torch.triu(
    torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)
```

`scores` 形状 `[b, h, seq_len, kv_len]`，其中 `kv_len = past_len + seq_len`。因果 mask 只应作用在**右下角的 `seq_len × seq_len` 块**上（新 query 对全部 key）：

```
             kv_len 列
        ┌───────────┬─────────────┐
        │           │             │
seq_len │  (不管)   │  triu(-inf) │   ← 只有这块加 mask
  行    │           │             │
        └───────────┴─────────────┘
                      ↑ -seq_len:
```

- 无 cache 时 `seq_len == kv_len`，`-seq_len:` 覆盖全部列 → 退化成标准因果 mask；
- 解码时 `seq_len == 1`，`triu(1×1, diagonal=1)` 是全零矩阵 → 不加任何 mask，因为新 token 本来就该看到全部历史。

`torch.triu(..., diagonal=1)`：保留主对角线**以上**（即 $j > i$）的部分置 `-inf`，表示"第 $i$ 个 token 不能看第 $j$ 个 token"。softmax 后这些位置权重为 0。

用 `-inf` 而非某个大负数，是因为 softmax 会精确归零。前提是**每行至少有一个可见位置**（$j \le i$ 恒成立，至少对角线可见），否则整行 `-inf` 会 softmax 出 NaN。

### 4.8 padding mask：加性转换

```python
if attention_mask is not None:
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)   # [b,1,1,kv_len]
    extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
    scores = scores + extended_attention_mask
```

- `attention_mask` 是 0/1 形式（1=可见），转成**加性 mask**：可见位置 `(1-1)*-1e9 = 0`（无影响），padding 位置 `(1-0)*-1e9 = -1e9`（压掉）。
- 两个 `unsqueeze` 把 `[b, kv_len]` 广播到 `[b, 1, 1, kv_len]`——对所有头和所有 query 位置生效。
- 这里用 `-1e9` 而不是 `-inf`：padding 整行可能全被 mask 掉，`-inf` 会产出 NaN；`-1e9` 经 softmax 后近似为 0，数值安全。

### 4.9 softmax 与输出

```python
scores = F.softmax(scores.float(), dim=-1).type_as(xq)   # 升 fp32 再降回
scores = self.attn_dropout(scores)
output = scores @ xv

output = output.transpose(1, 2).reshape(bsz, seq_len, -1)   # 拼头
output = self.resid_dropout(self.o_proj(output))
```

- **`scores.float()` 升精度**：bf16 只有 8 位尾数，softmax 里的 `exp` 和除法误差会被放大。softmax 在 fp32 下算完再降回原精度，是标准做法（和 `RMSNorm.forward` 里的 `x.float()` 同一个理由）。
- **`transpose(1,2).reshape(b, s, -1)`**：把 `[b, h, s, d]` 还原成 `[b, s, h*d]`，即多头输出拼接。注意 `transpose` 后内存不连续，`reshape` 会触发一次隐式拷贝。
- **`o_proj`** 把拼接后的多头结果重新混合。没有它，各头的输出只是简单并列，缺少跨头的交互。
- **两个 dropout**：`attn_dropout` 作用在注意力权重上（随机切断 token 间的信息流），`resid_dropout` 作用在子层输出上。MiniMind 默认 `dropout=0.0`，两者都是恒等映射。

---

## 5. 训练 vs 推理：两条路径的行为差异

| | 训练 / prefill | 逐 token 解码 |
|---|---|---|
| `seq_len` | > 1 | 1 |
| `past_key_value` | None | 有 |
| 计算路径 | FlashAttention（`is_causal=True`） | 手写（显式 mask） |
| causal mask | SDPA 内部生成，不落显存 | 显式构造 `seq_len × seq_len` |
| 返回 `past_kv` | 仅当 `use_cache=True` | 是 |
| 瓶颈 | 算力（矩阵乘） | 显存带宽（读 KV cache） |

两条路径的**数学结果应当一致**，差别只在显存占用和访存模式。这也是为什么 flash 分支的四个条件里，只要有任一不满足就退回手写——正确性优先。

---

## 6. 显存与复杂度

**注意力矩阵的 $O(n^2)$**：`scores` 是 `[b, h, n, n]`。以 b=1, h=8, n=2048, bf16 计，单个矩阵就有 $8 \times 2048^2 \times 2\text{B} = 64$ MB。这是长上下文的主要显存障碍，也是 FlashAttention 存在的理由——它通过分块计算，**从不把完整的 $n \times n$ 矩阵写进显存**。

**KV cache 的 $O(n)$**（每 token、每层）：

| | 公式 | MiniMind 实际值 |
|---|---|---|
| GQA（2 KV 头） | $2 \times 2 \times 64$ | 256 元素 / 层 |
| 8 层合计 | | 2048 元素 ≈ **4 KB**（bf16） |
| 若用 MHA（8 头） | $2 \times 8 \times 64$ | 8192 元素 ≈ 16 KB |

4 倍差距，且随上下文线性增长。32K 上下文时，单条序列的 KV cache 约 128 MB。

---

## 7. 要点速查

| 问题 | 答案 |
|---|---|
| 注意力做什么 | 用 Q·K 打分，softmax 归一化，对 V 加权平均 |
| 为什么除以 √d | 内积方差随 head_dim 线性增长，不缩放则 softmax 饱和、梯度消失 |
| 多头的作用 | 在不同子空间关注不同关系，参数量不变 |
| MiniMind 用哪种 | GQA：8 个 Q 头共享 2 个 KV 头（`n_rep=4`） |
| GQA 为什么省显存 | 推理瓶颈是读 KV cache 的带宽，KV 头 8→2 即读取量降到 1/4 |
| KV cache 存什么 | 只存 K、V，不存 Q；且存的是 **repeat 之前**的 2 个头 |
| `repeat_kv` 怎么实现 | `expand` + `reshape`，等价于 `repeat_interleave`，每 KV 头连续重复 4 次 |
| RoPE 作用在哪 | 只旋 q、k，不旋 v；在拆头之后、拼接 cache 之前 |
| mask 的 `-seq_len:` | 只给右下角 `seq_len × seq_len` 块加因果 mask，使其同时适配有/无 cache |
| 因果 mask 用 `-inf` | 精确归零；每行至少有对角线可见，不会 NaN |
| padding mask 用 `-1e9` | 整行可能被全 mask，用有限大负数避免 NaN |
| 为什么要升 fp32 | bf16 尾数只有 8 位，softmax 的 exp/除法误差会被放大 |
| flash 分支何时走 | 训练/prefill；解码时 `seq_len=1` 且带 cache，退回手写 |
| `o_proj` 的作用 | 混合各头输出，提供跨头交互；同时投影回 hidden_size |
| 注意力输出的本质 | V 的凸组合——表达能力的边界，也是稳定性的来源 |
