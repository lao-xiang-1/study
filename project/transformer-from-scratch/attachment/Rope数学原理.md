# Rope
旋转位置编码

### 传统位置编码的局限性
* **绝对位置编码**：直接给输入向量加上一个位置向量（如正弦/余弦编码），简单粗暴，但模型很难直接从 $q^\top k$ 的结果中提取出“这两个词相隔多远”的**相对距离**信息。
* **相对位置编码**：直接在计算注意力矩阵时加上相对距离的偏置，效果好，但计算复杂度高，且难以被目前的硬件加速（如 FlashAttention）完美支持。

---

## 基础知识

### 二维旋转公式

在二维空间中，位置 $m$ 的旋转矩阵为：

$$\textbf{R}_m = \begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{bmatrix}, \quad \textbf{R}_m^\top = \textbf{R}_{-m}$$

变换公式：
$$
\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} x\cos m\theta-y\sin m\theta & x\sin m\theta+y\cos m\theta \end{bmatrix}
$$
性质：
$$
\textbf{R}_m \textbf{R}_n =\textbf{R}_{m+n}
$$
很好理解，向量乘以$R_n$后 即旋转$m\theta$度，再乘以$R_m$，加起来旋转了$(m+n)\theta$度。

---

## 原理
在 Transformer 的自注意力（Self-Attention）机制中，计算两个词的相关性依赖于查询向量 $q_m$ 和键向量 $k_n$ 的**内积**：$q_m^\top k_n=\langle q_m, k_n \rangle$

一个好的位置编码，应当能保证 **注意力分数只依赖于相对位置** $m-n$，即：
$$\langle \tilde{\textbf{q}}_m, \tilde{\textbf{k}}_n \rangle = \langle \textbf{R}_m \textbf{q}, \textbf{R}_n \textbf{k} \rangle = \langle \textbf{q}, \textbf{R}_{m-n} \textbf{k} \rangle$$
### 二维情形：
$$
q = \begin{pmatrix} q_1\\ q_2 \end{pmatrix}, \quad k = \begin{pmatrix} k_1\\ k_2 \end{pmatrix}
$$
$$
\textbf R_m \textbf q = \begin{pmatrix} q_1\cos m\theta-q_2\sin m\theta \\ q_1\sin m\theta+q_2\cos m\theta \end{pmatrix},
\quad \textbf R_n \textbf k = \begin{pmatrix} k_1\cos n\theta-k_2\sin n\theta \\ k_1\sin n\theta+k_2\cos n\theta \end{pmatrix}
$$
$$\langle \textbf{R}_m \textbf{q}, \textbf{R}_n \textbf{k} \rangle = (\textbf{R}_m \textbf{q})^\top (\textbf{R}_n \textbf{k}) = \textbf{q}^\top \textbf{R}_m^\top \textbf{R}_n \textbf{k} = \textbf{q}^\top \textbf{R}_{n-m} \textbf{k} = \langle \textbf{q}, \textbf{R}_{n-m} \textbf{k} \rangle$$
### 拓展到高维：

现实中，大模型的隐藏层维度（Hidden Size）通常很大，比如 4096 维，而不是简单的 2 维。
RoPE 的做法是：**两两分组**。
它把 4096 维的向量切分成 2048 个二维向量。每一对二维向量使用不同的旋转频率 $\theta_i$（随着维度增加，旋转速度呈指数级衰减，通常使用 $\theta_i = 10000^{-2i/d}$）。

这样，整个 $q$ 向量相当于乘上了一个巨大的**分块对角矩阵**：
$$
\begin{pmatrix}
\cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots \\
\sin m\theta_1 & \cos m\theta_1 & 0 & 0 & \cdots \\
0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots \\
0 & 0 & \sin m\theta_2 & \cos m\theta_2 & \cdots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{pmatrix}
\begin{pmatrix}
q_1 \\
q_2 \\
q_3 \\
q_4 \\
\vdots
\end{pmatrix}
$$
这里的低维位置加上的低频信号，高维为高频。


