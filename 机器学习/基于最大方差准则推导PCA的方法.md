---
sr-due: 2026-07-08
sr-interval: 1
sr-ease: 230
---
#review 

主成分分析（Principal Component Analysis, PCA）是一种常用的降维算法。其核心思想是将高维数据投影到低维空间中，同时尽可能多地保留原始数据的特征。

基于**最大方差准则**（Maximum Variance Criterion）的推导是 PCA 最经典的推导方法。其基本思想是：**投影后的数据在新的坐标轴（主成分）上的方差越大，说明该方向上保留的信息量越多。**

以下是基于最大方差准则推导 PCA 的完整过程：

### 1. 数据准备与去中心化

假设我们有 $N$ 个样本数据，每个样本有 $D$ 个特征：
$$X = \{x_1, x_2, \dots, x_N\}, \quad x_i \in \mathbb{R}^D$$

为了简化计算，首先对数据进行**去中心化（中心化）**处理，使得数据的均值为 0。
计算样本均值：
$$\mu = \frac{1}{N} \sum_{i=1}^N x_i$$

将每个样本减去均值，得到中心化后的数据（为了叙述方便，后面用 $x_i$ 直接表示中心化后的样本）：
$$x_i \leftarrow x_i - \mu$$

此时，这组数据的**协方差矩阵**（Covariance Matrix）$\Sigma$ 可以表示为：
$$\Sigma = \frac{1}{N} \sum_{i=1}^N x_i x_i^T$$
其中，$\Sigma$ 是一个 $D \times D$ 的实对称矩阵。

### 2. 投影与方差表示
>这一段是证明 变换后的方差 可用协方差矩阵表示

假设我们要将数据投影到一维空间。设投影方向的单位向量为 $w \in \mathbb{R}^D$，满足：
$$w^T w = 1$$

对于任意一个样本点 $x_i$，它在方向 $w$ 上的投影大小为**标量**：
$$z_i = w^T x_i$$

*由于原始数据已经过中心化，投影后数据的均值依然为 0*：
$$\bar{z} = \frac{1}{N} \sum_{i=1}^N w^T x_i = w^T \left( \frac{1}{N} \sum_{i=1}^N x_i \right) = 0$$

现在，我们计算投影后数据的**方差**（Variance）。方差越大，代表信息保留得越完整：
$$D(w) = \frac{1}{N} \sum_{i=1}^N (z_i - \bar{z})^2 = \frac{1}{N} \sum_{i=1}^N (w^T x_i)^2$$

由于 $(w^T x_i)^2 = (w^T x_i)(w^T x_i)^T = w^T x_i x_i^T w$，我们可以将方差写成矩阵形式：
$$D(w) = \frac{1}{N} \sum_{i=1}^N w^T x_i x_i^T w = w^T \left( \frac{1}{N} \sum_{i=1}^N x_i x_i^T \right) w$$

注意到括号中的部分正是协方差矩阵 $\Sigma$，因此方差可以简化为：
$$D(w) = w^T \Sigma w$$

### 3. 构建优化问题

基于最大方差准则，我们的目标是找到一个投影方向 $w$，使得投影后的方差 $D(w)$ 最大，同时满足 $w$ 是单位向量的约束条件。

这可以转化为如下的有约束最优化问题：
$$\max_{w} \quad w^T \Sigma w$$
$$\text{s.t.} \quad w^T w = 1$$

### 4. 求解优化问题（拉格朗日乘子法）
?
>证明 投影向量$w$ 就是协方差矩阵 $\Sigma$ 的 特征向量

引入拉格朗日乘子 $\lambda$，构建拉格朗日函数：
$$L(w, \lambda) = w^T \Sigma w - \lambda (w^T w - 1)$$

对 $w$ 求偏导数，并令其等于 0：
$$\frac{\partial L}{\partial w} = 2 \Sigma w - 2 \lambda w = 0$$

化简后得到：
$$\Sigma w = \lambda w$$

这正是线性代数中经典的**特征值与特征向量**定义式！
* $w$ 是协方差矩阵 $\Sigma$ 的特征向量。
* $\lambda$ 是对应的特征值。

### 5. 确定第一主成分

将 $\Sigma w = \lambda w$ 代回我们想要最大化的方差公式中：
$$D(w) = w^T \Sigma w = w^T (\lambda w) = \lambda (w^T w) = \lambda$$

这表明，**投影后的方差大小正好等于协方差矩阵的特征值 $\lambda$**。

为了使方差 $D(w)$ 最大化，我们应该选择**最大**的特征值 $\lambda_1$，其对应的特征向量 $w_1$ 即为**第一主成分（First Principal Component）**。

---

### 6. 推广到多维（后续主成分）

如果我们希望将数据降到 $k$ 维（$k > 1$），我们需要寻找 $k$ 个正交的投影方向 $w_1, w_2, \dots, w_k$。

对于第二主成分 $w_2$，它除了需要满足自身模长为 1（$w_2^T w_2 = 1$）外，还必须与第一主成分 $w_1$ 正交，即：
$$w_2^T w_1 = 0$$

此时构建的拉格朗日函数为：
$$L(w_2, \lambda, \phi) = w_2^T \Sigma w_2 - \lambda (w_2^T w_2 - 1) - \phi (w_2^T w_1)$$

对 $w_2$ 求导并令其为 0：
$$2 \Sigma w_2 - 2 \lambda w_2 - \phi w_1 = 0$$

两边同时左乘 $w_1^T$：
$$2 w_1^T \Sigma w_2 - 2 \lambda w_1^T w_2 - \phi w_1^T w_1 = 0$$

因为 $\Sigma$ 是对称矩阵，且 $\Sigma w_1 = \lambda_1 w_1$，所以：
$$w_1^T \Sigma w_2 = (w_2^T \Sigma w_1)^T = (w_2^T \lambda_1 w_1)^T = \lambda_1 w_2^T w_1 = 0$$
又因为 $w_1^T w_2 = 0$ 且 $w_1^T w_1 = 1$，代入上式可得：
$$0 - 0 - \phi = 0 \implies \phi = 0$$

因此，方程简化为：
$$\Sigma w_2 = \lambda w_2$$

这说明 $w_2$ 同样是 $\Sigma$ 的特征向量。为了使方差 $w_2^T \Sigma w_2$ 最大且与 $w_1$ 不同，我们应当选择**第二大**特征值对应的特征向量作为第二主成分。

以此类推，前 $k$ 个主成分即为协方差矩阵 $\Sigma$ 前 $k$ 个最大特征值所对应的特征向量。

---

### 总结

基于最大方差准则的 PCA 推导可以总结为以下四个步骤：
1. **去中心化**：使数据均值为 0。
2. **计算协方差矩阵**：$\Sigma = \frac{1}{N} X^T X$。
3. **特征值分解**：求解 $\Sigma w = \lambda w$，计算其特征值和特征向量。
4. **投影降维**：选择前 $k$ 个最大特征值对应的特征向量组成投影矩阵 $W = [w_1, w_2, \dots, w_k]$，将数据投影到低维空间 $Y = XW$。