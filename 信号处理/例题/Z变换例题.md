---
sr-due: 2026-11-13
sr-interval: 18
sr-ease: 230
---
#review 

## 题1

**求 $x(n)$ 的 z 变换和收敛域：**

$$x(n) = \frac{1}{n}, \quad n \ge 1$$

> 知识点参考：[[Z变换例题]]

### 前置知识点

- **双边 Z 变换定义**：
  $$X(z) = \mathcal{Z}\{x(n)\} = \sum_{n=-\infty}^{\infty} x(n)\, z^{-n}$$
- **常用幂级数**（$|w| < 1$）：
  $$\sum_{n=1}^{\infty} \frac{w^{\,n}}{n} = -\ln(1-w)$$
- **因果序列（causal / right-sided）**：$n<0$ 时 $x(n)=0$。本题序列只在 $n\ge 1$ 非零，属因果序列。

### 求解

**1. 写出 Z 变换定义式**

$x(n)$ 只在 $n \ge 1$ 非零，代入定义（$n \le 0$ 的项为零）：

$$X(z) = \sum_{n=-\infty}^{\infty} x(n)\, z^{-n} = \sum_{n=1}^{\infty} \frac{1}{n}\, z^{-n}$$

**2. 方法一：套用已知幂级数**

令 $w = z^{-1}$，直接套用上面的幂级数：

$$X(z) = \sum_{n=1}^{\infty} \frac{(z^{-1})^{\,n}}{n} = -\ln\!\left(1 - z^{-1}\right) = -\ln\!\left(\frac{z-1}{z}\right) = \ln\!\left(\frac{z}{z-1}\right)$$

### 收敛域（ROC）

幂级数 $\sum \frac{w^n}{n}$ 在 $|w| < 1$ 时**绝对收敛**，即要求
$$|z^{-1}| < 1 \;\Longleftrightarrow\; |z| > 1$$

边界 $|z| = 1$ 上，$\sum \frac{1}{n}$ 为调和级数、**发散**，故 ROC 为开区域 $|z| > 1$。

### 答案

$$\boxed{\,X(z) = \sum_{n=1}^{\infty} \frac{1}{n}\,z^{-n} = -\ln\!\left(1 - z^{-1}\right) = \ln\frac{z}{z-1}\,,\qquad \text{ROC}:\ |z| > 1\,}$$

---

## 题2

**求函数的 z 逆变换：**

$$X(z) = \frac{1 - 2z^{-1}}{1 - \frac14 z^{-1}}, \qquad |z| < \frac14$$

### 分析

$X(z)$ 的极点为 $z = \dfrac14$，收敛域 $|z| < \dfrac14$ 位于**极点内侧**，因此 $x(n)$ 是**左边序列（反因果序列）**，只在 $n \le 0$ 非零。

- 当 $n\to+\infty$（右边/因果方向）时，$z^{-n}\to 0$ 需要 **$|z|$ 大** → 级数在大 $|z|$ 处收敛 → ROC 在极点**外侧**，向外扩到 $\infty$。
- 当 $n\to-\infty$（左边/反因果方向）时，$z^{-n}=z^{|n|}\to 0$ 需要 **$|z|$ 小** → 级数在小 $|z|$ 处收敛 → ROC 在极点**内侧**，向内缩到 $0$。

分子分母同乘 $z$，化成有理式：

$$X(z) = \frac{z - 2}{z - \frac14}$$

### 求解（部分分式展开法）

**1. 拆分**

$$X(z) = 8 - 7\frac{z}{z - \frac14}$$

**2. 对 $|z| < \frac14$ 使用反因果变换对**

$$\frac{z}{z-a} \;\longleftrightarrow\; -a^n u(-n-1), \qquad (|z| < |a|)$$

### 答案
$$\boxed{\,x(n) = 8\,\delta(n) + 7\left(\frac14\right)^n u(-n-1)\,}$$
