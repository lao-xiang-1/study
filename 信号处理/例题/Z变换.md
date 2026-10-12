---
sr-due: 2026-10-13
sr-interval: 1
sr-ease: 230
---
#review 

## 题1

**求 $x(n)$ 的 z 变换和收敛域：**

$$x(n) = \frac{1}{n}, \quad n \ge 1$$

> 知识点参考：[[Z变换]]

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

> $X(z)$ **不是有理分式**，其奇点是 $z=1$ 处的**对数支点**，因此不能用"极点在 ROC 之外"来判定 ROC。

---

## 题2

**求函数的 z 逆变换：**

$$X(z) = \frac{1 - 2z^{-1}}{1 - \frac14 z^{-1}}, \qquad |z| < \frac14$$

### 分析

$X(z)$ 的极点为 $z = \dfrac14$，收敛域 $|z| < \dfrac14$ 位于**极点内侧**，因此 $x(n)$ 是**左边序列（反因果序列）**，只在 $n \le 0$ 非零。

分子分母同乘 $z$，化成有理式：

$$X(z) = \frac{z - 2}{z - \frac14}$$

### 求解（部分分式展开法）

**1. 拆分**

$$X(z) = \frac{z}{z - \frac14} - \frac{2}{z - \frac14}$$

**2. 对 $|z| < \frac14$ 使用反因果变换对**

$$\frac{z}{z-a} \;\longleftrightarrow\; -a^n u(-n-1), \qquad (|z| < |a|)$$

- 第一项：
  $$\frac{z}{z - \frac14} \;\longleftrightarrow\; -\left(\tfrac14\right)^n u(-n-1)$$
- 第二项（提出 $z^{-1}$ 后用**时移性质** $z^{-1}F(z) \leftrightarrow f(n-1)$）：
  $$\frac{2}{z - \frac14} = 2z^{-1}\cdot\frac{z}{z-\frac14} \;\longleftrightarrow\; -2\left(\tfrac14\right)^{n-1}u(-n) = -8\left(\tfrac14\right)^n u(-n)$$
  > 其中 $\left(\tfrac14\right)^{n-1} = 4\left(\tfrac14\right)^n$。

**3. 合并两项**

$$x(n) = -\left(\tfrac14\right)^n u(-n-1) + 8\left(\tfrac14\right)^n u(-n)$$

利用 $u(-n) = u(-n-1) + \delta(n)$：

$$x(n) = \left(\tfrac14\right)^n\big[7\,u(-n-1) + 8\,\delta(n)\big]$$

### 答案

$$\boxed{\,x(n) = 8\,\delta(n) + 7\left(\frac14\right)^n u(-n-1)\,}$$

展开成具体值：

$$x(n) = \begin{cases} 8, & n = 0 \\[2pt] 7\cdot 4^{-n}\ \big(= 28\cdot 4^{-n-1}\big), & n \le -1 \\[2pt] 0, & n \ge 1 \end{cases}$$

### 验证

$$X(z) = 8 + 7\sum_{m=1}^{\infty}4^m z^m = 8 + \frac{28z}{1-4z} = \frac{8-4z}{1-4z} = \frac{z-2}{z-\frac14}\ \checkmark$$

（收敛条件 $|4z| < 1 \Leftrightarrow |z| < \frac14$，与题目 ROC 一致。）
