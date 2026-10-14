**Z 变换（Z-Transform）** 是分析**离散时间信号与系统**的核心数学工具，地位相当于连续时间系统中的**拉普拉斯变换**。它把离散序列 $x[n]$ 映射为复变量 $z$ 的函数 $X(z)$，从而把**差分方程**化为**代数方程**，把**时域卷积**化为 **z 域乘积**。

## 一、定义

### 1. 双边 Z 变换

对序列 $x[n]$（$n$ 取遍所有整数），其**双边（Bilateral）Z 变换**定义为：

$$X(z) = \mathcal{Z}\{x[n]\} = \sum_{n=-\infty}^{\infty} x[n]\, z^{-n}$$

其中 $z$ 为**复变量**，$z^{-n} = (e^{sT})^{-n}$。只有当该幂级数收敛时，$X(z)$ 才有意义（收敛范围即**收敛域 ROC**，见第三节）。

### 2. 单边 Z 变换

若只关心 $n \ge 0$ 的部分，定义为**单边（Unilateral）Z 变换**：

$$X(z) = \mathcal{Z}\{x[n]\} = \sum_{n=0}^{\infty} x[n]\, z^{-n}$$

> **单边 vs. 双边**：单边变换默认 $n<0$ 时 $x[n]=0$，常用于**因果序列**与**求解带初始条件的差分方程**；双边变换能处理非因果序列，收敛域讨论更完整。工程中（如 DSP）默认用单边，但理论分析必须交代收敛域。

### 3. 逆 Z 变换

$$x[n] = \mathcal{Z}^{-1}\{X(z)\} = \frac{1}{2\pi j} \oint_C X(z)\, z^{\,n-1}\, dz$$

其中 $C$ 是 z 平面上**位于收敛域内**、绕原点逆时针一周的闭合围线。

> 该式是围线积分，直接计算困难，实际求逆多用**部分分式展开**、**留数法**或**幂级数展开**（见第六节）。

---

## 二、Z 变换的来源：与三大变换的关系

### 1. 从采样信号到 Z 变换

理想采样信号为

$$x_s(t) = \sum_{n=-\infty}^{\infty} x(nT)\,\delta(t-nT)$$

对其作拉普拉斯变换：

$$X_s(s) = \sum_{n=-\infty}^{\infty} x(nT)\, e^{-snT}$$

令 $z = e^{sT}$（$T$ 为采样周期），即得

$$X(z) = \sum_{n=-\infty}^{\infty} x(nT)\, z^{-n} = \mathcal{Z}\{x[n]\}\big|_{x[n]=x(nT)}$$

> **结论**：Z 变换本质上是**采样信号拉普拉斯变换的变量代换** $z = e^{sT}$，因此它继承了两者的联系。

### 2. s 平面到 z 平面的映射

由 $z = e^{sT} = e^{(\sigma + j\omega)T}$，得 $|z| = e^{\sigma T}$，$\angle z = \omega T$：

| s 平面 | z 平面 |
|:---|:---|
| 虚轴 $\sigma = 0$ | 单位圆 $\lvert z \rvert = 1$ |
| 左半平面 $\sigma < 0$ | 单位圆**内** $\lvert z \rvert < 1$ |
| 右半平面 $\sigma > 0$ | 单位圆**外** $\lvert z \rvert > 1$ |

> 由于 $e^{sT}$ 以 $j\omega_s = j\frac{2\pi}{T}$ 为周期，**s 平面沿虚轴以 $\omega_s$ 为周期重复映射到同一个 z 平面**——这正是离散信号频谱呈周期性的根源（频率混叠 [[11. 取样]]）。

### 3. 与 DTFT、DFT 的关系

- 当 $z$ 取**单位圆上**的值 $z = e^{j\omega}$（即 $\sigma=0$，$\omega = \Omega T$）时，Z 变换退化为**离散时间傅里叶变换（DTFT）**：
  $$X(e^{j\omega}) = X(z)\big|_{z=e^{j\omega}}$$
  前提是**收敛域包含单位圆**。
- 在单位圆上再对 $\omega$ 等间隔取样，即得**离散傅里叶变换（DFT）**，是可计算的频域工具（FFT）。

$$
\boxed{\text{DFT} \subset \text{DTFT} = \text{Z 变换在单位圆上的“切片”}}
$$

---

## 三、收敛域（ROC, Region of Convergence）

### 1. 定义

使幂级数 $\sum x[n] z^{-n}$ **绝对收敛**的 $z$ 的集合：

$$\text{ROC} = \left\{ z : \sum_{n=-\infty}^{\infty} \big| x[n]\, z^{-n} \big| < \infty \right\}$$

### 2. 三类序列的收敛域

| 序列类型 | 特点 | 收敛域（ROC） |
|:---|:---|:---|
| **有限长序列** | 仅在有限个 $n$ 上非零 | 整个 z 平面（可能挖去 $z=0$ 或 $z=\infty$） |
| **右边序列** | $n < N_1$ 时 $x[n]=0$ | $\lvert z \rvert > r_1$（某圆**外**部） |
| **左边序列** | $n > N_2$ 时 $x[n]=0$ | $\lvert z \rvert < r_2$（某圆**内**部） |
| **双边序列** | $-\infty < n < +\infty$ | 环形区域 $r_1 < \lvert z \rvert < r_2$ |

### 3. 重要结论

- ROC 是 z 平面上的**环形区域**，其边界**由极点位置决定**，且 **ROC 内不含任何极点**。
- ROC 在 $z$ 平面上以原点为中心，是**环状/圆盘**形状（由 $\lvert z \rvert$ 决定）。
- **同一个 $X(z)$ 表达式，配合不同的 ROC，对应不同的序列**。因此讨论 Z 变换时必须**同时给出 $X(z)$ 与 ROC**。

> **因果性**：因果序列（$n<0$ 时 $x[n]=0$）的 ROC 是**最外层极点之外**的区域，且包含 $z=\infty$。

---

## 四、常用 Z 变换对

设 $u[n]$ 为单位阶跃序列，$\lvert a \rvert$ 为常数，$T$ 为采样周期。

| 时域 $x[n]$ | z 域 $X(z)$ | 收敛域 |
|:---|:---|:---|
| $\delta[n]$ | $1$ | 全平面 |
| $u[n]$ | $\dfrac{1}{1-z^{-1}} = \dfrac{z}{z-1}$ | $\lvert z \rvert > 1$ |
| $a^n u[n]$ | $\dfrac{1}{1-az^{-1}} = \dfrac{z}{z-a}$ | $\lvert z \rvert > \lvert a \rvert$ |
| $-a^n u[-n-1]$ | $\dfrac{z}{z-a}$ | $\lvert z \rvert < \lvert a \rvert$ |
| $n\,u[n]$ | $\dfrac{z}{(z-1)^2}$ | $\lvert z \rvert > 1$ |
| $n\,a^n u[n]$ | $\dfrac{az}{(z-a)^2}$ | $\lvert z \rvert > \lvert a \rvert$ |
| $e^{-an} u[n]$ | $\dfrac{z}{z-e^{-a}}$ | $\lvert z \rvert > e^{-a}$ |
| $\cos(\omega_0 n)\,u[n]$ | $\dfrac{z(z-\cos\omega_0)}{z^2 - 2z\cos\omega_0 + 1}$ | $\lvert z \rvert > 1$ |
| $\sin(\omega_0 n)\,u[n]$ | $\dfrac{z\sin\omega_0}{z^2 - 2z\cos\omega_0 + 1}$ | $\lvert z \rvert > 1$ |

> 注意第 3、4 行的 $X(z)$ 表达式相同，但 **ROC 不同** → 一个对应**因果**序列 $a^n u[n]$，一个对应**反因果**序列 $-a^n u[-n-1]$。这正是“必须给出 ROC”的典型例子。

---

## 五、Z 变换的主要性质

设 $\mathcal{Z}\{x[n]\} = X(z)$，$\text{ROC} = R_x$；$\mathcal{Z}\{y[n]\} = Y(z)$，$\text{ROC} = R_y$。

### 1. 线性

$$\mathcal{Z}\{a\,x[n] + b\,y[n]\} = a X(z) + b Y(z), \quad \text{ROC} \supseteq R_x \cap R_y$$

### 2. 时移（位移）

- **双边**：
  $$\mathcal{Z}\{x[n-n_0]\} = z^{-n_0} X(z)$$
- **单边**（含初值项，求解差分方程时关键）：
  $$\mathcal{Z}\{x[n-1]\} = z^{-1}X(z) + x[-1]$$
  $$\mathcal{Z}\{x[n-2]\} = z^{-2}X(z) + z^{-1}x[-1] + x[-2]$$

> 单边时移多出的 $x[-1], x[-2], \dots$ 项即**初始条件**，这是用 Z 变换求**全响应**（零输入 + 零状态）的来源。

### 3. z 域尺度变换（乘指数序列）

$$\mathcal{Z}\{a^n x[n]\} = X\!\left(\frac{z}{a}\right), \quad \text{ROC} = \lvert a \rvert R_x$$

### 4. 时域反转

$$\mathcal{Z}\{x[-n]\} = X\!\left(\frac{1}{z}\right), \quad \text{ROC} = \frac{1}{R_x}$$

### 5. z 域微分（乘 $n$）

$$\mathcal{Z}\{n\,x[n]\} = -z \frac{d X(z)}{dz}$$

### 6. 卷积定理（时域卷积）

$$\mathcal{Z}\{x[n] * y[n]\} = X(z)\, Y(z), \quad \text{ROC} \supseteq R_x \cap R_y$$

其中序列卷积定义为 $x[n] * y[n] = \sum_{k=-\infty}^{\infty} x[k]\, y[n-k]$。

> 这是求解**线性时不变（LTI）系统零状态响应**的核心：时域卷积 ⇔ z 域乘积，与拉氏变换的卷积定理完全对应（[[../复变函数与积分变换/4. 拉普拉斯变换的性质]]）。

### 7. 初值定理（因果序列）

若 $x[n]$ 为因果序列，则

$$x[0] = \lim_{z \to \infty} X(z)$$

### 8. 终值定理

若 $(z-1)X(z)$ 的**极点全部位于单位圆内**（即 $x[n]$ 收敛），则

$$x[\infty] = \lim_{n \to \infty} x[n] = \lim_{z \to 1} (z-1) X(z)$$

> **注意**：若 $(z-1)X(z)$ 在单位圆上或圆外有极点，终值不存在，不能使用此定理。

---

## 六、逆 Z 变换的三种方法

### 1. 幂级数展开法（长除法）

将 $X(z)$ 展开为 $z^{-1}$ 的幂级数 $\sum x[n] z^{-n}$，**$z^{-n}$ 的系数即为 $x[n]$**。

- 适用于 $X(z)$ 为有理分式、只需前若干项或找规律的情形。

### 2. 部分分式展开法

将 $\dfrac{X(z)}{z}$ 展开为部分分式（先除 $z$ 便于查表），再逐项对照常用变换对求逆。

> 关键技巧：对 $X(z)$ 除以 $z$ 后再展开，可直接用 $\dfrac{z}{z-a} \leftrightarrow a^n u[n]$ 等标准对。

### 3. 留数法（围线积分）

$$x[n] = \frac{1}{2\pi j} \oint_C X(z) z^{\,n-1}\, dz = \sum_{k} \text{Res}\big[ X(z) z^{\,n-1},\ z = p_k \big]$$

即对**围线 $C$ 内所有极点**求留数之和。

- 一阶极点 $p_k$：$\text{Res} = \lim_{z \to p_k} (z-p_k)\, X(z) z^{n-1}$
- $m$ 阶极点：$\text{Res} = \dfrac{1}{(m-1)!} \lim_{z \to p_k} \dfrac{d^{m-1}}{dz^{m-1}}\big[ (z-p_k)^m X(z) z^{n-1} \big]$

---

## 七、应用：差分方程与系统函数

### 1. 用 Z 变换求解差分方程

对线性常系数差分方程**两边取 Z 变换**（利用时移性质代入初始条件），得到关于 $Y(z)$ 的**代数方程**，解出 $Y(z)$ 后作**逆变换**即得 $y[n]$。

$$
\text{差分方程} \xrightarrow{\ \mathcal{Z}\ } \text{代数方程} \xrightarrow{\ \text{解出 } Y(z)\ } \xrightarrow{\ \mathcal{Z}^{-1}\ } y[n]
$$

> 全响应 = **零输入响应**（由初始条件 $x[-1], x[-2], \dots$ 产生）+ **零状态响应**（由输入 $X(z)$ 产生）。

### 2. 系统函数 $H(z)$

对 LTI 系统，定义**系统函数（传递函数）**

$$H(z) = \frac{Y(z)}{X(z)}\bigg|_{\text{零初始条件}} = \mathcal{Z}\{h[n]\}$$

其中 $h[n]$ 为**单位脉冲响应**。$Y(z) = H(z) X(z)$ 即为卷积定理的直接体现。

### 3. 因果性与稳定性判据

| 性质 | 条件 |
|:---|:---|
| **因果（Causal）** | $H(z)$ 的 ROC 在**最外层极点之外**，且包含 $z=\infty$ |
| **稳定（Stable）** | ROC **包含单位圆** $\lvert z \rvert = 1$ |
| **因果且稳定** | $H(z)$ 的**全部极点位于单位圆内** $\lvert p_k \rvert < 1$ |

> **因果稳定判据**是离散系统设计的核心结论，与连续系统中“极点位于 s 左半平面”一一对应（经 $z = e^{sT}$ 映射）。

---

## 相关

- [[../复变函数与积分变换/3. 拉普拉斯变换概念]] —— Z 变换的连续对应物，$z = e^{sT}$ 的联系
- [[../复变函数与积分变换/4. 拉普拉斯变换的性质]] —— 性质体系的对照
- [[11. 取样]] —— 采样与频率混叠，Z 变换的来源
- [[1.1 信号种类]] —— 单位冲激、阶跃等基本序列
