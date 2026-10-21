---
sr-due: 2026-11-01
sr-interval: 11
sr-ease: 270
---
#math 

### §1 复数项级数

### 1. 复数列的极限
>这可以看成通项的极限，而不是求和

设 $\{\alpha_{n}\}(n=1,2,\cdots)$为一复数列，其中 $\alpha_{n}=a_{n}+ib_{n}$，又设 $\alpha=a+ib$为一确定的复数。如果任意给定 $\varepsilon>0$，相应地能找到一个正数 $N(\varepsilon)$，使 $\left|\alpha_{n}-\alpha\right|<\varepsilon$在 $n>N$时成立，那么 $\alpha$称为复数列 $\{\alpha_{n}\}$当 $n\to\infty$时的极限，记作
$$ \lim_{n\to\infty}\alpha_{n}=\alpha. $$ 
此时也称复数列 $\{\alpha_{n}\}$收敛于 $\alpha$.

#### 定理一 复数列 $\{\alpha_{n}\}(n=1,2,\cdots)$收敛于 $\alpha$的充要条件是
$$ \lim_{n\to\infty}a_{n}=a,\quad\lim_{n\to\infty}b_{n}=b. $$ 
### 2. 级数概念
>级数默认是无穷项的

设 $\{\alpha_{n}\}=\{a_{n}+\mathrm{i}b_{n}\}(n=1,2,\cdots)$为一复数列，表达式
$$ \sum_{n=1}^{\infty}\alpha_{n}=\alpha_{1}+\alpha_{2}+\cdots+\alpha_{n}+\cdots $$
称为无穷级数，其最前面 n 项的和
$$ s_{n}=\alpha_{1}+\alpha_{2}+\cdots+\alpha_{n} $$
称为级数的部分和.

如果部分和数列 $\{s_{n}\}$收敛，那么级数 $\sum_{n=1}^{\infty}\alpha_{n}$称为 **收敛**，并且极限 $\lim_{n\to\infty}s_{n}=s$ 称为级数的 **和**。如果数列 $\{s_{n}\}$不收敛，那么级数 $\sum_{n=1}^{\infty}\alpha_{n}$称为 **发散**。

#### 定理二 级数  $\sum_{n=1}^{\infty} \alpha_{n}$ 收敛的充要条件是 级数  $\sum_{n=1}^{\infty} a_{n}$ 和  $\sum_{n=1}^{\infty} b_{n}$ 都收敛.

定理二将复数项级数的审敛问题转化为实数项级数的审敛问题，而由实数项级数  $\sum_{n=1}^{\infty}a_{n}$ 和  $\sum_{n=1}^{\infty}b_{n}$ 收敛的必要条件
$$ \lim_{n\to\infty}a_{n}=0 和 \lim_{n\to\infty}b_{n}=0, $$ 
立即可得  $\lim_{n\to\infty}\alpha_{n}=0$，从而推出复数项级数  $\sum_{n=1}^{\infty}\alpha_{n}$ 收敛的**必要条件**是  $\lim_{n\to\infty}\alpha_{n}=0$

#### 定理三 如果  $\sum_{n=1}^{\infty} |\alpha_n|$ 收敛，那么  $\sum_{n=1}^{\infty} \alpha_n$ 也收敛，且不等式  $\left|\sum_{n=1}^{\infty} \alpha_n\right| \leqslant \sum_{n=1}^{\infty} |\alpha_n|$ 成立.

如果  $\sum_{n=1}^{\infty}\left|\alpha_{n}\right|$ 收敛，那么称级数  $\sum_{n=1}^{\infty}\alpha_{n}$ 为  **绝对收敛**. 非绝对收敛的收敛级数称为  **条件收敛级数**

顺便指出，由于 $|\alpha_n| = \sqrt{a_{n}^{2}+b_{n}^{2}}\leqslant|a_{n}|+|b_{n}|$，因此
$$ \sum_{k=1}^{n}\sqrt{a_{n}^{2}+b_{n}^{2}}\leqslant\sum_{k=1}^{n}\left|a_{n}\right|+\sum_{k=1}^{n}\left|b_{n}\right|, $$ 
所以当  $\sum_{n=1}^{\infty}a_{n}$ 与  $\sum_{n=1}^{\infty}b_{n}$ 绝对收敛时， $\sum_{n=1}^{\infty}\alpha_{n}$ 也绝对收敛。结合定理三的证明过程，可知  $\sum_{n=1}^{\infty}\alpha_{n}$ 绝对收敛的充要条件是级数  $\sum_{n=1}^{\infty}a_{n}$ 与  $\sum_{n=1}^{\infty}b_{n}$ 绝对收敛。

另外，因为  $\sum_{n=1}^{\infty}|\alpha_{n}|$ 的各项都是非负的实数，所以它的收敛性可用正项级数的 判定法来判定.

### 3. 等比级数（几何级数）

**等比数列**：从第二项起，每一项与前一项之比为同一常数 $q$（公比，$q$ 可为复数），即
$$ a_1,\ a_1q,\ a_1q^{2},\ \cdots,\ a_1q^{\,n-1},\ \cdots $$

**前 $n$ 项和（部分和）**：记 $s_n=a_1+a_1q+\cdots+a_1q^{\,n-1}$，由错位相减 $s_n-q s_n=a_1-a_1q^{n}$ 得
$$ s_n=\frac{a_1(1-q^{n})}{1-q}\qquad (q\ne 1), $$
当 $q=1$ 时 $s_n=n a_1$。此式对任意 $q\ne 1$ 都成立，与级数是否收敛无关。

**等比级数（无穷级数）**：$\displaystyle\sum_{n=0}^{\infty}a_1q^{\,n}$ 的敛散由公比 $q$ 决定（上面的 n 指的是项数，和这里不同）：


- 当 $|q|<1$ 时收敛，且
$$ \sum_{n=0}^{\infty}a_1q^{\,n}=\frac{a_1}{1-q}; $$
- 当 $|q|\ge 1$ 时发散。

#### 常用结论
1. $$ \sum_{n=0}^{\infty}q^{n}=\frac{1}{1-q},\qquad (|q|<1), $$
 2. $$\displaystyle\sum_{n=1}^{\infty}q^{n}=\frac{q}{1-q}\ (|q|<1)$$
### 例题
#### 例 1 下列数列是否收敛？如果收敛，求出其极限.

1)  $\alpha_{n}=\left(1+\frac{1}{n}\right)\mathrm{e}^{\mathrm{i}\frac{\pi}{n}}$; 2)  $\alpha_{n}=n\cos i n$.

[解] 1) 因  $\alpha_{n}=\left(1+\frac{1}{n}\right)\mathrm{e}^{\mathrm{i}\frac{\pi}{n}}=\left(1+\frac{1}{n}\right)\left(\cos\frac{\pi}{n}+\mathrm{i}\sin\frac{\pi}{n}\right)$，故
$$ a_{n}=\left(1+\frac{1}{n}\right)\cos\frac{\pi}{n},\quad b_{n}=\left(1+\frac{1}{n}\right)\sin\frac{\pi}{n}. $$
而
$$ \lim_{n\to\infty}a_n=1,\quad\lim_{n\to\infty}b_n=0 $$
所以数列 $\{\alpha_{n}\}$收敛，且有 $\lim_{n\to\infty}\alpha_{n}=1$

2）由于  $\alpha_{n}=n\cos in=n\mathrm{chn}$（详见[双曲余弦](attachment/双曲余弦.md)），因此，当  $n\to\infty$ 时， $\alpha_{n}\to\infty$。所以  $\{\alpha_{n}\}$ 发散。

#### 例2 下列级数是否收敛？是否绝对收敛？

1)  $\sum_{n=1}^{\infty}\frac{1}{n}\left(1+\frac{\mathrm{i}}{n}\right)$; 2)  $\sum_{n=0}^{\infty}\frac{(8\mathrm{i})^{n}}{n!}$; 3)  $\sum_{n=1}^{\infty}\left[\frac{(-1)^{n}}{n}+\frac{1}{2^{n}}\mathrm{i}\right]$.

[解] 1) 因  $\sum_{n=1}^{\infty}a_{n}=\sum_{n=1}^{\infty}\frac{1}{n}$ 发散， $\sum_{n=1}^{\infty}b_{n}=\sum_{n=1}^{\infty}\frac{1}{n^{2}}$ 收敛. 故原级数发散.

2) 因  $\left|\frac{(8i)^{n}}{n!}\right|=\frac{8^{n}}{n!}$，由正项级数的比值审敛法知  $\sum_{n=1}^{\infty}\frac{8^{n}}{n!}$ 收敛，故原级数收敛，且为绝对收敛.

3）因  $\sum_{n=1}^{\infty}\frac{(-1)^{n}}{n}$ 收敛， $\sum_{n=1}^{\infty}\frac{1}{2^{n}}$ 也收敛，故原级数收敛。但因  $\sum_{n=1}^{\infty}\frac{(-1)^{n}}{n}$ 为条件收敛，所以原级数非绝对收敛。

