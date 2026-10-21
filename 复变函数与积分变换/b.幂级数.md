---
sr-due: 2026-10-20
sr-interval: 3
sr-ease: 250
---
#math 

### §2 幂级数

### 1. 幂级数概念

设 $\left\{f_{n}(z)\right\}(n=1,2,\cdots)$为一复变函数序列，其中各项在区域D内有定义.表达式
$$ \sum_{n=1}^{\infty}f_{n}(z)=f_{1}(z)+f_{2}(z)+\cdots+f_{n}(z)+\cdots $$ 
称为 **复变函数项级数**，记作 $\sum_{n=1}^{\infty}f_{n}(z)$.这级数的最前面n项的和
$$ s_{n}(z)=f_{1}(z)+f_{2}(z)+\cdots+f_{n}(z) $$ 
称为这级数的 **部分和**.

如果对于 D 内的某一点  $z_{0}$，极限
$$ \lim_{n\to\infty}s_{n}(z_{0})=s(z_{0}) $$ 
存在，那么我们称复变函数项级数(4.2.1)在 $z_{0}$  **收敛**，而 $s(z_{0})$称为它的 **和**。如果级数在D内处处收敛，那么它的和一定是z的一个函数 $s(z)$：
$$ s\left(z\right)=f_{1}\left(z\right)+f_{2}\left(z\right)+\cdots+f_{n}\left(z\right)+\cdots. $$ 
 $s(z)$称为级数  $\sum_{n=1}^{\infty}f_{n}(z)$ 的和函数.

当  $f_{n}(z)=c_{n-1}(z-a)^{n-1}$ 或  $f_{n}(z)=c_{n-1}z^{n-1}$ 时，就得到函数项级数的特殊情形.

$$  \sum_{n=0}^{\infty}c_{n}(z-a)^{n}=c_{0}+c_{1}(z-a)+c_{2}(z-a)^{2}+\cdots+c_{n}(z-a)^{n}+\cdots   \tag*{（4.2.2）}$$

或
$$  \sum_{n=0}^{\infty}c_{n}z^{n}=c_{0}+c_{1}z+c_{2}z^{2}+\cdots+c_{n}z^{n}+\cdots.   \tag*{（4.2.3）}$$

这种级数称为幂级数.

如果令  $z-a=\zeta$，那么 (4.2.2) 成为  $\sum_{n=0}^{\infty}c_{n}\zeta^{n}$，这是 (4.2.3) 的形式。为了方便，今后常就 (4.2.3) 来讨论。

同高等数学中的实变幂级数一样，复变幂级数也有所谓幂级数的收敛定理，即阿贝尔定理.

#### 定理一（阿贝尔（Abel）定理）
>级数收不收敛**只取决于 $|z|$ 的大小**，与辐角无关。

如果级数 $\sum_{n=0}c_{n}z^{n}$在 $z=z_{0}(\neq0)$收敛，那么对满足 $|z|<|z_{0}|$的z，级数必绝对收敛。如果在 $z=z_{0}$级数发散，那么对满足 $|z|>|z_{0}|$的z，级数必发散。


## 2. 收敛圆与收敛半径

利用阿贝尔定理，可以定出幂级数的收敛范围。对一个幂级数来说，它的收敛情况不外乎下述三种：

i）对所有的正实数 $z$ 都是收敛的.这时，根据阿贝尔定理可知级数在复平面内处处绝对收敛.

ii) 对所有的正实数除 z=0 外都是发散的.这时,级数在复平面内除原点外处处发散.

iii) 既存在使级数收敛的正实数，也存在使级数发散的正实数。级数在 $|z|=\alpha$ 收敛，在 $|z|=\beta$ 发散→ 收敛半径就夹在 $\alpha$ 与 $\beta$ 之间，$\alpha$ 往外扩、$\beta$ 往内缩，最终逼近那个分界圆 $C_R$。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//81c57dab-3df9-4c57-911f-8bebdc657387/markdown_1/imgs/img_in_image_box_771_1194_1109_1520.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-10-17T11%3A20%3A18Z%2F-1%2F%2Fd497d653525f3c6f5af4226690b66647a135472fa786b03ac315fce6ced6d8a2" alt="Image" width="28%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图4.1</div> </div>
对幂级数(4.2.2)来说，它的收敛范围是以 z=a 为中心的圆域。在收敛圆的圆周上是收敛还是发散，不能作出一般的结论，要对具体级数进行具体分析。

### 例 1 求幂级数

 $$ \sum_{n=0}^{\infty}z^{n}=1+z+z^{2}+\cdots+z^{n}+\cdots $$ 

的收敛范围与和函数.

[解] 级数的部分和为
$$ s_{n}=1+z+z^{2}+\cdots+z^{n-1}=\frac{1-z^{n}}{1-z}\quad(z\neq1). $$ 
当  $|z|<1$ 时，由于  $\lim_{n\to\infty}z^n=0$ ，从而有  $\lim_{n\to\infty}s_n=\frac{1}{1-z}$，即  $|z|<1$ 时，级数  $\sum_{n=0}^{\infty}z^n$ 收敛，和函数为  $\frac{1}{1-z}$；
当  $|z|\geqslant1$ 时，由于  $n\to\infty$ 时，级数的一般项  $z^n$ 不趋于零 $^{①}$，故级数发散。
由阿贝尔定理知级数的收敛范围为一单位圆域  $|z|<1$，在此圆域内，级数不仅收敛，而且绝对收敛。收敛半径为 1，并有
$$ \frac{1}{1-z}=1+z+z^{2}+\cdots+z^{n}+\cdots. $$ 

## 3. 收敛半径的求法

关于幂级数(4.2.3)的收敛半径的求法，我们有

#### 定理二（比值法） 如果 $\lim_{n\to\infty}\left|\frac{c_{n+1}}{c_n}\right|=\lambda\neq0$，那么收敛半径 $R=\frac{1}{\lambda}$.

我们必须注意，定理中的极限是假定存在的而且不为零。
如果  $\lambda=0$，那么对任何 z，级数  $\sum_{n=0}^{\infty}|c_{n}||z|^{n}$ 收敛，从而级数  $\sum_{n=0}^{\infty}c_{n}z^{n}$ 在复平面内处处收敛，即  $R=\infty$。
如果  $\lambda=+\infty$，那么对于复平面内除 z=0 以外的一切 z，级数  $\sum_{n=0}^{\infty}|c_{n}|||z|^{n}$ 都不收敛。因此  $\sum_{n=0}^{\infty}c_{n}z^{n}$ 也不能收敛，即 R=0。否则，根据阿贝尔定理将有  $z\neq0$ 使得级数  $\sum_{n=0}^{\infty}|c_{n}|||z|^{n}$ 收敛。

#### 定理三（根值法） 如果 $\lim_{n\to\infty}\sqrt[n]{|c_n|}=\mu\neq0$，那么收敛半径 $R=\frac{1}{\mu}$.

### 例2 求下列幂级数的收敛半径：

1)  $\sum_{n=1}^{\infty}\frac{z^{n}}{n^{3}}$（并讨论在收敛圆周上的情形）；

2)  $\sum_{n=1}^{\infty}\frac{(z-1)^{n}}{n}$ (并讨论 z=0,2 时的情形);

3)  $\sum_{n=0}^{\infty}(\cos in)z^{n}$.


【解】

1) 因为  $\lim_{n\to\infty}\left|\frac{c_{n+1}}{c_n}\right|=\lim_{n\to\infty}\left(\frac{n}{n+1}\right)^3=1$

所以收敛半径 R=1，也就是原级数在圆  $\left|z\right|=1$ 内收敛，在圆外发散。在圆周  $\left|z\right|=1$ 上，级数  $\sum_{n=1}^{\infty}\left|\frac{z^{n}}{n^{3}}\right|=\sum_{n=1}^{\infty}\frac{1}{n^{3}}$ 是收敛的，因为这是一个 p 级数，p=3>1。所以原级数在收敛圆上是处处收敛的。

2)  $\lim_{n\to\infty}\left|\frac{c_{n+1}}{c_n}\right|=\lim_{n\to\infty}\frac{n}{n+1}=1$，即 R=1。用根值审敛法也得同样结果。
>这里的收敛半径是对于 $(z - 1)$ 而言的。

在收敛圆  $\left|z-1\right|=1$ 上，当 z=0 时，原级数成为  $\sum_{n=1}^{\infty}(-1)^{n}\cdot\frac{1}{n}$，它是交错级数，根据莱布尼茨准则，级数收敛；当 z=2 时，原级数成为  $\sum_{n=1}^{\infty}\frac{1}{n}$，它是调和级数，所以发散。这个例子表明，在收敛圆周上既有级数的收敛点，也有级数的发散点。

3) 因为  $c_{n} = \cos in = \mathrm{ch}n = \frac{1}{2} (\mathrm{e}^{n} + \mathrm{e}^{-n})$（关于双曲余弦的定义、与三角函数的关系及常用恒等式，详见[双曲余弦](attachment/双曲余弦.md)），所以
$$ \lim_{n\to\infty}\left|\frac{c_{n+1}}{c_{n}}\right|=\lim_{n\to\infty}\frac{\mathrm{e}^{n+1}+\mathrm{e}^{-n-1}}{\mathrm{e}^{n}+\mathrm{e}^{-n}}=\mathrm{e}, $$
故收敛半径  $R = \frac{1}{e}$.

## 4. 幂级数的运算和性质

像实变幂级数一样，复变幂级数也能进行有理运算。具体说来，设
$$ f(z)=\sum_{n=0}^{\infty}a_{n}z^{n},\quad R=r_{1},\quad g(z)=\sum_{n=0}^{\infty}b_{n}z^{n},\quad R=r_{2}, $$ 
那么在以原点为中心, $r_{1},r_{2}$中较小的一个为半径的圆内（即两者都收敛），这两个幂级数可以像多项式那样进行相加、相减、相乘，所得到的幂级数的和函数分别就是 $f(z)$与 $g(z)$的和、差与积。即：
$$ f(z)\pm g(z)=\sum_{n=0}^{\infty}a_{n} z^{n}\pm\sum_{n=0}^{\infty}b_{n} z^{n}=\sum_{n=0}^{\infty}(a_{n}\pm b_{n})z^{n},\quad|z|<R, $$  $$ \begin{aligned}f(z)g(z)=&\left(\sum_{n=0}^{\infty}a_{n} z^{n}\right)\left(\sum_{n=0}^{\infty}b_{n} z^{n}\right)\\ =&\sum_{n=0}^{\infty}(a_{n} b_{0}+a_{n-1}b_{1}+a_{n-2}b_{2}+\cdots+a_{0}b_{n})z^{n},\quad\mid z\mid<R.\end{aligned} $$ 
- 这里  $R=\min\{r_{1},r_{2}\}$. 
- 所得到的幂级数的收敛半径大于或等于 $r_{1}$与 $r_{2}$中较小的一个
- 对于乘积形式，后面 $z^n$ 的系数 $(a_{n} b_{0}+a_{n-1}b_{1}+a_{n-2}b_{2}+\cdots+a_{0}b_{n})$ 的直观理解：在 $f(x)$ 中，系数 $a_n$ 对应 $z^n$；在 $g(x)$ 中，系数 $b_n$ 对应 $z^n$。也就是说系数的下标对应 $z$ 的次数，而 $z^n$ 可以由 $z^n * z^0,\; z^{n-1} * z^1,\; z^{n-2} * z^2,\; \cdots$ 计算得到，其系数也就是$(a_{n} b_{0}+a_{n-1}b_{1}+a_{n-2}b_{2}+\cdots+a_{0}b_{n})$

为了说明两个幂级数经过运算后所得的幂级数的收敛半径确实可以大于  $r_{1}$ 与  $r_{2}$ 中较小的一个，下面举一个例子.

### 例3 
设有幂级数  $\sum_{n=0}^{\infty}z^{n}$ 与  $\sum_{n=0}^{\infty}\frac{1}{1+a^{n}}z^{n}(0<a<1)$，求  $\sum_{n=0}^{\infty}z^{n}-\sum_{n=0}^{\infty}\frac{1}{1+a^{n}}z^{n}=\sum_{n=0}^{\infty}\frac{a^{n}}{1+a^{n}}z^{n}$ 的收敛半径.

[解] 容易验证， $\sum_{n=0}^{\infty}z^{n}$ 与  $\sum_{n=0}^{\infty}\frac{1}{1+a^{n}}z^{n}$ 的收敛半径都等于1.但级数  $\sum_{n=0}^{\infty}\frac{a^{n}}{1+a^{n}}z^{n}$ 的收敛半径

 $$ \lim_{n\to\infty}\left|\frac{a^{n}}{1+a^{n}}\right|\cdot\frac{a^{n+1}}{1+a^{n+1}}=\lim_{n\to\infty}\frac{1+a^{n+1}}{a(1+a^{n})}=\frac{1}{a}>1. $$ 

这就是说， $\sum_{n=0}^{\infty}\frac{a^{n}}{1+a^{n}}z^{n}$ 自身的收敛圆域大于  $\sum_{n=0}^{\infty}z^{n}$ 与  $\sum_{n=0}^{\infty}\frac{1}{1+a^{n}}z^{n}$ 的公共收敛圆域  $\left|z\right|<1$，但应注意，使等式

 $$ \sum_{n=0}^{\infty}z^{n}-\sum_{n=0}^{\infty}\frac{1}{1+a^{n}}z^{n}=\sum_{n=0}^{\infty}\frac{a^{n}}{1+a^{n}}z^{n} $$ 

成立的收敛圆域仍应为 $\left|z\right|<1$，不能扩大。

更为重要的是所谓代换(复合)运算, 就是: 如果当  $|z| < r$ 时,  $f(z) = \sum_{n=0}^{\infty} a_n z^n$, 又设在  $|z| < R$ 内  $g(z)$ 解析且满足  $|g(z)| < r$, 那么当  $|z| < R$ 时,  $f[g(z)] = \sum_{n=0}^{\infty} a_n[g(z)]^n$. 这个代换运算, 在把函数展开成幂级数时, 有着广泛的应用.

### 例4 
把函数 $\frac{1}{z-b}$表成形如  $\sum_{n=0}^{\infty}c_{n}(z-a)^{n}$ 的幂级数，其中 a 与 b 是不相等的复常数.

[解] 把函数 $\frac{1}{z-b}$写成如下的形式：

 $$ \frac{1}{z-b}\!=\!\frac{1}{(z-a)\!-\!(b-a)}\!=\!-\!\frac{1}{b-a}\!\cdot\!\frac{1}{1\!-\!\frac{z-a}{b-a}}, $$ 由例1知道，当 $\left|\frac{z-a}{b-a}\right|<1$时，有

 $$ \frac{1}{1-\frac{z-a}{b-a}}=1+\left(\frac{z-a}{b-a}\right)+\left(\frac{z-a}{b-a}\right)^{2}+\cdots+\left(\frac{z-a}{b-a}\right)^{n}+\cdots, $$ 

从而得到

 $$ \begin{aligned}\frac{1}{z-b}=&-\frac{1}{b-a}-\frac{1}{(b-a)^{2}}(z-a)-\frac{1}{(b-a)^{3}}(z-a)^{2}-\cdots-\\&\frac{1}{(b-a)^{n+1}}(z-a)^{n}-\cdots.\end{aligned} $$ 

设  $\left|b-a\right|=R$，那么当  $\left|z-a\right|<R$ 时，上式右端的级数收敛，且其和为  $\frac{1}{z-b}$。因为 z=b 时，上式右端的级数发散，故由阿贝尔定理知，当  $\left|z-a\right|>\left|b-a\right|=R$ 时，级数发散，即上式右端的级数的收敛半径为  $R=\left|b-a\right|$。

细察本题的解题步骤，不难看出：首先要把函数作代数变形，使其分母中出现量z-a，因为我们要展成z-a的幂级数。再把它按照展开式为已知的函数 $\frac{1}{1-z}$的形式写成 $\frac{1}{1-g(z)}$，其中 $g(z)=\frac{z-a}{b-a}$。然后把 $\frac{1}{1-z}$展开式中的z换成 $g(z)$。

以后，把函数展成幂级数时，常用例4中的方法，希望读者注意.

复变幂级数也像实变幂级数一样，在其收敛圆内具有下列性质（证明从略）：

### 定理四 
设幂级数  $\sum_{n=0}^{\infty} c_{n}(z-z_{0})^{n}$ 的收敛半径为 R，那么

1) 它的和函数  $f(z)$，即
$$ f(z)=\sum_{n=0}^{\infty}c_{n}(z-a)^{n} $$
是收敛圆 $|z-a|<R$ 内的**解析函数**.

 2) $f(z)$在收敛圆内的导数可将其幂级数逐项求导得到，即
$$ f^{\prime}(z)=\sum_{n=1}^{\infty}n c_{n}(z-a)^{n-1}. $$  
 3) $f(z)$ 在收敛圆内可以逐项积分，即
$$ \int\limits_{C}f(z)\mathrm{d}z=\displaystyle\sum_{n=0}^{\infty}c_{n}\int\limits_{C}(z-a)^{n}\mathrm{d}z,\quad C\in\mid z-a\mid<R,$$
或
$$\int\limits_{a}^{z}f(\zeta)\mathrm{d}\zeta=\displaystyle\sum_{n=0}^{\infty}\frac{c_{n}}{n+1}(z-a)^{n+1}. $$ 