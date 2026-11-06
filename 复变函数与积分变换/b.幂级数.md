
#mathf
## 定理一（阿贝尔（Abel）定理）
?
>级数收不收敛**只取决于 $|z|$ 的大小**，与辐角无关。

如果级数 $\sum_{n=0}c_{n}z^{n}$在 $z=z_{0}(\neq0)$收敛，那么对满足 $|z|<|z_{0}|$的z，级数必绝对收敛。如果在 $z=z_{0}$级数发散，那么对满足 $|z|>|z_{0}|$的z，级数必发散。


## 2. 收敛圆与收敛半径

利用阿贝尔定理，可以定出幂级数的收敛范围。对一个幂级数来说，它的收敛情况不外乎下述三种：

i）对所有的正实数 $z$ 都是收敛的.这时，根据阿贝尔定理可知级数在复平面内处处绝对收敛.

ii) 对所有的正实数除 z=0 外都是发散的.这时,级数在复平面内除原点外处处发散.

iii) 既存在使级数收敛的正实数，也存在使级数发散的正实数。级数在 $|z|=\alpha$ 收敛，在 $|z|=\beta$ 发散→ 收敛半径就夹在 $\alpha$ 与 $\beta$ 之间，$\alpha$ 往外扩、$\beta$ 往内缩，最终逼近那个分界圆 $C_R$。
那么就称这个 $R$ 为**收敛半径**

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//81c57dab-3df9-4c57-911f-8bebdc657387/markdown_1/imgs/img_in_image_box_771_1194_1109_1520.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-10-17T11%3A20%3A18Z%2F-1%2F%2Fd497d653525f3c6f5af4226690b66647a135472fa786b03ac315fce6ced6d8a2" alt="Image" width="28%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图4.1</div> </div>
- 对 形如(4.2.2)的幂级数 来说，它的收敛范围是以 z=a 为中心的圆域。此时收敛半径是相对于 $(z-a)$ 而言的。
- 在收敛圆的圆周上是收敛还是发散，不能作出一般的结论，要对具体级数进行具体分析。
<!--SR:!2026-11-09,3,250-->

---

## 3. 收敛半径的求法

#### 定理二（比值法） 如果 $\lim_{n\to\infty}\left|\frac{c_{n+1}}{c_n}\right|=\lambda\neq0$，那么收敛半径 $R=\frac{1}{\lambda}$.

#### 定理三（根值法） 如果 $\lim_{n\to\infty}\sqrt[n]{|c_n|}=\mu\neq0$，那么收敛半径 $R=\frac{1}{\mu}$.

## 4. 幂级数的运算和性质

像实变幂级数一样，复变幂级数也能进行有理运算。具体说来，设
$$ f(z)=\sum_{n=0}^{\infty}a_{n}z^{n},\quad R=r_{1},\quad g(z)=\sum_{n=0}^{\infty}b_{n}z^{n},\quad R=r_{2}, $$ 
那么在以原点为中心, $r_{1},r_{2}$中较小的一个为半径的圆内（即两者都收敛），这两个幂级数可以像多项式那样进行相加、相减、相乘，所得到的幂级数的和函数分别就是 $f(z)$与 $g(z)$的和、差与积。即：
$$ f(z)\pm g(z)=\sum_{n=0}^{\infty}a_{n} z^{n}\pm\sum_{n=0}^{\infty}b_{n} z^{n}=\sum_{n=0}^{\infty}(a_{n}\pm b_{n})z^{n},\quad|z|<R, $$  $$ \begin{aligned}f(z)g(z)=&\left(\sum_{n=0}^{\infty}a_{n} z^{n}\right)\left(\sum_{n=0}^{\infty}b_{n} z^{n}\right)\\ =&\sum_{n=0}^{\infty}(a_{n} b_{0}+a_{n-1}b_{1}+a_{n-2}b_{2}+\cdots+a_{0}b_{n})z^{n},\quad\mid z\mid<R.\end{aligned} $$ 
- 这里 收敛半径$R=\min\{r_{1},r_{2}\}$. 
- 所得到的幂级数的收敛半径大于或等于 $r_{1}$与 $r_{2}$中较小的一个
- 对于乘积形式，后面 $z^n$ 的系数 $(a_{n} b_{0}+a_{n-1}b_{1}+a_{n-2}b_{2}+\cdots+a_{0}b_{n})$ 的**直观理解**：在 $f(x)$ 中，系数 $a_n$ 对应 $z^n$；在 $g(x)$ 中，系数 $b_n$ 对应 $z^n$。也就是说系数的下标对应 $z$ 的次数，而 $z^n$ 可以由 $z^n * z^0,\; z^{n-1} * z^1,\; z^{n-2} * z^2,\; \cdots$ 计算得到，其系数也就是$(a_{n} b_{0}+a_{n-1}b_{1}+a_{n-2}b_{2}+\cdots+a_{0}b_{n})$

## ~~定理四~~ 
>抽象

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
