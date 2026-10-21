
## §3 泰勒级数

在上一节中，我们已经知道一个幂级数的和函数在它的收敛圆的内部是一个解析函数。现在我们来研究与此相反的问题，就是：任何一个解析函数是否能用幂级数来表达？这个问题不但具有理论意义，而且很有实用价值。

设函数  $f(z)$ 在区域 D 内解析，而  $\left|\zeta - z_{0}\right| = r$

为D内以 $z_{0}$为中心的任何一个圆周，它与它的内部

全含于 D，把它记作 K，又设 z 为 K 内任一点（图 4.

2).于是按照柯西积分公式，有



$$  f(z)=\frac{1}{2\pi\mathrm{i}}\oint_{k}\frac{f(\zeta)}{\zeta-z}\mathrm{d}\zeta,   \tag*{（4.3.1）}$$

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//538fab39-5e96-4ca7-852c-075ad1bca35c/markdown_1/imgs/img_in_image_box_816_672_1112_931.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-10-17T11%3A20%3A19Z%2F-1%2F%2F00f873a0b929bbce5ccdabbf4c93f915cd45bf6c0c49f41179ce62d32951c503" alt="Image" width="24%" /></div>


其中 K 取正方向. 由于积分变量  $\zeta$ 取在圆周 K 上，点 z 在 K 的内部，所以  $\left|\frac{z-z_{0}}{\zeta-z_{0}}\right|<1$. 根据 S2 中的例 4，就有

 $$ \begin{align*}\frac{1}{\zeta-z}=&\frac{1}{(\zeta-z_{0})-(z-z_{0})}=\frac{1}{\zeta-z_{0}}\frac{1}{1-\frac{z-z_{0}}{\zeta-z_{0}}}\\=&\frac{1}{\zeta-z_{0}}\bigg[1+\left(\frac{z-z_{0}}{\zeta-z_{0}}\right)+\left(\frac{z-z_{0}}{\zeta-z_{0}}\right)^{2}+\cdots+\left(\frac{z-z_{0}}{\zeta-z_{0}}\right)^{n}+\cdots\bigg]\\=&\sum_{n=0}^{\infty}\frac{1}{(\zeta-z_{0})^{n+1}}(z-z_{0})^{n}.\end{align*} $$ 

以此代入（4.3.1），并把它写成

 $$ f(z)=\sum_{n=0}^{N-1}\left[\frac{1}{2\pi i}\oint_{K}\frac{f(\zeta)\mathrm{d}\zeta}{(\zeta-z_{0})^{n+1}}\right](z-z_{0})^{n}+ $$ 

