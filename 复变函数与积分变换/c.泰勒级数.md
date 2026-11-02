---
sr-due: 2026-11-10
sr-interval: 11
sr-ease: 270
---
#math 

>将任何一个解析函数用 幂级数来表达

## 定理
设  $f(z)$ 在区域 D 内解析， $z_{0}$ 为 D 内的一点，d 为  $z_{0}$ 到 D 的边界上各点的最短距离，那么当  $|z-z_{0}|<d$ 时（人话：**收敛域的一个圆内**），
$$ f(z)=\sum_{n=0}^{\infty}c_{n}(z-z_{0})^{n} $$
成立，其中  $c_{n}=\frac{1}{n!}f^{(n)}(z_{0}),n=0,1,2,\cdots$

> 证明详见 [泰勒级数定理的证明](attachment/泰勒级数定理的证明.md)。

应当指出，如果  $f(z)$ 在  $z_{0}$ 解析，那么使  $\underbrace{f(z)\text{在}z_{0}\text{的泰勒展开式成立的圆域}}_{}$ 的半径 R 就等于从  $z_{0}$ 到  $f(z)$ 的距  $z_{0}$ 最近一个奇点  $\alpha$ 之间的距离，即  $R=|\alpha-z_{0}|$。这是因为  $f(z)$ 在收敛圆内解析，故奇点  $\alpha$ 不可能在收敛圆内。又因为奇点  $\alpha$ 不可能在收敛圆外，不然收敛半径还可以扩大，因此奇点  $\alpha$ 只能在收敛圆上。

## 常见的展开式

$$ \mathrm{e}^{z}=1+z+\frac{z^{2}}{2!}+\frac{z^{3}}{3!}+\cdots+\frac{z^{n}}{n!}+\cdots\quad(|z|<\infty) $$

$$ \cos z=1-\frac{z^{2}}{2!}+\frac{z^{4}}{4!}-\cdots+(-1)^{n}\frac{z^{2n}}{(2n)!}+\cdots\quad(|z|<\infty) $$

$$ \sin z=z-\frac{z^{3}}{3!}+\frac{z^{5}}{5!}-\cdots+(-1)^{n}\frac{z^{2n+1}}{(2n+1)!}+\cdots\quad(|z|<\infty) $$

