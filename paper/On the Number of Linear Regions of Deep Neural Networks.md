---
viewed: 2026-11-20
---

# 论深层神经网络的线性区域数量

## 摘要

我们研究具有分段线性（piecewise linear）激活函数的深度前馈神经网络所计算函数的复杂度，刻画的角度是这些函数所具有的对称性及其线性区域的数量。深度网络能够将每一层输入空间的部分区域依次映射到相同的输出。通过这种方式，深度模型计算出的函数能够对不同输入的复杂模式作出相同的响应。这些函数的复合结构使它们能够随着网络深度以指数级频繁的方式复用计算片段。本文研究此类复合映射的复杂度，并给出关于"深度对于具有分段线性激活函数的神经网络之优势"的新理论结果。特别地，我们的分析并不局限于某一类特定模型；作为一个例子，我们将其应用于整流器（rectifier）网络和 maxout 网络。我们改进了既有工作的复杂度界，并研究了更高层中单元的行为。

关键词：深度学习，神经网络，输入空间划分，整流器，maxout

## 1 引言

具有若干隐藏层的人工神经网络（称为深度神经网络）由于在多种机器学习任务上取得了空前的成功而变得流行（参见，例如 Krizhevsky et al. 2012、Ciresan et al. 2012、Goodfellow et al. 2013、Hinton et al. 2012）。鉴于这些经验证据，深度神经网络正越来越受到青睐，超过浅层网络（即只有单层隐藏单元的网络），并且通常以超过五层的深度实现。然而，目前只有数量有限的文献从理论角度研究深度网络。最近，Delalleau 和 Bengio（2011）表明，为了计算某些多项式族，浅层网络所需的求和-乘积（sum-product）隐藏单元 $^{1}$ 比深层求和-乘积网络多出指数级。我们感兴趣的是将这类分析推广到更常用的神经网络上。

有大量文献讨论人工神经网络的逼近、估计与复杂度（参见，例如 Anthony 和 Bartlett 1999）。一个著名的结果是：具有单个巨大隐藏层的前馈神经网络是 Borel 可测函数的通用逼近器（参见 Hornik et al. 1989、Cybenko 1989）。其他工作研究了深度信念网络对概率分布的通用逼近（Le Roux 和 Bengio 2010、Montúfar 和 Ay 2011），以及它们的逼近性质（Montúfar 2014、Krause et al. 2013）。
<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//87d6c133-c161-42da-b599-f2de86a6cb87/markdown_1/imgs/img_in_image_box_239_159_975_334.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-20T10%3A20%3A15Z%2F-1%2F%2Fd011004686efe7ff82fa26a491c41ae800607edb8922ce10fc56fc0bf62a8325" alt="Image" width="60%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图 1：使用一个具有 20 个隐藏单元的浅层模型（实线）和一个具有两层、每层 10 个单元的深度模型（虚线）进行二分类。右图是左图的局部放大。实心标记表示浅层模型所犯的错误。</div> </div>


然而，这些先前的理论结果并不能简单地适用于近年来取得成功的那些类型的深度神经网络。传统神经网络通常采用具有有界光滑激活函数的隐藏单元，或者布尔隐藏单元。另一方面，近来使用分段线性函数变得越来越普遍，例如整流器激活 $g(a) = \max\{0, a\}$（Glorot et al. 2011、Nair 和 Hinton 2010）或 maxout 激活 $g(a_1, \ldots, a_k) = \max\{a_1, \ldots, a_k\}$（Goodfellow et al. 2013）。具有分段线性单元的深度神经网络在实际中的成功，要求针对该类神经网络进行专门的理论分析。

在这方面，Pascanu et al.（2013）报告了一个关于具有整流器单元的深度前馈网络所计算函数复杂度的理论结果。他们表明，在隐藏层数很多的渐近极限下，尽管使用相同数量的计算单元，深度网络能够将其输入空间划分出的线性响应区域数量比其浅层对应物多出指数级。

在（Pascanu et al. 2013）思想的基础上，我们发展了一个分析具有分段线性激活的深度模型的通用框架。这些模型的中间层能够将其输入的若干片段映射到相同的输出。以这种方式逐层复合所计算出的函数，会随着层数的增加以指数级频繁地复用低层计算。这一关键性质使深度网络能够计算出高度复杂且有结构的函数。我们通过估计两类重要的分段线性网络（具有整流器单元的网络和具有 maxout 单元的网络）所计算函数的线性区域数量，来支撑这一思想。

我们关于深度整流器网络复杂度的结果，相较上述关于整流器网络的先前结果有显著改进，表明即便隐藏层数适中，深度网络相对浅层网络也表现出有利的行为。我们对深度整流器网络和 maxout 网络的分析，可作为研究广泛相关网络（例如卷积网络）的平台。

给定模型可计算函数的线性区域数量是衡量该模型灵活性的一个指标。图 1 给出了一个例子，它比较了相同隐藏单元数量下单层模型与两层模型所学到的决策边界（详见附录 F）。这说明了深度的优势：深度模型更准确地捕捉到所期望的边界，用更多的线性片段去逼近它。

如前所述，深度网络能够通过将大量输入邻域映射到某个中间隐藏层的共同输出，来识别指数级数量的输入邻域。在该中间层激活上所进行的计算被多次复制，在被识别出的每个邻域中各复制一次。这使网络即便在只用相对少量参数定义的情况下，也能计算出看起来非常复杂的函数。

参数的数量是网络可计算函数集合维数的一个上界，参数数量少意味着可计算函数类具有低维数。深度前馈分段线性网络可计算的函数集合尽管维数低，却通过逐层复用与复合特征而达到指数级的复杂度。

## 2 前馈神经网络及其复合性质

在本节中，我们讨论深度前馈网络仅使用相对少量计算单元来重映射其输入空间、从而产生复杂对称性的能力。我们分析的关键观察是：深度模型的每一层都能够将其输入的不同区域映射到共同的输出。这导致一种复合结构，其中更高层上的计算会在所有于给定层产生相同输出的输入区域中被有效地复制。在输入空间上复制计算的能力随着网络层数以指数级增长。在展开这些思想之前，我们先介绍本文其余部分所需的基本定义。在本节末尾，我们给出一个关于深度模型复制能力的直观视角。

### 2.1 定义

**前馈神经网络**是若干层计算单元的复合，它定义一个函数 $F: \mathbb{R}^{n_0} \to \mathbb{R}^{\text{out}}$，形式为

$$  F(\mathbf{x};\theta)=f_{\mathrm{o u t}}\circ g_{L}\circ f_{L}\circ\cdots\circ g_{1}\circ f_{1}(\mathbf{x}),   \tag*{(1)}$$

其中 $f_l$ 是线性预激活函数，$g_l$ 是非线性激活函数。参数 $\theta$ 由每一层 $l \in [L]$ 的输入权重矩阵 $\mathbf{W}_l \in \mathbb{R}^{k \cdot n_l \times n_{l-1}}$ 和偏置向量 $\mathbf{b}_l \in \mathbb{R}^{k \cdot n_l}$ 组成。

第 $l$ 层的输出是一个向量 $\mathbf{x}_l = [\mathbf{x}_{l,1}, \ldots, \mathbf{x}_{l,n_l}]^\top$，由该层中单元 $i \in [n_l]$ 的激活值 $\mathbf{x}_{l,i}$ 组成。它由前一层激活值计算：$\mathbf{x}_l = g_l(f_l(\mathbf{x}_{l-1}))$。给定第 $(l-1)$ 层单元的激活值 $\mathbf{x}_{l-1}$，第 $l$ 层的预激活为

 $$ f_{l}(\mathbf{x}_{l-1})=\mathbf{W}_{l}\mathbf{x}_{l-1}+\mathbf{b}_{l}, $$ 

其中 $f_l = [f_{l,1}, \ldots, f_{l,n_l}]^l$ 是由 $n_l$ 个预激活向量 $f_{l,i} \in \mathbb{R}^k$ 组成的数组，而第 $l$ 层第 $i$ 个单元的激活为

 $$ \mathbf{x}_{l,i}=g_{l,i}(f_{l,i}(\mathbf{x}_{l-1})). $$ 

我们将把 $g_{l} \circ f_{l}$ 简记为 $h_{l}$。当层索引 $l$ 明确时，我们将省略相应的下标。我们关注分段线性激活，并考虑以下两种重要类型。

• 整流器单元：$g_i(f_i) = \max\{0, f_i\}$，其中 $f_i \in \mathbb{R}$ 且 $k = 1$。

• 秩-$k$ maxout 单元：$g_i(f_i) = \max\{f_{i,1}, \ldots, f_{i,k}\}$，其中 $f_i = [f_{i,1}, \ldots, f_{i,k}] \in \mathbb{R}^k$。

网络的结构指的是其单元的排布方式。它由输入维数 $n_{0}$、层数 L 以及每层的单元数（宽度）$n_{l}$ 所确定。

我们将根据线性区域的数量，对不同网络结构、不同参数选择下所计算的函数进行分类。分段线性函数 $F: \mathbb{R}^{n_0} \to \mathbb{R}^m$ 的一个线性区域，是输入空间 $\mathbb{R}^{n_0}$ 的一个极大连通子集，$F$ 在其上是线性的。对于我们考虑的函数，每个线性区域都是满维的，即维数为 $n_0$。

### 2.2 浅层神经网络

整流器单元有两种行为方式：根据其输入，它们要么恒为 0，要么是线性的。这两种行为之间的分界由一个超平面给出，而来自整流器层中所有单元的所有超平面构成一个超平面划分（hyperplane arrangement）。一般地，如果激活函数 $g: \mathbb{R} \to \mathbb{R}$ 在零点具有特殊（即非规则）行为（例如拐点或非线性），那么函数 $\mathbb{R}^{n_0} \to \mathbb{R}^{n_1}$；$\mathbf{x} \mapsto g(\mathbf{W}\mathbf{x} + \mathbf{b})$ 在任何来自超平面 $H_i := \{\mathbf{x} \in \mathbb{R}^{n_0} : \mathbf{W}_{i, :} \mathbf{x} + \mathbf{b}_i = 0\}$（$i \in [n_1]$）的输入处都具有特殊行为。捕捉这一特殊行为的这些超平面同样构成一个超平面划分。

划分中的超平面将输入空间分割成若干区域。形式上，超平面划分 $\{H_1, \ldots, H_{n_1}\}$ 的一个区域是补集 $\mathbb{R}^{n_0} \setminus (\cup_i H_i)$ 的一个连通分支，即由这些超平面所界定的一组点（可能向无穷延伸）。
划分的区域数量可以借助划分的特征函数来表达，如 Zaslavsky（1975）的一个著名结果所示。$\mathbb{R}^{n_0}$ 中 $n_1$ 个超平面的一个划分至多有 $\sum_{j=0}^{n_0} \binom{n_1}{j}$ 个区域。
此外，当且仅当这些超平面处于一般位置（general position）时，才取到这一区域数。这意味着，具有 $n_0$ 个输入和 $n_1$ 个隐藏单元的浅层整流器网络所计算函数的最大线性区域数为 $\sum_{j=0}^{n_0} \binom{n_1}{j}$（参见 Pascanu et al. 2013；命题 5）。

### 2.3 深度神经网络

我们首先更形式化地定义引言中提到的输入邻域识别：

#### 定义 1. 
一个映射 $F$ 识别其输入域的两个邻域 $S$ 和 $T$，是指它将它们映射到其输出域的同一个子集 $F(S) = F(T)$。此时我们也称 $S$ 和 $T$ 被 $F$ 识别。

例如，二维欧氏空间的四个象限是被绝对值函数 $g\colon \mathbb{R}^2 \to \mathbb{R}^2$ 所识别的区域；

$$  g(x_{1},x_{2})=\left[\left|x_{1}\right|,\ \left|x_{2}\right|\right]^{\top}.   \tag*{(2)}$$
？
前馈网络第 $l$ 层对来自第 $(l-1)$ 层的一组激活值所进行的计算，实际上对输入空间中所有导致第 $(l-1)$ 层激活值相同的区域都会执行。
可以选择某一层的输入权重和偏置，使得所计算的函数在那些于输入空间中具有最多原像的前一层激活值上表现得最为有趣，从而在输入空间中将有趣的计算复制许多次，并生成一个整体上看起来复杂的函数。

对于网络参数的任意给定选择，每个隐藏层 $l$ 都在前一层的输出激活值上计算一个函数 $h_l = g_l \circ f_l$。我们考虑函数 $F_l: \mathbb{R}^{n_0} \to \mathbb{R}^{n_l}$；$F_l := h_l \circ \cdots \circ h_1$，它计算第 $l$ 个隐藏层的激活值。我们用 $S_l \subseteq \mathbb{R}^{n_l}$ 表示 $F_l$ 的像，即第 $l$ 层对所有可能输入所能到达的（向量值）激活值集合。给定子集 $R \subseteq S_l$，我们用 $P_l^l$ 表示由 $h_l$ 映射到 $R$ 上的一组子集 $\bar{R}_1, \ldots, \bar{R}_k \subseteq S_{l-1}$；即满足 $h_l(\bar{R}_1) = \cdots = h_l(\bar{R}_k) = R$ 的那些子集。示例参见图 2。

#### 输入空间邻域数量
被映射到共同邻域 $R \subseteq S_l \subseteq \mathbb{R}^{n_l}$ 的、彼此分离的输入空间邻域数量可递归地表示为

$$  \mathcal{N}_{R}^{l}=\sum_{R^{\prime}\in P_{R}^{l}}\mathcal{N}_{R^{\prime}}^{l-1},\qquad\mathcal{N}_{R}^{0}=1,\mathrm{f o r~e a c h~r e g i o n~}R\subseteq\mathbb{R}^{n_{0}}.   \tag*{(3)}$$

例如，$P_R^1$ 是所有互不相交的输入空间邻域的集合，这些邻域经第一层所计算的函数 $h_1: \mathbf{x} \mapsto g(\mathbf{W}\mathbf{x} + \mathbf{b})$ 映射后的像等于 $R \subseteq S_1 \subseteq \mathbb{R}^{n_1}$。

递归公式 (3) 通过沿着一棵以第 $j$ 层输出空间的集合 $R$ 为根的树的各分支移动，来计数被识别的集合的数量（参见图 2 (c)）。基于这些观察，我们可以如下估计线性区域的最大数量。

#### 引理 2. 
由具有分段线性激活的 L 层神经网络所计算函数的最大线性区域数量至少为 $\mathcal{N} = \sum_{R \in P^L} \mathcal{N}_R^{L-1}$，其中 $\mathcal{N}_R^{L-1}$ 由式 (3) 定义，$P^L$ 是最后隐藏层所计算函数的不同线性区域中的一组邻域。

这里，构造具有许多线性区域的函数的思路是：用前 $L-1$ 个隐藏层来识别许多输入空间邻域，将它们全部映射到第 $(L-1)$ 个隐藏层的激活邻域 $P^{L}$，而 $P^{L}$ 中的每一个都属于最后隐藏层的一个不同线性区域。

我们将在第 3 节和第 4 节中给出整流器网络与 maxout 网络的详细分析。<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//be27ce54-3e70-449f-9166-53682c506b0b/markdown_0/imgs/img_in_image_box_277_161_589_294.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-20T10%3A20%3A21Z%2F-1%2F%2F002be241ec571219521167db23cd56fa46b5e34df488fdf1a50cc2b179d9c315" alt="Image" width="25%" /></div>


<div style="text-align: center;"><div style="text-align: center;">(a)</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//be27ce54-3e70-449f-9166-53682c506b0b/markdown_0/imgs/img_in_image_box_278_340_587_505.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-20T10%3A20%3A21Z%2F-1%2F%2Fb3f6aede94796ab73a6a39ef3766e251bd04d022e80981e5093d864c2a685dfe" alt="Image" width="25%" /></div>

<div style="text-align: center;"><div style="text-align: center;">(b)</div> </div>

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//be27ce54-3e70-449f-9166-53682c506b0b/markdown_0/imgs/img_in_image_box_675_169_970_498.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-20T10%3A20%3A21Z%2F-1%2F%2Ff1e813e32bc7c176b02d3c9ebafd42ee1bfe4af9decda82e66dfea6af97e2b3d" alt="Image" width="24%" /></div>

<div style="text-align: center;"><div style="text-align: center;">(c)</div> </div>


<div style="text-align: center;"><div style="text-align: center;">图 2：(a) 二维欧氏空间沿两个坐标轴的空间折叠。(b) 顶层划分（右侧）如何被复制回原始输入空间（左侧）的图示。(c) 深度模型各层之间区域的识别。</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//be27ce54-3e70-449f-9166-53682c506b0b/markdown_0/imgs/img_in_image_box_371_622_852_755.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-20T10%3A20%3A21Z%2F-1%2F%2F47395a6da78d58cafdf05f6a4e91a9fd9bb16eb198c1540f3dc68a7ebd1079d4" alt="Image" width="39%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图 3：二维空间以非平凡方式进行的空间折叠。注意这种折叠如何可能识别出它需要学习的边界中的对称性。</div> </div>


### 2.4 将输入识别视为空间折叠

在本节中，我们以空间折叠（space folding）的角度讨论引理 2 背后的直观。一个识别两个子集 $\mathcal{S}$ 和 $\mathcal{S}'$ 的映射 $F$，可以被看作一个算子，它把其定义域折叠，使得子集 $\mathcal{S}$ 和 $\mathcal{S}'$ 重合并且被映射到同一输出。例如，式 (2) 中的绝对值函数 $g: \mathbb{R}^2 \to \mathbb{R}^2$ 将其定义域折叠两次（沿每个坐标轴各一次），如图 2 (a) 所示。这种折叠识别了二维欧氏空间的四个象限。通过复合这类操作，同类的映射可以再次施加到输出上，以便对第一次折叠进行再次折叠。

深度神经网络的每个隐藏层都可与一个折叠算子相关联。每个隐藏层折叠前一层激活值的空间。进而，深度神经网络从第一层开始，有效地递归折叠其输入空间。这种递归折叠的后果是：在最终折叠空间上计算的任何函数，都将适用于由这一系列折叠所对应的映射识别出的所有被折叠到一起的子集。这意味着在深度模型中，最后一层像空间的任何划分都会被复制到由这一系列折叠所识别的所有输入空间区域中。图 2 (b) 给出了这种复制性质的图示。

空间折叠并不限于沿坐标轴的折叠，也不必保持长度。相反，空间根据输入权重 $\mathbf{W}$、偏置 b 中所编码的方向和位移，以及每个隐藏层所使用的非线性激活函数而被折叠。特别地，这意味着被识别的输入空间区域的大小和方向可能彼此不同。参见图 3。

### 2.5 对扰动的稳定性

我们对深度模型可达复杂度的界（第 3 节和第 4 节）基于对网络权重的合适选择。然而，这并不意味着所指出的复杂度只能在奇异情形下达到。神经网络所计算函数的参数化是连续的。更确切地说，映射 $\psi: \mathbb{R}^N \to C(\mathbb{R}^{n_0}; \mathbb{R}^{n_L}); \theta \mapsto F_\theta$，它将输入权重和偏置 $\theta = \{\mathbf{W}_i, \mathbf{b}_i\}_{i=1}^L$ 映射到网络所计算的连续函数 $F_\theta: \mathbb{R}^{n_0} \to \mathbb{R}^{n_L}$，是连续的。我们的分析考虑函数 $F_\theta$ 的线性区域数量。根据定义，每个线性区域都包含输入空间 $\mathbb{R}^{n_0}$ 的一个开邻域。给定任意具有有限个线性区域的函数 $F_\theta$，存在 $\epsilon > 0$，使得对参数 $\theta$ 的每一次 $\epsilon$-扰动，所得函数 $F_{\theta+\epsilon}$ 所具有的线性区域数至少与 $F_\theta$ 一样多。$F_\theta$ 的线性区域在参数的微小扰动下得以保持，因为它们具有有限的体积。

如果我们在参数空间上定义一个概率密度，那么"网络所表示的函数具有给定数量线性区域"这一事件的概率是多少？由上述讨论可知，获得不少于任何特定参数选择（在有界域内取均匀测度）所导致的区域数的概率是非零的，尽管它可能非常小。这是因为在该特定参数选择周围存在一个体积非零的 epsilon-球，在该球内至少能达到相同数量的线性区域。

作为未来工作，一个有趣的方向是研究参数空间 $\mathbb{R}^N$ 被划分为若干部分、使得所得函数将其输入空间划分为同构的线性区域这一问题，并研究参数空间的这些部分中有多少对应于具有给定数量线性区域的函数。

### 2.6 整流器 MLP 中折叠的经验评估

我们通过实验考察了一个已训练 MLP 的行为，看它是否以上述方式折叠输入空间。首先，我们观察到：追踪该模型中每个隐藏单元的激活值，给出一个分段线性映射 $\mathbb{R}^{n_0} \to \mathbb{R}$（从输入到该单元的激活值）。因此，我们可以通过可视化对应于该映射不同线性片段的不同权重矩阵，来分析每个单元的行为。该映射某一线性片段的权重矩阵，可以通过从一个输入样本出发、追踪每个中间层所使用的线性片段来得到。这种可视化技术是我们理论分析的一个副产品，它类似于 Zeiler 和 Fergus（2013）提出的方法，但出于不同的视角。

在为每个训练样本计算某个中间隐藏单元的激活值之后，我们可以例如考察两个对某个隐藏单元导致相似激活水平的样本。借助这两个样本所对应的该隐藏单元的线性映射，我们扰动其中一个样本，直到它恰好导致相同的激活值。这两个输入于是可以安全地被视为该隐藏单元所识别的两个区域中的点。我们在附录 G 中给出这一可视化技术的细节和示例。我们还展示了被一个深度 MLP 所识别的输入。

## 3 深度整流器网络

在本节中，我们基于第 2 节的一般观察来分析具有整流器单元的深度神经网络。我们改进了 Pascanu et al.（2013）的结果，给出了深度整流器网络所计算函数最大线性区域数的一个更紧的下界。

首先，让我们注意以下上界，它直接来自如下事实：整流器网络的每个线性区域对应于一种隐藏单元处于激活状态的模式：

命题 3. 任何具有总共 N 个隐藏单元的整流器网络所计算函数的最大线性区域数，其上界为 $2^{N}$。

### 3.1 构造的说明

考虑一层具有 $n$ 个整流器、$n_0$ 个输入变量的情形，其中 $n \geq n_0$。我们把整流器单元的集合划分为 $n_0$ 个（互不重叠的）子集，每个子集的基数为 $p = \lfloor \sqrt[n]{n_0} \rfloor$，并忽略其余<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//be27ce54-3e70-449f-9166-53682c506b0b/markdown_2/imgs/img_in_image_box_453_159_773_339.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-20T10%3A20%3A24Z%2F-1%2F%2F4c76a9b92c08856cdb4ec823a39b8d1987d0cffc380b56c4294756ead4e7f317" alt="Image" width="26%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图 4：由整流器之和将实直线折叠为等长线段。</div> </div>


单元。考虑第 $j$ 个子集中的单元。我们可以这样选择它们的输入权重和偏置，使得

 $$ h_{1}(\mathbf{x})=\max\left\{0,\mathbf{w}\mathbf{x}\right\}, $$ 

 $$ h_{2}(\mathbf{x})=\max\left\{0,2\mathbf{w}\mathbf{x}-1\right\}, $$ 

 $$ \begin{array}{c} h_{3}(\mathbf{x})\;=\;\max\left\{0,2\mathbf{w}\mathbf{x}-2\right\},\end{array} $$ 

 $$ h_{p}(\mathbf{x})=\max\left\{0,2\mathbf{w}\mathbf{x}-(p-1)\right\}, $$ 

其中 w 是一个行向量，其第 $j$ 个元素为 1，其余元素为 0。乘积 wx 选出 x 的第 $j$ 个坐标。将这些整流器以交替的符号相加，我们得到如下标量函数：

$$  \tilde{h}_{j}(\mathbf{x})=\left[1,-1,1,\ldots,(-1)^{p-1}\right]\left[h_{1}(\mathbf{x}),h_{2}(\mathbf{x}),h_{3}(\mathbf{x}),\ldots,h_{p}(\mathbf{x})\right]^{\top}.   \tag*{(4)}$$

由于 $\tilde{h}_{j}$ 只作用于第 $j$ 个输入坐标，我们可以重新将它定义为取标量输入，即 x 的第 $j$ 个坐标。该函数具有 p 个线性区域，由以下区间给出：

 $$ \left(-\infty,0\right],\left[0,1\right],\left[1,2\right],\ldots,\left[p-1,\infty\right). $$ 

这些区间中的每一个都有一个子集被 $\tilde{h}_j$ 映射到区间 $(0,1)$ 上，如图 4 所示。函数 $\tilde{h}_j$ 识别第 $j$ 个坐标 $\mathbf{x}_j$ 被限制在区间 $(0,1), (1,2), \ldots, (p-1,p)$ 内的那些输入空间条带。现在考虑全部 $n_0$ 个整流器子集以及函数 $\tilde{h} = [\tilde{h}_1, \tilde{h}_2, \ldots, \tilde{h}_p]^\top$。该函数关于每个第 $j$ 个坐标固定为 $\mathbf{x}_j = 1, \ldots, \mathbf{x}_j = p-1$ 的超平面（图 4 中的竖直线）局部对称，对所有 $j = 1, \ldots, n_0$ 成立。注意由此出现的周期模式。事实上，函数 $\tilde{h}$ 总共识别出由这些超平面界定的 $p^{n_0}$ 个超立方体。

现在注意，$\tilde{h}$ 由 h 与一个线性函数（交替和）复合得到。这个线性函数可以有效地被吸收进下一层的预激活函数中。因此我们可以把 $\tilde{h}$ 视为当前层所计算的函数。更深层所作的计算，作为该整流器层单位超立方体输出的函数，被复制到这 $p^{n_0}$ 个被识别的输入空间超立方体中的每一个上。

### 3.2 形式化结果

我们可以将上述构造推广到具有 $n_0$ 个输入、L 个隐藏层且对所有 $i \in [L]$ 宽度为 $n_i \geq n_0$ 的深度整流器网络的情形。我们得到深度整流器网络最大线性区域数的如下下界：

定理 4. 由具有 $n_0$ 个输入单元、L 个隐藏层且第 $i$ 层有 $n_i \geq n_0$ 个整流器的神经网络所计算函数的最大线性区域数，其下界为

 $$ \left(\prod_{i=1}^{L-1}\left\lfloor\frac{n_{i}}{n_{0}}\right\rfloor^{n_{0}}\right)\sum_{j=0}^{n_{0}}\binom{n_{L}}{j}. $$ 

下面的推论给出了这些界渐近行为的一个表达式。假设 $n_{0}=O(1)$ 且对所有 $i\geq1$ 有 $n_{i}=n$，则具有 $Ln$ 个隐藏单元的单层模型的区域数行为为 $O(L^{n_{0}}n^{n_{0}})$（参见 Pascanu et al. 2013；命题 10）。对于深度模型，定理 4 蕴含：

推论 5. 一个具有 $n_0$ 个输入单元、L 个宽度为 $n \geq n_0$ 的隐藏层的整流器神经网络，可以计算具有 $\Omega\left(\binom{n}{n_0}^{(L-1)n_0}n^{n_0}\right)$ 个线性区域的函数。

于是我们看到，深度模型的线性区域数随 L 指数增长、随 $n$ 多项式增长，这比具有 $nL$ 个隐藏单元的浅层模型要快得多。我们的结果相较 Pascanu et al.（2013）所得到的界 $\Omega\left(\binom{n/n_{0}}{n_{0}}^{L-1}n^{n_{0}}\right)$ 是一个显著的改进。特别地，我们的结果表明，即便 L 和 $n$ 的取值较小，深度整流器模型也能产生比浅层整流器模型多得多的线性区域。此外，采用与 Pascanu et al.（2013）相同的策略，我们的结果可以用每个参数所对应的线性区域数来重新表述。这导致类似的行为，即深度模型比浅层模型指数级地更高效（参见附录 C）。

## 4 深度 Maxout 网络

maxout 网络是一种前馈网络，其各层定义如下：

定义 6. 一个具有 $n$ 个输入单元和 $m$ 个输出单元的秩-$k$ maxout 层，由如下形式的预激活函数定义：$f: \mathbb{R}^n \to \mathbb{R}^{m \cdot k}$；$f(\mathbf{x}) = \mathbf{W}\mathbf{x} + \mathbf{b}$，其输入权重和偏置为 $\mathbf{W} \in \mathbb{R}^{m \cdot k \times n}$、$\mathbf{b} \in \mathbb{R}^{m \cdot k}$，激活为 $g_j(\mathbf{z}) = \max\{\mathbf{z}_{(j-1)k+1}, \ldots, \mathbf{z}_{jk}\}$，对所有 $j \in [m]$ 成立。该层计算一个函数

$$  g\circ f\colon\quad\mathbb{R}^{n}\to\mathbb{R}^{m};\quad\mathbf{x}\mapsto\left(\begin{matrix}{\operatorname*{m a x}\{f_{1}(\mathbf{x}),\ldots,f_{k}(\mathbf{x})\}}\\ {\vdots}\\ {\operatorname*{m a x}\{f_{(m-1)k+1}(\mathbf{x}),\ldots,f_{m k}(\mathbf{x})\}}\\ \end{matrix}\right).   \tag*{(5)}$$

由于两个凸函数的极大值仍是凸函数，maxout 单元和 maxout 层计算的是凸函数。一组函数的极大值称为它们的上包络（upper envelope）。我们可以把每个线性函数 $f_i: \mathbb{R}^n \to \mathbb{R}$ 的图像，看作 $(n + 1)$ 维空间中某个凸集的支撑超平面。特别地，若每个 $f_i, i \in [k]$ 在某个输入邻域内是唯一的最大值 $f_i = \max\{f'_i: i' \in [k]\}$，则上包络 $g_1 \circ f = \max\{f_i: i \in [k]\}$ 的线性区域数恰为 $k$。这表明一个 maxout 单元的最大线性区域数等于它的秩。

maxout 层的线性区域是各个 maxout 单元的线性区域的交集。为了得到该层的线性区域数，我们需要描述每个 maxout 单元线性区域的结构，并研究它们可能的交集。

Voronoi 图可以被提升为线性函数的上包络，因此它们描述了由 maxout 单元生成的输入空间划分。那么，将 $m$ 个各含 $k$ 个区域的 Voronoi 图求交，我们得到多少个区域？一般来说，计算 Voronoi 图的交并不容易。线性区域数的一个平凡上界是 $k^m$，它对应于不同单元的区域之间所有交集彼此都不同的情形。我们将在命题 7 中给出一个更好的界。

现在，为了计算下界，这里只需考虑某些行为良好的特殊情形就足够了。一个简单的例子是用 $k - 1$ 个平行超平面划分输入空间。若 $m \leq n$，我们可以对每个 maxout 单元 $j \in [m]$ 考虑超平面 $H_i = \{\mathbf{x} \in \mathbb{R}^n: \mathbf{x}_j = i\}$（$i = 1, \ldots, k - 1$）的划分。此时区域数为 $k^m$。若 $m > n$，同样的论证给出 $k^n$ 个区域。

命题 7. 一个具有 $n$ 个输入、$m$ 个秩为 $k$ 的输出的单层 maxout 网络，其最大区域数的下界为 $k^{\min\{n,m\}}$，上界为 $\min\{\sum_{j=0}^{n}\binom{k^2m}{j}, k^m\}$。

现在我们来看深度 maxout 模型。注意一个秩-2 的 maxout 层可以由一个单元数两倍的整流器层来模拟。于是，由上一节的结果，一个具有 $L-1$ 个宽度为 $n=n_0$ 的隐藏层的秩-2 maxout 网络可以识别 $2^{n_0(L-1)}$ 个输入空间区域，进而可以计算具有 $2^{n_0(L-1)}2^{n_0}=2^{n_0L}$ 个线性区域的函数。

对于秩-$k$ 的情形，我们注意到一个秩-$k$ maxout 单元可以从其输入域中识别出 $k$ 个锥（cone），其中每个锥是对应于线性函数 $f_i$ 的梯度 $\mathbf{W}_i$（$i \in [k]$）的正半射线 $\{r\mathbf{W}_i \in \mathbb{R}^n: r \in \mathbb{R}_+\}$ 的一个邻域。细化这一观察，我们得到：

定理 8. 一个具有 L 个宽度为 $n_0$、秩为 k 的层的 maxout 网络，可以计算至少具有 $k^{L-1}k^{n_0}$ 个线性区域的函数。

定理 8 和命题 7 表明，深度 maxout 网络可以计算线性区域数随层数指数增长的函数，而且其增长速度比具有相同单元数的浅层模型的最大区域数快指数倍。与整流器模型类似，这种指数行为也可以就网络参数的数量来建立。

我们注意到，尽管某些可由 maxout 层计算的函数也可由整流器层计算，但上一节的整流器构造所导致的函数不能由 maxout 网络计算（秩-2 情形除外）。定理 8 的证明基于第 2 节相同的普遍论证，但使用了与定理 4 不同的构造（细节见附录 D）。

## 5 结论与展望

我们研究了深度前馈神经网络所计算函数在其线性区域数量意义上的复杂度。我们特别关注具有分段线性隐藏单元的深度神经网络，因为近来在许多机器学习应用中发现这类网络能提供优越的性能。我们讨论了如下思想：深度模型的每一层都能够以这样的方式识别其输入的片段，使得各层的复合识别出指数级数量的输入区域。这导致模型中更高层所计算函数的复杂度被指数级地复制。以这种方式由深度模型计算出的函数虽然复杂，但由于这些复制，它们仍具有一种内在的刚性，这可能有助于深度模型比浅层模型更好地泛化到未见样本。

这一框架适用于任何具有分段线性激活函数的神经网络。例如，如果我们考虑一个具有整流器单元的卷积网络，如（Krizhevsky et al. 2012）中所用的那样，我们可以看到：每一层的卷积加上最大池化，识别出池化区域内输入的所有图块（patch）。这将使这样的深度卷积神经网络递归地识别更低层图像的图块，从而在输入空间中得到指数级多的线性区域。

给定网络的参数空间被划分为若干区域，在这些区域中所得函数具有相应的线性区域。这些所计算函数线性区域之间的对应关系，可以用它们的邻接结构，或区域的交所构成的偏序集来描述。这类组合结构一般来说难以计算，即便对简单的超平面划分也是如此。未来分析中一个有趣的问题是：给定网络参数空间的许多区域，是否对应于具有给定数量线性区域的函数。

## 参考文献

M. Anthony and P. Bartlett. Neural Network Learning: Theoretical Foundations. Cambridge University Press, 1999.

D. Ciresan, U. Meier, J. Masci, and J. Schmidhuber. Multi column deep neural network for traffic sign classification. Neural Networks, 32:333–338, 2012.

G. Cybenko. Approximation by superpositions of a sigmoidal function. Mathematics of Control, Signals and Systems, 2(4):303–314, 1989.

O. Delalleau and Y. Bengio. Shallow vs. deep sum-product networks. In NIPS, 2011.

X. Glorot, A. Bordes, and Y. Bengio. Deep sparse rectifier neural networks. In AISTATS, 2011.

I. J. Goodfellow, D. Warde-Farley, M. Mirza, A. Courville, and Y. Bengio. Maxout networks. In Proceedings of The 30th International Conference on Machine Learning (ICML'2013), 2013.

G. Hinton, L. Deng, G. E. Dahl, A. Mohamed, N. Jaitly, A. Senior, V. Vanhoucke, P. Nguyen, T. Sainath, and B. Kingsbury. Deep neural networks for acoustic modeling in speech recognition. IEEE Signal Processing Magazine, 29(6):82–97, Nov. 2012.

K. Hornik, M. Stinchcombe, and H. White. Multilayer feedforward networks are universal approximators. Neural Networks, 2:359–366, 1989.O. Krause, A. Fischer, T. Glasmachers, and C. Igel. Approximation properties of DBNs with binary hidden units and real-valued visible units. In Proceedings of The 30th International Conference on Machine Learning (ICML'2013), 2013.

A. Krizhevsky, I. Sutskever, and G. Hinton. ImageNet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems 25 (NIPS'2012). 2012.

N. Le Roux and Y. Bengio. Deep belief networks are compact universal approximators. Neural Computation, 22(8):2192–2207, Aug. 2010.

G. Montúfar. Universal approximation depth and errors of narrow belief networks with discrete units. Neural Computation, 26, July 2014.

G. Montúfar and N. Ay. Refinements of universal approximation results for deep belief networks and restricted Boltzmann machines. Neural Computation, 23(5):1306–1319, May 2011.

V. Nair and G. E. Hinton. Rectified linear units improve restricted Boltzmann machines. In L. Bottou and M. Littman, editors, Proceedings of the Twenty-seventh International Conference on Machine Learning (ICML-10), pages 807–814. ACM, 2010.

R. Pascanu and Y. Bengio. Revisiting natural gradient for deep networks. In International Conference on Learning Representations, 2014.

R. Pascanu, G. Montúfar, and Y. Bengio. On the number of inference regions of deep feed forward networks with piece-wise linear activations. arXiv:1312.6098 [cs.LG], Dec. 2013.

R. Stanley. An introduction to hyperplane arrangements. In Lect. notes, IAS/Park City Math. Inst., 2004.

J. Susskind, A. Anderson, and G. E. Hinton. The Toronto face dataset. Technical Report UTML TR 2010-001, U. Toronto, 2010.

T. Zaslavsky. Facing Up to Arrangements: Face-Count Formulas for Partitions of Space by Hyperplanes. Number no. 154 in Memoirs of the American Mathematical Society. American Mathematical Society, 1975.

M. D. Zeiler and R. Fergus. Visualizing and understanding convolutional networks. Technical Report Arxiv 1311.2901, 2013.### A 输入空间邻域的识别

引理 2 的证明。每个输出空间邻域 $R \in P^L$ 的原像，是所有被 $\eta_L$ 以 $R$ 识别的输入空间邻域（即经 $\eta_L$——网络前 L 层所计算的函数——映射后像等于 $R$ 的那些输入空间邻域）。$R$ 的输入空间原像数量记为 $\mathcal{N}_R^L$。若每个 $R \in P^L$ 都是最后一层所计算函数 $h_L = g_L \circ f_L$ 的一个不同线性区域的像，那么由连续性，所有不同 $R \in P^L$ 的所有原像都属于 $\eta_L$ 的不同线性区域。因此，整个网络所计算函数的线性区域数至少等于所有 $R \in P^L$ 的原像数之和，即 $\mathcal{N} = \sum_{R \in P^L} \mathcal{N}_R^{L-1}$。

### B 整流器网络

定理 4 的证明。证明通过对合适选择的网络参数计数区域数来完成。构造的思路是把网络的前 $L-1$ 层分成 $n_{0}$ 个相互独立的部分；每个输入神经元对应一个部分。每个部分的参数选择成使其一维输入空间多次折叠到自身。对每个部分，每层的折叠次数等于每层的单元数。参见图 5。

如上所述，我们把每层的单元组织成 $n_0$ 个非空单元组，大小分别为 $p_1, \ldots, p_{n_0}$。这些大小的一种简单选择是，例如对宽度为 $n_l$ 的层取 $p_1 = \cdots = p_{n_0} = \lfloor n_1/n_0 \rfloor$，丢弃余下的单元。我们以这样的方式定义每个组的输入权重：使该组中的单元只对 $n_0$ 维输入空间的一个坐标敏感。由第 3.1 节的讨论，以正确的方式选择输入权重和偏置，一个组内 $p$ 个单元激活值的交替和会把它们的输入空间坐标折叠 $p$ 次到自身。由于激活值的交替和是仿射映射，它可以被吸收进下一层的预激活函数中。为使论证更透明，我们把一个组内 $p$ 个单元激活值的交替和 $\tilde{h} = h_1 - h_2 + \cdots \pm h_p$ 视为一个虚构中间单元的激活。所有这些 $n_0$ 个中间单元的复合输出把输入空间 $\mathbb{R}^{n_0}$ 划分为 $\prod_{i=1}^{n_0} p_i$ 个被识别区域的网格。这些输入空间区域中的每一个都被映射到中间层输出空间中的 $n_0$ 维单位立方体。我们把这个单位立方体视为下一隐藏层的有效输入，并重复该构造。这样，随着网络的每一层，被识别区域的数量乘以 $\prod_{i=1}^{n_0} p_i$，根据引理 2。下面我们明确讨论折叠的细节。

考虑其中一个网络部分，并考虑第 3.1 节中所用的权重。函数 $h$ 计算响应 $h_k$（$k \in [p]$）的交替和。只需证明该和把输入坐标折叠 $p$ 次到区间 $(0,1)$ 内即可。考察 $h_k$ 的取值，我们看到只需考察区间 $(0,1), (1,2), \ldots, (p-1,p)$。

考虑这些区间之一 $(k-1,k)$，对某个 $k\in[p]$。那么对所有 $x\in(k-1,k)$，有 $\tilde{h}(x)=x+2\sum_{i=1}^{k-1}(-1)^{i}(x-i)=(-1)^{k-1}(x-(k-1))-\frac{1}{2}((-1)^{k-1}-1)$。因此 $\tilde{h}(k-1)=-\frac{1}{2}((-1)^{k-1}-1)$ 且 $\tilde{h}(k)=(-1)^{k-1}-\frac{1}{2}((-1)^{k-1}-1)$。这两个值中总有一个为零，另一个为一，于是 $\tilde{h}(\{k-1,k\})=\{0,1\}$。由于函数在 $k-1$ 与 $k$ 之间是线性的，我们得到 $\tilde{h}$ 把区间 $(k-1,k)$ 映射到区间 $(0,1)$。

总的来说，被前 $L-1$ 层映射到第 $(L-1)$ 层（有效）输出空间的开单位超立方体 $(0,1)^{n_0}$ 上的输入空间邻域数量为

$$  \mathcal{N}_{(0,1)^{n_{0}}}^{L-1}=\prod_{l=1}^{L-1}\prod_{i=1}^{n_{0}}p_{l,i},   \tag*{(6)}$$

其中 $p_{l,i}$ 是第 $l$ 层中第 i 组单元的单元数。

最后隐藏层的输入和偏置可以这样选择，使得函数 $h_L$ 用 $n_L$ 个处于一般位置的超平面把其（有效）输入邻域 $(0,1)^{n_0}$ 划分，即划分为 $\sum_{j=0}^{n_0} \binom{n_L}{j}$ 个区域（参见第 2.2 节）。<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//f02051a8-6244-428f-9b9d-eed1b5785b88/markdown_3/imgs/img_in_image_box_215_163_1001_490.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-20T10%3A20%3A33Z%2F-1%2F%2F929b201ab5ddd979d3a8eb0f13c6bc0302d7d36ad702055165defd0df170224d" alt="Image" width="64%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图 5：定理 8 证明的图示。该图展示了一个被划分为 $n_0$ 个独立部分的整流器网络。每个部分只对输入空间的一个坐标敏感。每个部分的每一层都送入一个虚构的中间仿射单元（由下一层的预激活函数计算），该单元计算传往下一层的激活值。图中展示了所描绘整流器网络在 $n_0 = 2$ 时、在中间层所计算的函数。该函数由两次折叠组成；第一对隐藏单元沿平行于 x 轴的直线折叠输入空间 $\mathbb{R}^2$，第二对则沿平行于 y 轴的直线折叠。</div> </div>


令 $m_l$ 表示 $n_l$ 除以 $n_0$ 的余数。取 $p_{l,1} = \cdots = p_{l,n_0-m_l} = \lfloor n_l/n_0 \rfloor$ 以及 $p_{l,n_0-m_l+1} = \cdots = p_{l,n_0} = \lfloor n_l/n_0 \rfloor + 1$，我们得到线性区域总数为

$$  \left(\prod_{l=1}^{L-1}\left\lfloor\frac{n_{l}}{n_{0}}\right\rfloor^{n_{0}-m_{l}}\left(\left\lfloor\frac{n_{l}}{n_{0}}\right\rfloor+1\right)^{m_{l}}\right)\sum_{j=0}^{n_{0}}\binom{n_{L}}{j}.   \tag*{(7)}$$

当所有余数 $m_l = n_l - n_0 \lfloor n_l/n_0 \rfloor$ 为零时，它等于定理中给出的界，否则更大。证毕。

### C 用参数表示的我们的界

我们以隐藏单元数量为单位，计算了不同网络可计算函数最大线性区域数的界。不难把这些结果用网络的参数数量来表达，并推导出每增加一个参数所对应的线性区域数之渐近增长率的表达式。这类展开已在 Pascanu et al.（2013；命题 8）中计算过。一个具有 L 个宽度为 $n$ 的层的深度模型的参数数量行为为 $\Theta(Ln^2)$，即渐近地被 $Ln^2$ 从上下界住。一个具有 $Ln$ 个隐藏单元的浅层模型的参数数量行为为 $\Theta(Ln)$。我们的定理 4 以及第 2.2 节中关于浅层网络的讨论，蕴含如下渐近速率（每个参数对应的线性区域数）：

• 对于深度模型：$\Omega\left(\left(\frac{n}{n_0}\right)^{n_0(L-1)} \frac{n^{n_0-2}}{L}\right)$。

• 对于浅层模型：$O\left(L^{n_{0}-1}n^{n_{0}-1}\right)$。

这表明，对深度模型，最大线性区域数随参数数量指数增长；而对浅层模型，它仅随参数数量多项式增长。

### D Maxout 网络

命题 7 的证明。这里我们研究一个具有 n 个输入、m 个输出的秩-k maxout 层的最大线性区域数。在整流器单元的情形下，解就是<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//3874602f-16f0-4f9c-ba57-67541f14b14b/markdown_0/imgs/img_in_image_box_213_189_931_488.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-20T10%3A20%3A36Z%2F-1%2F%2Fc210c677c4d6171b8d17bf3e38bb05378564d7b062736c37f7e0779c6f59898a" alt="Image" width="58%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图 6：一个具有 $n = 3$ 个输入、$m = 2$ 个输出的秩-2 maxout 层的图示。预激活函数把输入映射到 $mk$ 维空间，其中 $k = 2$ 是该层的秩。激活函数对每组 $k$ 个预激活值取最大。该图展示了三维 Shi 划分 $S_3 \subset \mathbb{R}^3$（图中画的是与 $\{\mathbf{x} \in \mathbb{R}^3 : \mathbf{x}_1 + \mathbf{x}_2 + \mathbf{x}_3 = 1\}$ 的交）。这个划分对应一个具有 $n = 3$ 个输入、$m = 3$ 个输出的秩-3 maxout 层（在某组参数选择下）的输入空间划分。每对平行超平面界定一个 maxout 单元的线性区域。</div> </div>


超平面划分的最大区域数。在 maxout 单元的情形下，我们没有超平面划分。然而，我们可以用一个超平面划分的区域数来给出 maxout 层线性区域数的上界。论证如下。

如第 4 节所述，每个 maxout 单元把其输入划分为 $k$ 个实值线性函数上包络的线性区域。换句话说，输入空间被若干超平面片段划分，这些片段界定了"预激活向量中一个元素大于另一个元素"的输入之间的边界。这样的边界至多有 $k^2$ 个，因为每个都对应形如 $f_i(\mathbf{x}) = f_j(\mathbf{x})$ 的方程的解集。如果我们把每个这样的边界延拓成一个超平面，区域数只会增加。

该层的线性区域由各个单元区域的交集给出。因此，该层线性区域数被（很松地）以 $n$ 维空间中 $k^2 \cdot m$ 个超平面划分的区域数为上界。由 Zaslavsky（1975），后者为 $\sum_{j=0}^n \binom{k^2 m}{j}$，其行为为 $O((k^2 m)^n)$，即关于 $k$ 和 $m$ 的多项式。

定理 8 的证明。考虑一个每层具有 $n = n_0$ 个秩为 $k$ 的 maxout 单元的网络。参见图 5。我们定义 maxout 单元 $q_j$ 的种子，使得 $\{\mathbf{W}_{i,\cdot}\}_i$ 是指向 $\lfloor k/2\rfloor$ 个坐标向量正方向和负方向的单位向量。如果 $k$ 大于 $2n_0$，那么我们忽略其中 $k - 2n_0$ 个（对 $i > 2n_0$ 直接取 $\mathbf{W}_{i,\cdot} = 0$）。在此情形下，$q_j$ 关于法向量为 $e_i$（$i \leq \lfloor k/2\rfloor$）的坐标超平面对称，并对每个这样的 $i$ 有一个线性区域，梯度为 $e_i$。对其余的 $q_j$，我们考虑类似的函数，只是把坐标系在某个独立方向上稍微旋转。

这意味着每个 $q_j \circ (f_1, \ldots, f_k)$ 的输出是一个区间 $[0, \infty)$。每个这样的复合的线性区域把输入空间划分为 $r$ 个区域 $R_{j,1}, \ldots, R_{j,k}$。由于对它们各自所用的坐标变换是在独立方向上的轻微旋转，我们有 $R_i := \cap_j R_{j,i}$ 对每个 $i \in [k]$ 都是维数为 $n_0$ 的锥。此外，$q_j \circ f_j$（$j \in [n_0]$）在每个 $R_i$ 上的梯度构成 $\mathbb{R}^{n_0}$ 的一组基。因此，每个 $R_i$ 经该 maxout 层的像包含 $\mathbb{R}^{n_0}$ 中一个对所有 $i \in [k]$ 都相同的开锥。这个像可以通过偏置项平移，使得下一层的有效输入包含 $\mathbb{R}^{n_0}$ 中原点的一个开邻域。

上述论证表明，一个宽度为 $n_0$、秩为 $k$ 的 maxout 层可以识别其输入中至少 $k$ 个区域。因此一个具有 $L - 1$ 层的网络将识别输入中 $k^{L-1}$ 个区域。我们在第 4 节提到，maxout 层可以计算其线性区域对应于 Voronoi 图交集的函数。一般来说，描述 Voronoi 图的交集是困难的。有一些 Voronoi 图的叠加对应于已被充分理解的超平面划分。这里有两个特别好的例子：

例 9. 考虑一个具有 $n$ 个输入和 $m = n(n-1)/2$ 个秩-3 maxout 单元的层，这些单元用数对 $(i, j)$（$1 \leq i < j \leq n$）标记。输入权重和偏置可以这样选择，使得单元 $(i, j)$ 的区域由两个超平面 $H_{(i,j),s} = \{\mathbf{x} \in \mathbb{R}^n : \mathbf{x}_i - \mathbf{x}_j = s\}$（$s \in \{0,1\}$）界定。所有单元区域的交由超平面划分 $\{H_{(i,j),s}\}_{1 \leq i < j \leq n, s=1,2}$ 的区域给出，该划分称为 Shi 划分 $S_n$，具有 $(n+1)^{n-1}$ 个区域。图 6 右图展示了 Shi 划分 $S_3$。

一个相关的划分，对应于秩-4 maxout 单元，是 Catalan 划分，它包含平行的超平面三元组，总共有 $n!C_n$ 个区域，其中 $C_n := \frac{1}{n+1} \binom{2n}{n}$ 是 Catalan 数。关于这些划分的细节参见（Stanley 2004；推论 5.1 与命题 5.15）。

### E 其他网络

在引言中我们提到，我们对整流器网络和 maxout 网络的分析可作为研究其他类型前馈神经网络的平台。不深入许多细节，我们以卷积网络这一特殊情形为例加以说明。卷积网络是这样一种网络：其单元取值于特征（实值数组）空间，其边通过滤波器（实值数组）的卷积传递特征。由于卷积是线性映射，卷积网络的预激活函数与本文所考虑的预激活函数形式相同。它的输出是一个特征，但可以写成一个向量，就像我们所考虑的 $f_{l,i}$ 那样。因此，具有分段线性激活的卷积网络属于我们这里所考虑的网络类。唯一的区别在于，卷积网络相应的输入权重矩阵属于受限制的矩阵类。

### F 正弦边界实验

这里，我们描述为得到正文中图 1 所进行的实验。

在这个实验中，我们考虑了两个 MLP，其中一个是具有 20 个隐藏单元的单隐藏层，另一个是具有两个隐藏层、每层 10 个单元。这两个 MLP 在同一个合成数据集上使用共轭自然梯度（conjugate natural gradient）（Pascanu 和 Bengio 2014）训练，目的是最小化优化过程的影响。我们画出多次运行中最好的一次。浅层模型误分类了 123 个样本，而深度模型只误分类了 24 个样本。两层模型在捕捉正弦决策边界方面更好，因为它能定义更多的线性区域。

### G 可视化更高层中隐藏单元的行为

在本节中，我们描述如何可视化整流器 MLP 中折叠效果的细节，如第 2.6 节所讨论的。

任何分段线性函数都由组成它的不同线性片段完全确定。每个片段由其定义域——输入空间的一个区域 $R_i \subseteq \mathbb{R}^{n_0}$——以及描述其在 $R_i$ 上行为的线性映射 $f_i$ 给出。由于 $f_i$ 是仿射映射，它可以像浅层模型中的隐藏单元那样被解释。即，我们可以把 $f_i$ 写成：

 $$ \boldsymbol{f}_{i}(\mathbf{x})=\mathbf{u}^{\top}\mathbf{x}+c,\ \ \mathbf{x}\in R_{i}, $$ 

其中 $\mathbf{u}^\top$ 是一个行向量，且 $\mathbf{u}^\top \in \mathbb{R}^{n_0}$。于是 $f_i$ 度量 $\mathbf{x}$ 与 $\mathbf{u}^\top$ 之间（未归一化的）余弦距离。如果 $\mathbf{x}$ 是某个图像，那么 $\mathbf{u}^\top$ 也是一个图像，显示当 $\mathbf{x} \in R_i$ 时该单元所响应的模式（模板）。

给定输入空间中任意区域 $R_i$ 的一个输入样本 x，我们可以构造由第 l 层第 j 个单元生成的相应线性映射 $f_i$。具体地，线性映射 $f_i$ 的权重 $\mathbf{u}^\top$ 由下式计算：

$$  \mathbf{u}^{\top}=\left(\mathbf{W}_{l}\right)_{j:}\mathrm{d i a g}\left(\mathbf{I}_{f_{l-1}>0}\left(\mathbf{x}\right)\right)\mathbf{W}_{l-1}\cdots\mathrm{d i a g}\left(\mathbf{I}_{f_{1}>0}\left(\mathbf{x}\right)\right)\mathbf{W}_{1}.   \tag*{(8)}$$线性映射的偏置可以类似地计算。

由式 (8) 可以看到，某层 $l$ 上特定隐藏单元 $f_{l,j}$ 的线性映射，是通过追踪每一层直到第 $l$ 层所用的线性片段（$\mathbf{I}_{f_p>0}, p < l$——即指示函数）得到的。最后，乘以权重矩阵 $\mathbf{W}_l$ 的第 $j$ 行（$(\mathbf{W}_l)_j$）。虽然我们给出的是针对整流器 MLP 的公式，但把它推广到任何具有分段线性激活的 MLP（例如具有 maxout 激活的卷积神经网络）都是直接的。

由于由式 (8) 计算的线性映射依赖于每个样本/点 x，我们需要遍历一组点（例如训练样本）来识别某个隐藏单元的不同线性响应。尽管这不能给出所有可能的响应，但如果点集足够大，我们就可以得到足够多，从而更好地理解其行为。

我们在 Toronto Faces Dataset（TFD）（Susskind et al. 2010）上训练了一个具有三个隐藏层的整流器 MLP。前两个隐藏层各有 1000 个隐藏单元，最后一个有 100 个。

我们使用随机梯度下降训练该模型。我们使用系数为 $10^{-3}$ 的 $L_2$ 惩罚作为正则化，在前两个隐藏层上使用 dropout（丢弃概率为 0.5），并且通过在每一步 SGD 后投影权重，强制权重按列具有单位范数。我们使用学习率 0.1，输出层由 sigmoid 单元组成。这些正则化方案以及 sigmoid 输出层的目的是获得更干净、更锐利的滤波器。该模型在数据集第 1 折上训练，达到 20.49% 的错误率，对该数据集和一个非卷积模型而言这是合理的。

由于第一层的每个隐藏单元只响应单个线性区域，我们直接可视化第一隐藏层中随机选取的 16 个单元所学到的权重向量。它们显示在图 7 的第一行。

另一方面，对于任何其他隐藏层，我们每层随机选取 20 个单元，并根据每个单元不同线性响应之间的最大欧氏距离，从中可视化最有趣的四个单元。每个单元的线性响应，是通过把在训练集上得到的响应（我们只考虑激活为正的那些响应）用 K-means 算法聚类成四类来计算的。我们展示每个聚类中的代表性线性响应（参见图 7 的第二行和第三行）。

类似地，我们可视化每个输出单元所学到的线性映射。对于输出层，我们在图 8 中展示全部七个单元的可视化。

通过观察某个隐藏单元所响应的不同线性区域之间的差异，我们可以研究该单元所学到的不变性类型。在图 9 中，我们展示第三个隐藏层最后一个被可视化的隐藏单元所学到的四个线性映射之间的差异（可视化线性映射的最后一列）。

从这些可视化中，我们可以看到隐藏单元在更高层学会对更抽象、更有趣的平移保持不变。我们还能清楚地看到更高层中某个隐藏单元的各种不变性类型。

Zeiler 和 Fergus（2013）试图可视化上层单元的行为，具体地说是具有整流器的深度卷积网络中单元的行为。这一方法在某种程度上类似于我们这里提出的方法，区别在于我们除了"网络中的隐藏单元使用分段线性激活函数"之外，不做任何其他假设。

考虑可视化的视角也不同。Zeiler 和 Fergus（2013）从（近似）求神经网络前馈计算的逆这一角度来处理可视化问题，而我们的方法是通过识别每个隐藏单元的一组线性映射来导出的。

这一差异导致在实际实现上有若干小的不同。例如，Zeiler 和 Fergus（2013）通过简单地使用另一个整流器来近似整流器的逆。而我们不需要近似整流器的逆。相反，我们试图识别输入空间中映射到相同激活的区域。

在我们的方法中，可以可视化输入空间中映射到某个隐藏单元相同激活的真实点。在图 10 中，我们展示输入空间中三个不同的点，它们使<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//3874602f-16f0-4f9c-ba57-67541f14b14b/markdown_3/imgs/img_in_image_box_257_160_869_1011.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-20T10%3A20%3A40Z%2F-1%2F%2Fab798b28882f7ce9a91a32ba66183f5cddc4f50ad7b0731bdf0c5ffe6adea23e" alt="Image" width="50%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图 7：在一个于 TFD 数据集上训练的整流器 MLP 中，各隐藏层所学线性映射的可视化。每一行对应一个隐藏层。第一列显示未归一化的线性映射，最后一列显示归一化的线性映射，只展示每个映射的方向。颜色仅用于更好地区分不同的滤波器。</div> </div>


第三个隐藏层中一个随机选取的隐藏单元激活恰好为 2.5。我们通过以下方式找到这些点：先找到三个映射到同一隐藏单元激活接近 2.5 的训练样本，然后从每个找到的样本出发，沿着线性映射（由式 (8) 计算）搜索一个恰好导致激活为 2.5 的点。显然，找到的点不是训练样本之一。从这三个点，我们可以看到所选的隐藏单元响应于一张张着大嘴、睁着一组眼睛的脸，同时对脸的其他特征（例如眉毛）保持不变。通过扰动分析，我们可以假定在每一个这样的点周围存在一个开集被该隐藏单元所识别。<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//cb45cdee-8ef9-497a-9724-2d05867745d9/markdown_0/imgs/img_in_image_box_222_205_1005_423.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-20T10%3A20%3A42Z%2F-1%2F%2Fa797a1c8168a1f4775fddab7fd98ac36721c391cdaa059faeddae7858937ef58" alt="Image" width="63%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图 8：在 TFD 数据集上训练的整流器 MLP 输出单元的线性映射。各列对应的类别标签为：(1) 愤怒，(2) 厌恶，(3) 恐惧，(4) 快乐，(5) 悲伤，(6) 惊讶，(7) 中性。</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//cb45cdee-8ef9-497a-9724-2d05867745d9/markdown_0/imgs/img_in_image_box_349_613_877_900.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-20T10%3A20%3A42Z%2F-1%2F%2F4d10e2994509f9bc92c6cc48908ecc5b7cb4ac52955bd51546a79327bb8e4ce1" alt="Image" width="43%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图 9：在 TFD 上训练的整流器 MLP 第三个隐藏层中单个隐藏单元不同线性区域之间的差异。</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//cb45cdee-8ef9-497a-9724-2d05867745d9/markdown_0/imgs/img_in_image_box_448_1090_772_1311.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-20T10%3A20%3A42Z%2F-1%2F%2F6d0e24101ff19f817b09250eb075b327b416f7349753b95ba8a3f2a9c0181f4b" alt="Image" width="26%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图 10：输入空间中三个不同点的可视化，它们映射到第三个隐藏层中一个随机选取的隐藏单元的相同激活。顶行显示输入空间中的三个点（非训练/测试样本），对每个点，我们在下方画出其线性映射。</div> </div>

