arXiv:2008.02217v3 [cs.NE] 28 Apr 2021

# 霍普菲尔德网络就是你所需要的一切

## 摘要

我们提出了一种具有连续状态的现代霍普菲尔德网络（modern Hopfield network）及其相应的更新规则。新的霍普菲尔德网络能够存储指数级（随联想空间的维度）多的模式，只需一次更新即可检索出模式，并且检索误差呈指数级小。它具有三类能量极小值（更新的不动点）：(1) 对所有模式求平均的全局不动点，(2) 对模式子集求平均的亚稳态，以及 (3) 存储单个模式的不动点。新的更新规则等价于 Transformer 中使用的注意力机制。这一等价性使得我们能够刻画 Transformer 模型的注意力头（head）。这些注意力头在浅层倾向于执行全局平均，在较深层则通过亚稳态执行部分平均。新的现代霍普菲尔德网络可以作为层集成到深度学习架构中，以支持对原始输入数据、中间结果或学习到的原型（prototype）进行存储与访问。
这些霍普菲尔德层（Hopfield layer）开启了超越全连接、卷积或循环网络的新深度学习方式，并提供了池化、记忆、联想和注意力机制。我们在多个领域展示了霍普菲尔德层的广泛适用性。在所考虑的四项多示例学习（multiple instance learning）问题中，霍普菲尔德层在三项上刷新了最优结果，并在具有数十万个实例的免疫组库分类任务上取得提升。在深度学习方法通常表现不佳的 UCI 小规模分类任务基准集上，与不同的机器学习方法相比，霍普菲尔德层取得了新的最优结果。最后，霍普菲尔德层在两个药物设计数据集上达到了最优性能。实现代码见：https://github.com/ml-jku/hopfield-layers

## 1 引言

深度学习社区一直在寻找替代循环神经网络（RNN）来存储信息的方法。例如，线性记忆网络（linear memory network）使用序列的线性自编码器作为记忆（Carta et al., 2020）。人们还提出了用于 RNN 的其他记忆形式，如全息缩减表示（holographic reduced representation）（Danihelka et al., 2016）、张量积表示（tensor product representation）（Schlag & Schmidhuber, 2018; Schlag et al., 2019）以及经典联想记忆（被扩展为快速权重方法）（Schmidhuber, 1992; Ba et al., 2016a; b; Zhang & Zhou, 2017; Schlag et al., 2021）。大多数关于新记忆的方法都基于注意力。神经图灵机（NTM）配备了一个外部记忆和一个注意力过程（Graves et al., 2014）。记忆网络（memory network）（Weston et al., 2014）使用 arg max 注意力：先将查询和模式映射到一个空间，然后检索点积最大的模式。端到端记忆网络（EMN）通过用 softmax 替代 arg max，使这一注意力方案可微（Sukhbaatar et al., 2015a; b）。使用点积的 EMN 变得非常流行，并为自注意力实现了一种键值注意力（key-value attention）（Daniluk et al., 2017）。EMN 的一种改进是 Transformer（Vaswani et al., 2017a; b）及其

扩展（Dehghani et al., 2018）。Transformer 对自然语言处理（NLP）领域产生了巨大影响，尤其是通过 BERT 模型（Devlin et al., 2018; 2019）。

本工作的贡献：(i) 引入通过现代霍普菲尔德网络配备记忆的新型深度学习层；(ii) 为连续现代霍普菲尔德网络引入新的能量函数和新的更新规则，它们可微，并且通常在一次更新后即可检索出模式。可微性是梯度下降参数更新所必需的，而一次更新完成检索则与深度网络中每层只激活一次的特性相兼容。

我们建议使用现代霍普菲尔德网络在神经网络的各层中存储信息或学习到的原型。二值霍普菲尔德网络最初是作为能够存储和检索模式的联想记忆而被提出的（Hopfield, 1982）。一个查询模式可以检索出与它最相似的模式，或者对相似模式取平均。霍普菲尔德网络看起来是一种古老的技术，但新的能量函数改善了它的性质。伪状态或亚稳态的稳定性被显著降低（Barra et al., 2018）。最大且最具影响力的成功报道在于提升霍普菲尔德网络的存储容量。在 d 维空间中，标准霍普菲尔德模型可以无误差地存储 d 个不相关模式，但对于固定稳定模式只能存储  $Cd/log(d)$ 个随机模式（ $C < 1/2$），若要求所有模式都稳定则为  $C < 1/4$（McEliece et al., 1987）。非线性学习规则也服从相同的界（Mazza, 1997）。通过折衷取舍并允许较小的检索误差，存储容量约为  $0.138d$（Crisanti et al., 1986; Hertz et al., 1991; Torres et al., 2002）。如果学习规则与 Hebb 规则无关，那么最多可以存储 d 个模式（Abu-Mostafa & StJacques, 1985）。对于具有非零对角矩阵的霍普菲尔德网络，存储容量可以提升到  $Cd\log(d)$（Folli et al., 2017）。与存储容量形成对比的是，霍普菲尔德网络的能量极小值（伪状态、稳定状态）数目随  $d$ 呈指数增长（Tanaka & Edwards, 1980; Bruck & Roychowdhury, 1990; Wainrib & Touboul, 2013）。

标准二值霍普菲尔德网络的能量函数可以表示为交互函数  $F$ 之和，其中  $F(x) = x^2$。现代霍普菲尔德网络，也称为"稠密联想记忆"（dense associative memory, DAM）模型，使用交互函数形如  $F(x) = x^n$ 的能量函数，从而获得与  $d^{n-1}$ 成正比的存储容量（Krotov & Hopfield, 2016; 2018）。现代霍普菲尔德网络的能量函数使其对对抗攻击具有鲁棒性（Krotov & Hopfield, 2018）。基于形如  $F(x) = \exp(x)$ 的交互函数的现代二值霍普菲尔德网络，甚至可以达到  $2^{d/2}$ 的存储容量，其中所有存储的二值模式都是不动点，但吸引半径消失（Demircigil et al., 2017）。然而，为了将霍普菲尔德网络集成到深度学习架构中，有必要使其可微，即我们需要连续霍普菲尔德网络（Hopfield, 1984; Koiran, 1994）。

因此，我们将 Demircigil et al. (2017) 基于指数交互函数的能量函数推广到连续模式和状态，得到了一个新的现代霍普菲尔德网络。我们还提出了一个新的更新规则，它保证全局收敛到能量的驻点（局部极小值或鞍点）。我们证明，新的现代霍普菲尔德网络通常在一次更新步骤内检索出模式（ $\epsilon$-接近不动点），且误差呈指数级小，其存储容量与  $c^{\frac{d-1}{4}}$ 成正比（定理 3 给出了  $c = 1.37$ 和  $c = 3.15$ 的合理设置）。一次更新即可检索模式，对于将霍普菲尔德网络集成到层只激活一次的深度学习架构中至关重要。令人惊讶的是，我们的新更新规则正是 Transformer 和 BERT 模型中使用的键值注意力（见图 1）。我们的现代霍普菲尔德网络可以作为新的一层集成到深度学习架构中，用于池化、记忆、原型学习和注意力。我们在不同的基准数据集和任务（如免疫组库分类）上测试了这些新层。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//23c5d1cc-85ec-47b8-8da6-11dc5c52f794/markdown_1/imgs/img_in_image_box_215_1228_1008_1342.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-09-22T02%3A19%3A36Z%2F-1%2F%2F4392ded76e819787c6a6fae63e9a72dd87285ecd016da4023df5d7fc6ecd1673" alt="Image" width="64%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图 1：我们将二值现代霍普菲尔德网络的能量推广到连续状态，同时保持快速收敛和存储容量特性。我们还提出了一个使能量最小化的新更新规则。该新更新规则就是 Transformer 的注意力机制。公式经过修改以将 softmax 表示为行向量。"=" 号表示"保持这些性质"。</div> </div>


## 2 具有连续状态的现代霍普菲尔德网络

连续状态霍普菲尔德网络的新能量函数。为了将现代霍普菲尔德网络集成到深度学习架构中，我们必须使其连续化。为了允许连续状态，我们提出了一个新的能量函数，它是对现代霍普菲尔德网络能量（Demircigil et al., 2017）的修改。我们还提出了一个新的更新规则，可以证明它收敛到能量的驻点（局部极小值或鞍点）。

我们有 $N$ 个存储的（键）模式 $\boldsymbol{x}_i \in \mathbb{R}^d$，由矩阵 $\boldsymbol{X} = (\boldsymbol{x}_1, \ldots, \boldsymbol{x}_N)$ 表示，其中最大模式为 $M = \max_i \|\boldsymbol{x}_i\|$。状态（查询）模式为 $\boldsymbol{\xi} \in \mathbb{R}^d$。对于指数交互函数，我们需要 $0 < \beta$ 时的 log-sum-exp 函数（lse）

 $$ \mathrm{l s e}(\beta,\pmb{x})=\beta^{-1}\log\left(\sum_{i=1}^{N}\exp(\beta x_{i})\right), $$ 

它是凸的（见附录式 (461) 和引理 A22）。对于二值模式  $\boldsymbol{x}_i$ 和二值状态模式  $\boldsymbol{\xi}$，现代霍普菲尔德网络的能量函数 E 为  $E = -\sum_{i=1}^N F\left(\boldsymbol{\xi}^T \boldsymbol{x}_i\right)$。（Krotov & Hopfield, 2016）。这里， $F(x) = x^n$ 是交互函数，其中  $n = 2$ 给出经典霍普菲尔德网络。存储容量与  $d^{n-1}$ 成正比（Krotov & Hopfield, 2016）。该模型被 Demircigil et al. (2017) 推广到指数交互函数  $F(x) = \exp(x)$，给出能量  $E = -\exp(\text{lse}(1, \boldsymbol{X}^T \boldsymbol{\xi}))$。该能量对二值模式产生  $N = 2^{d/2}$ 的指数级存储容量。此外，只需一次更新，就能以高概率恢复随机模式的不动点。然而，这个现代霍普菲尔德网络仍然具有二值状态。

我们将这一能量函数推广到连续取值的模式，同时保持现代霍普菲尔德网络的性质，如指数级存储容量和极快的收敛速度（见图 1）。对于新能量，我们取现代霍普菲尔德网络负能量的对数，并加上当前状态的二次项。二次项确保状态向量  $\xi$ 的范数保持有限，且能量有界。经典霍普菲尔德网络不需要限制其状态向量的范数，因为它是二值的且长度固定。我们定义新的能量函数 E 为

 $$ \mathrm{E}=-\mathrm{l s e}(\beta,\boldsymbol{X}^{T}\boldsymbol{\xi})+\frac{1}{2}\boldsymbol{\xi}^{T}\boldsymbol{\xi}+\beta^{-1}\log N+\frac{1}{2}M^{2}. $$ 

我们有  $0 \leqslant \mathrm{E} \leqslant 2M^2$（见附录引理 A1）。使用  $p = \text{softmax}(\beta X^T \xi)$，我们定义一个新的更新规则（见图 1）：

 $$ \pmb{\xi}^{\mathrm{n e w}}=f(\pmb{\xi})~=~X p~=~X{\mathrm{s o f t m a x}}(\beta X^{T}\pmb{\xi})~. $$ 

下一个定理表明更新规则式 (3) 全局收敛。证明使用了凹凸过程（Concave-Convex Procedure, CCCP）（Yuille & Rangarajan, 2002; 2003），它等价于 Legendre 最小化（Rangarajan et al., 1996; 1999）算法（Yuille & Rangarajan, 2003）。

定理 1. 更新规则式 (3) 全局收敛：对于  $\xi^{t+1} = f(\xi^t)$，当  $t \to \infty$ 且  $\xi^*$ 为不动点时，能量  $\mathrm{E}(\xi^t) \to \mathrm{E}(\xi^*)$。

证明. 式 (3) 中的更新规则是用于最小化能量 E 的 CCCP，而 E 是凸的 $1/2\xi^{T}\xi$ 与凹的 -lse 之和（详见附录定理 1）。Yuille & Rangarajan (2002) 中的定理 2 给出了全局收敛性质。此外，Sriperumbudur & Lanckriet (2009) 中的定理 2 通过使用 Zangwill 迭代算法全局收敛理论的严格分析，证明了 CCCP 的全局收敛性。

全局收敛定理只保证当  $t \to \infty$ 时能量  $\mathrm{E}(\boldsymbol{\xi}^{t}) \to \mathrm{E}(\boldsymbol{\xi}^{*})$，而不保证  $\boldsymbol{\xi}^{t} \to \boldsymbol{\xi}^{*}$。下一个定理强化了 Zangwill 全局收敛定理（Meyer, 1976），并给出了类似于期望最大化（Wu, 1983）已知结果的收敛结论。

$\mathbf{1}$neorem $\mathcal{L}$. 对于迭代 $\mathcal{L}q$。 $(\mathcal{S})$，当 $t\to\infty$ 时我们有 $\mathcal{E}(\mathfrak{S}^{\circ})\to\mathcal{E}(\mathfrak{S}^{\circ})=\mathcal{E}^{\circ}$，对某个驻点 $\mathfrak{S}^{\circ}$。此外，$\left\|\mathfrak{S}^{t+1}-\mathfrak{S}^{t}\right\|\to0$，并且要么 $\{\mathfrak{S}^{t}\}_{t=0}^{\infty}$ 收敛，要么在另一种情况下，$\{\mathfrak{S}^{t}\}_{t=0}^{\infty}$ 的极限点集合是 $\mathcal{L}(\mathrm{E}^{\circ})$ 的一个连通紧子集，其中 $\mathcal{L}(a)=\{\mathfrak{S}\in\mathcal{L}\mid\mathrm{E}(\mathfrak{S})=a\}$，$\mathcal{L}$ 是迭代 $Eq.(3)$ 的驻点集合。如果 $\mathcal{L}(\mathrm{E}^{\circ})$ 有限，那么由迭代 $Eq.(3)$ 生成的任意序列 $\{\mathfrak{S}^{t}\}_{t=0}^{\infty}$ 都收敛到某个 $\mathfrak{S}^{\circ}\in\mathcal{L}(\mathrm{E}^{\circ})$。

证明见附录定理 2。因此，由迭代式 (3) 生成的任意序列的所有极限点都是能量函数 E 的驻点（局部极小值或鞍点）。该迭代要么收敛，要么其极限点集合是一个连通紧集。

下一个定理给出了关于我们新的连续状态现代霍普菲尔德网络存储容量的结果。我们首先定义使用连续状态现代霍普菲尔德网络存储和检索模式的含义。

定义 1（模式的存储与检索）. 我们假设每个模式  $\boldsymbol{x}_i$ 周围给定一个球  $\mathrm{S}_i$。如果存在唯一的不动点  $\boldsymbol{x}_i^* \in \mathrm{S}_i$，使得所有点  $\boldsymbol{\xi} \in \mathrm{S}_i$ 都收敛到它，并且当  $i \neq j$ 时  $\mathrm{S}_i \cap \mathrm{S}_j = \emptyset$，则称  $\boldsymbol{x}_i$ 被存储。对于给定的  $\epsilon$，如果迭代（更新规则）式 (3) 给出的点  $\tilde{\boldsymbol{x}}_i$ 至少  $\epsilon$-接近唯一不动点  $\boldsymbol{x}_i^* \in \mathrm{S}_i$，则称  $\boldsymbol{x}_i$ 被检索。检索误差为  $\|\tilde{\boldsymbol{x}}_i - \boldsymbol{x}_i\|$。

与经典霍普菲尔德网络一样，我们考虑球面上的模式，即具有固定范数的模式。对于随机选择的模式，可存储的模式数目随模式空间的维度  $d$ 呈指数增长（ $x_i \in \mathbb{R}^d$）。

定理 3. 我们假设失败概率为  $0 < p \leqslant 1$，且模式随机选取于半径为  $M := K\sqrt{d-1}$ 的球面上。我们定义  $a := \frac{2}{d-1}(1 + \ln(2\beta K^2 p(d-1)))$、 $b := \frac{2K^2\beta}{5}$、 $c := \frac{b}{W_0(\exp(a + \ln(b)))}$，其中  $W_0$ 是 Lambert W 函数的上分支（Olver et al., 2010, (4.13)），并确保  $c \geq \left(\frac{2}{\sqrt{p}}\right)^{\frac{4}{d-1}}$。那么以概率 1 - p，可存储的随机模式数目为

 $$ N\;\geq\;\sqrt{p}\;c^{\frac{d-1}{4}}\;. $$ 

因此，对于 $\beta=1$、$K=3$、$d=20$、$p=0.001$（$a+\ln(b)>1.27$），可证明 $c\geq3.1546$；对于 $\beta=1$、$K=1$、$d=75$、$p=0.001$（$a+\ln(b)<-0.94$），可证明 $c\geq1.3718$。

### 证明见附录定理 A5。

下一个定理表明，更新规则通常在一次更新后即可检索出模式。对于不动点  $\boldsymbol{x}_i^*$ 和查询  $\boldsymbol{\xi}$，模式  $\boldsymbol{x}_i$ 的检索通过  $\epsilon$ 定义为  $\|\boldsymbol{f}(\boldsymbol{\xi}) - \boldsymbol{x}_i^*\| < \epsilon$，即更新结果  $\epsilon$-接近不动点。一次更新完成检索，对于将现代霍普菲尔德网络集成到层只激活一次的深度学习架构中至关重要。首先我们需要模式分离度的概念。对于模式  $\boldsymbol{x}_i$，我们将其到其他模式的分离度  $\Delta_i$ 定义为：

 $$ \Delta_{i}\;:=\;\operatorname*{m i n}_{j,j\neq i}\left(\boldsymbol{x}_{i}^{T}\boldsymbol{x}_{i}\;-\;\boldsymbol{x}_{i}^{T}\boldsymbol{x}_{j}\right)\;=\;\boldsymbol{x}_{i}^{T}\boldsymbol{x}_{i}\;-\;\operatorname*{m a x}_{j,j\neq i}\boldsymbol{x}_{i}^{T}\boldsymbol{x}_{j}\;. $$ 

对于分离良好的模式，即具有较大  $\Delta_i$ 的模式，更新规则可以在一次更新后检索出模式。

定理 4. 给定查询 $\xi$，一次更新后新点 $f(\xi)$ 到不动点 $\boldsymbol{x}_i^*$ 的距离在分离度 $\Delta_i$ 上呈指数级小。使用雅可比矩阵 $\mathrm{J} = \frac{\partial f(\xi)}{\partial \xi}$ 及其在均值定理中的取值 $\mathrm{J}^m$，精确界为：

 $$ \begin{align*}\left\|f(\boldsymbol{\xi})-\boldsymbol{x}_{i}^{*}\right\|&\leq\left\|\mathrm{J}^{m}\right\|_{2}\left\|\boldsymbol{\xi}-\boldsymbol{x}_{i}^{*}\right\|,\\\left\|\mathrm{J}^{m}\right\|_{2}&\leq2\beta N M^{2}\left(N-1\right)\exp(-\beta\left(\Delta_{i}-2\max\{\left\|\boldsymbol{\xi}-\boldsymbol{x}_{i}\right\|,\left\|\boldsymbol{x}_{i}^{*}-\boldsymbol{x}_{i}\right\|\}\right.M)).\end{align*} $$ 

对于给定的 $\epsilon$ 和足够大的 $\Delta_i$，我们有 $\|f(\xi) - x_i^*\| < \epsilon$，即一次更新完成检索。

证明见附录定理 A8。

同时，检索误差随分离度  $\Delta_{i}$ 呈指数下降。

定理 5（指数级小的检索误差）. 模式  $x_i$ 的检索误差  $\|f(\xi) - x_i\|$ 的界为

 $$ \begin{array}{r}{\|f(\pmb{\xi})-\pmb{x}_{i}\|\leqslant2\left(N-1\right)\exp(-\beta\left(\Delta_{i}-2\operatorname*{m a x}\{\|\pmb{\xi}-\pmb{x}_{i}\|,\|\pmb{x}_{i}^{*}-\pmb{x}_{i}\}|\right)M\})}\end{array} $$ 

并且当  $\|\boldsymbol{x}_i - \boldsymbol{x}_i^*\| \leqslant \frac{1}{2\beta M}$ 且  $\|\boldsymbol{x}_i - \boldsymbol{\xi}\| \leqslant \frac{1}{2\beta M}$ 时，有

 $$ \left\| \mathbf{x}_{i}-\mathbf{x}_{i}^{*} \right\|\leqslant2e\left(N-1\right)M\exp(-\beta\Delta_{i}). $$ 

证明见附录定理 A9。

亚稳态与一个全局不动点。到目前为止，我们考虑的是分离良好的模式  $x_i$，迭代收敛到接近某个模式  $x_i$ 的不动点。如果没有模式  $x_i$ 与其他模式分离良好，则迭代收敛到接近这些向量算术平均值的全局不动点。此时 softmax 向量  $p$ 接近均匀分布，即  $p_i = 1/N$。如果某些向量彼此相似且与所有其他向量分离良好，则存在一个靠近这些相似向量的亚稳态。从亚稳态附近出发的迭代会收敛到该亚稳态，即使由这些相似模式之一初始化也是如此。关于收敛到一个全局不动点和收敛到亚稳态的证明，分别见附录引理 A7 和引理 A12。

霍普菲尔德更新规则即 Transformer 的注意力。霍普菲尔德网络的更新规则就是 Transformer 和 BERT 模型中使用的注意力机制（见图 1）。为说明这一点，我们假设有 $N$ 个存储的（键）模式 $y_i$ 和 $S$ 个状态（查询）模式 $r_i$，它们被映射到维度为 $d_k$ 的霍普菲尔德空间。我们令 $\boldsymbol{x}_i = \boldsymbol{W}_K^T \boldsymbol{y}_i$、$\boldsymbol{\xi}_i = \boldsymbol{W}_Q^T \boldsymbol{r}_i$，并将更新规则的结果乘以 $\boldsymbol{W}_V$。矩阵 $\boldsymbol{Y} = (\boldsymbol{y}_1, \ldots, \boldsymbol{y}_N)^T$ 和 $\boldsymbol{R} = (\boldsymbol{r}_1, \ldots, \boldsymbol{r}_S)^T$ 将 $\boldsymbol{y}_i$ 和 $\boldsymbol{r}_i$ 作为行向量组合起来。我们定义矩阵 $\boldsymbol{X}^T = \boldsymbol{K} = \boldsymbol{Y} \boldsymbol{W}_K$、$\boldsymbol{\Xi}^T = \boldsymbol{Q} = \boldsymbol{R} \boldsymbol{W}_Q$ 和 $\boldsymbol{V} = \boldsymbol{Y} \boldsymbol{W}_K \boldsymbol{W}_V = \boldsymbol{X}^T \boldsymbol{W}_V$，其中 $\boldsymbol{W}_K \in \mathbb{R}^{d_y \times d_k}$、$\boldsymbol{W}_Q \in \mathbb{R}^{d_r \times d_k}$、$\boldsymbol{W}_V \in \mathbb{R}^{d_k \times d_v}$。如果 $\beta = 1/\sqrt{d_k}$ 且 softmax $\in \mathbb{R}^N$ 改为行向量，那么对式 (3) 的更新规则乘以 $\boldsymbol{W}_V$ 可得：

 $$ \begin{array}{c} \boldsymbol{Z}~=~\mathrm{softmax}\left(1/{\sqrt{d_{k}}}\boldsymbol{Q}\boldsymbol{K}^{T}\right)~\boldsymbol{V}~=~\mathrm{softmax}\left(\beta\boldsymbol{R}\boldsymbol{W}_{Q}\boldsymbol{W}_{K}^{T}\boldsymbol{Y}^{T}\right)~\boldsymbol{Y}\boldsymbol{W}_{K}\boldsymbol{W}_{V}~. \end{array} $$ 

式 (10) 的左半部分是 Transformer 注意力。在 Transformer 自注意力中  $R = Y$，且  $W_K W_V$ 被替换为  $W_V$。除注意力机制外，霍普菲尔德网络还允许在深度网络架构中实现其他功能，我们将在下一节通过特定的层来介绍。式 (10) 的右半部分用于解释这些特定的层。

## 3 面向深度学习的新型霍普菲尔德层

具有连续状态的现代霍普菲尔德网络可以集成到深度学习架构中，因为它们相对于其参数是连续且可微的。此外，它们通常在一次更新后即可检索出模式，这与只激活一次的深度学习层相一致。基于这两个原因，现代霍普菲尔德网络可以作为深度网络中的专门层，为它们配备记忆。下面，我们介绍三种类型的霍普菲尔德层：Hopfield、HopfieldPooling 和 HopfieldLayer。霍普菲尔德层在深度网络架构中的可能应用包括：

• 多示例学习（MIL）（Dietterich et al., 1997）。

点集的处理与学习（Qi et al., 2017a; b; Xu et al., 2018），

基于集合的、置换不变的学习（Guttenberg et al., 2016; Ravanbakhsh et al., 2016; Zaheer et al., 2017; Korshunova et al., 2018; Ilse et al., 2018; Zhai et al., 2020）。

• 基于注意力的学习（Vaswani et al., 2017a）。

• 使用联想记忆的深度学习（Graves et al., 2014; Weston et al., 2014; Ba et al., 2016a; b; Schlag & Schmidhuber, 2018; Schlag et al., 2019），

• 自然语言处理（Devlin et al., 2018; 2019）。

序列分析与时间序列预测（Hochreiter, 1991; Hochreiter & Schmidhuber, 1997; Cho et al., 2014），以及

- 存储和检索参考数据，例如训练数据、离群点、高误差数据点、原型或聚类中心、支持向量与边界样本。

霍普菲尔德网络层可以替代现有的层，如池化层、置换等变层（Guttenberg et al., 2016; Ravanbakhsh et al., 2016）、GRU（Cho et al., 2014）和 LSTM（Hochreiter, 1991; Hochreiter & Schmidhuber, 1997）层，以及注意力层（Vaswani et al., 2017a;b; Bahdanau et al., 2014）。

神经网络的类型。我们考虑两类前馈神经网络：(I) 将激活向量从输入层传播到输出层的神经网络，例如全连接或卷积神经网络。(II) 将一组向量从输入层传播到输出层的神经网络，其中每一层对集合中的每个元素施加相同的操作，输出层可以通过一个向量汇总该集合。Transformer 就是一个例子。循环神经网络属于类型 (I) 的网络，它被迭代地施加到一个集合或序列上，其中中间结果存储在记忆中并可被重用。现代霍普菲尔德网络可以集成到这两类神经网络架构中，并能够为其每一层配备联想记忆。见图 2。

新型霍普菲尔德层的类型。我们引入三种类型的霍普菲尔德层：Hopfield、HopfieldPooling 和 HopfieldLayer。连续现代霍普菲尔德网络催生了大量新的深度学习架构，因为我们可以 (a) 传播集合或单个向量，(b) 传播查询、

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//3766c420-a123-4bc9-a073-3a9f5e81f6bc/markdown_2/imgs/img_in_image_box_727_167_1011_427.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-09-22T02%3A19%3A42Z%2F-1%2F%2F7969f740ea68d3ebdfbcf5d4516f25ed76e8c8e9513ea83d712bdcafa92754f3" alt="Image" width="23%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图 2：左：带有层（☐）的标准深度网络将向量或向量集合从输入传播到输出。右：一个深度网络，其中各层（☐）通过霍普菲尔德层（☐）配备联想记忆。</div> </div>


存储的模式，或两者兼有，(c) 学习静态查询或存储的模式，(d) 通过训练集、原型或外部数据填充记忆。接下来，我们提供三种实用的霍普菲尔德层类型。实现代码见：https://github.com/ml-jku/hopfield-layers

(1) 层 Hopfield，适用于通过状态（查询）模式 R 和存储的（键）模式 Y 传播向量集合的网络。层 Hopfield 是公式 (10) 的实现。Hopfield 层的记忆可以由输入或前几层的集合填充，见图 3。记忆也可以由参考集填充，这通过将参考集作为额外输入来提供。因此，层 Hopfield 允许两个集合之间的联想。执行这种联想的一个典型层是 Transformer 注意力机制，它关联键和查询，例如需要比较的两个点集。该层支持不同类型的序列到序列学习、点集操作和基于检索的方法。ResNet 架构中带有跳跃连接的层 Hopfield 与流行的 Transformer 和 BERT 模型完全相同。在实验中，我们在 Transformer 架构中分析了这些霍普菲尔德层。在比较 UCI 基准集小数据集上机器学习方法的实验中，也使用了层 Hopfield。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//3766c420-a123-4bc9-a073-3a9f5e81f6bc/markdown_2/imgs/img_in_image_box_214_977_1009_1117.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-09-22T02%3A19%3A42Z%2F-1%2F%2Fcee84cb6572623e2e529029416fc3ae38dff39aa064a66119700ee0617f31ebd" alt="Image" width="64%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图 3：层 Hopfield 允许两个集合 $\mathbf{R}$（☐）和 $\mathbf{Y}$（☐）之间的联想。它可以集成到传播向量集合的深度网络中。霍普菲尔德记忆由来自输入或前几层的集合填充。输出是一个向量集合 $\mathbf{Z}$（☐）。</div> </div>


(2) 层 HopfieldPooling，适用于通过存储的（键）模式 Y 传播模式的网络。该层对从前几层或输入中的查询获得的集合 Y 执行池化或汇总。HopfieldPooling 层的记忆由来自输入或前几层的集合填充。HopfieldPooling 层使用查询在记忆（存储的集合）中搜索模式。如果有更多模式与某个特定搜索模式（查询）相似，则结果是这些模式的平均。每一层的状态（查询）模式是静态的且可以学习。多个查询为下一层提供一个集合，其中每个查询对应集合中的一个元素。因此，层 HopfieldPooling 支持固定模式搜索、池化操作以及类似 LSTM 或 GRU 的记忆。当需要在数据中识别特定模式时，通常需要静态模式功能。

单个 HopfieldPooling 层即可实现多示例学习。静态状态（查询）

模式与键中的位置编码相结合，可以执行池化操作。位置编码可以是二维的，此时可以像卷积神经网络（CNN）中那样构造标准卷积滤波器。HopfieldPooling 层可以替代池化、平均、LSTM 和置换等变层。见图 4。层 HopfieldPooling 被用于多示例学习任务的实验，例如实验中的免疫组库分类。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//9eafe72e-d819-48bf-aa9c-b037a0d7822e/markdown_0/imgs/img_in_image_box_213_317_1008_460.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-09-22T02%3A19%3A32Z%2F-1%2F%2F371fc48aa9c9806fe04dfab312f1da55685e5cd3b122598795f5a98386bec96a" alt="Image" width="64%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图 4：层 HopfieldPooling 支持对集合的池化或汇总，这些集合来自输入或前几层。输入 $\mathbf{Y}$（ $\blacksquare$）可以是集合或序列。每一层的查询模式是静态的且可以学习。输出是一个向量集合 $\mathbf{Z}$（ $\blacksquare$），其中向量数目等于查询模式的数目。层 HopfieldPooling 可以实现多示例学习。</div> </div>


(3) 层 HopfieldLayer，适用于通过状态（查询）模式 R 传播一个向量或一组向量的网络。查询 R 可以是输入向量，或由前几层输出计算得到的查询。HopfieldLayer 层的记忆由一个固定集合填充，该集合可以是训练集、参考集、原型集或学习到的集合（学习到的矩阵）。存储的（键）模式是静态的且可以学习。如果训练集被存储在记忆中，那么每一层基于前几层的查询结果构造一组新的查询。存储的模式可以由训练集或参考集初始化，然后进行学习，此时它们会偏离训练集。存储的模式可以解释为从状态（查询）到具有 softmax 激活函数的隐神经元的权重（Krotov & Hopfield, 2020）。层 HopfieldLayer 可以替代全连接层，见图 5。单个 HopfieldLayer 层还支持类似支持向量机（SVM）的方法、类似 k 近邻的方法、类似学习向量量化的方法以及模式搜索。对于分类，原始数据  $y_i = (z_i, t_i)$ 可以是输入  $z_i$ 与目标  $t_i$ 的拼接。此时，矩阵  $W_K$ 和  $W_V$ 可以设计为：在 softmax 内部使用输入  $z_i$，在 softmax 外部使用目标  $t_i$。因此，softmax 基于查询与输入之间的相似度为目标向量提供加权平均。SVM 模型、k 近邻和学习向量量化也可以视为目标的加权平均。Transformer 的编码器-解码器注意力层就是一个 HopfieldLayer 层，其记忆由编码器输出集合填充。在药物设计基准数据集的实验中，层 HopfieldLayer 被应用并与其他机器学习方法进行了比较。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//9eafe72e-d819-48bf-aa9c-b037a0d7822e/markdown_0/imgs/img_in_image_box_213_1086_1008_1236.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-09-22T02%3A19%3A33Z%2F-1%2F%2F07869f4d9f6212f0d61a2622e311fdcdf14f6ee76c82bfefc8343430a17185dc" alt="Image" width="64%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图 5：层 HopfieldLayer 支持对训练集、参考集、原型集或学习到的集合（学习到的矩阵）进行多次查询。每一层的查询由前几层的结果计算得到。输入是一个向量集合  $\pmb{R}$（■）。输出也是一个向量集合  $\pmb{Z}$（■），其中输出向量的数目等于输入向量的数目。层 HopfieldLayer 可以实现 SVM 模型、k 近邻和 LVQ。</div> </div>


新型霍普菲尔德层的附加功能。关于能量、收敛和存储性质的洞见为所有新型霍普菲尔德层提供了附加功能：i) 多次更新

以控制寻找不动点的精确程度，且不需要额外参数。ii) 可变  $\beta$，用于决定不动点的类型，例如亚稳态的规模。变量  $\beta$ 控制对多少个模式进行平均。正如实验中观察到的，该变量与学习率结合时对引导学习动态具有重要意义。参数  $\beta$ 支配不动点动态，也可以被学习。iii) 通过联想空间的维度控制存储容量。对于实例数量巨大的任务（如免疫组库分类实验），存储容量可能很重要。iv) 模式归一化，像 layernorm 一样，通过模式的范数和偏移来控制不动点动态。更多细节见附录 A.6 节。

## 4 实验

我们展示了所提出的霍普菲尔德层可以成功应用于广泛的任务。这些任务来自自然语言处理，包含多示例学习问题、一组小规模分类任务以及药物设计问题。

Transformer 和 BERT 模型分析。Transformer 和 BERT 模型可以由层 Hopfield 实现。霍普菲尔德网络的不动点类型由模式  $\boldsymbol{x}_i$ 与其他模式的分离程度决定。(a) 全局不动点：模式与其他模式没有分离，(b) 接近单个模式的不动点：模式与其他模式分离，(c) 亚稳态：某些模式彼此相似且与所有其他向量分离良好。我们观察到，Transformer 和 BERT 模型的注意力头主要处于亚稳态，可分为四类：(I) 对非常大量模式求平均（非常大的亚稳态或不动点 (a)），(II) 对大量模式求平均（大亚稳态），(III) 对中等数量模式求平均（中等亚稳态），(IV) 对少量模式求平均（小亚稳态或不动点 (c)）。为了分析亚稳态，我们计算了 softmax 值求和达到 0.90 所需的最小数目  $k$。因此， $k$ 表示亚稳态的规模。为了确定某个注意力头主要工作在四类中的哪一类，我们计算了  $k$ 在序列上的分布。具体来说，对于  $N$ 个 token 和作为分布中位数的  $\tilde{k}$，如果  $1/2N < \tilde{k}$，则该头被归类为工作在 (I) 类；如果  $1/8N < \tilde{k} \leq 1/2N$，则工作在 (II) 类；如果  $1/32N < \tilde{k} \leq 1/8N$，则工作在 (III) 类；如果  $\tilde{k} \leq 1/32N$，则工作在 (IV) 类。我们按照这些工作类别分析了来自 Hugging Face Inc.（Wolf et al., 2019）的预训练 BERT 模型。附录图 A.3 描绘了预训练 bert-base-cased 模型的分布（其他模型见附录 A.5.1.4 节）。工作类别 (II)（大亚稳态）和 (IV)（小亚稳态）常见于中间层。工作类别 (I)（对非常大量模式求平均）在较低层中大量存在。其他研究也报道了类似的观察（Toneva & Wehbe, 2019a;b; Tay et al., 2020）。工作类别 (III)（中等亚稳态）在最后几层中占主导。

多示例学习数据集。对于多示例学习（MIL）（Dietterich et al., 1997），我们通过层 HopfieldPooling 将新的霍普菲尔德网络集成到深度学习架构中。近来，深度学习方法已被应用于 MIL 问题（Ilse et al., 2018），但在许多数据集上的性能仍缺乏提升。因此，MIL 数据集仍然是一个有趣的挑战，而配备记忆的霍普菲尔德层是一种有前景的方法。

• 免疫组库分类。第一个 MIL 任务是免疫组库分类，其中使用了带有 HopfieldPooling 的深度学习架构（DeepRC）（Widrich et al., 2020a;b）。免疫组库分类（Emerson et al., 2017）通常需要从大量序列（组库）中提取出少数能指示相应免疫状态的模式。这些数据集每个免疫组库包含约 300,000 个实例，这是迄今为止规模最大的多示例学习实验之一（Carbonneau et al., 2018）。大多数 MIL 方法因实例数量庞大而失败。该实验包含真实世界和模拟数据集。模拟数据集通过将低频序列基序（Akbar et al., 2019; Weber et al., 2020）植入模拟或实验观测的免疫受体序列中生成。DeepRC 的性能与其他机器学习方法进行了比较：(i) 已知基序，(ii) 使用 k-mer 和 MinMax 或 Jaccard 核的 SVM，(iii) 使用 k-mer 的 K 近邻（KNN），(iv) 使用 k-mer 的逻辑回归，(v) 使用 k-mer 的负担检验（burden test），以及 (vi) 逻辑多



<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>方法</td><td style='text-align: center; word-wrap: break-word;'>tiger</td><td style='text-align: center; word-wrap: break-word;'>fox</td><td style='text-align: center; word-wrap: break-word;'>elephant</td><td style='text-align: center; word-wrap: break-word;'>UCSB</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Hopfield（本文）</td><td style='text-align: center; word-wrap: break-word;'>$91.3 \pm 0.5$</td><td style='text-align: center; word-wrap: break-word;'>$64.05 \pm 0.4$</td><td style='text-align: center; word-wrap: break-word;'>$94.9 \pm 0.3$</td><td style='text-align: center; word-wrap: break-word;'>$89.5 \pm 0.8$</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Path encoding (Küçükası &amp; Baydoğan, 2018)</td><td style='text-align: center; word-wrap: break-word;'>$91.0 \pm 1.0^{a}$</td><td style='text-align: center; word-wrap: break-word;'>$71.2 \pm 1.4^{a}$</td><td style='text-align: center; word-wrap: break-word;'>$94.4 \pm 0.7^{a}$</td><td style='text-align: center; word-wrap: break-word;'>$88.0 \pm 2.2^{a}$</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>MInD (Cheplygina et al., 2016)</td><td style='text-align: center; word-wrap: break-word;'>$85.3 \pm 1.1^{a}$</td><td style='text-align: center; word-wrap: break-word;'>$70.4 \pm 1.6^{a}$</td><td style='text-align: center; word-wrap: break-word;'>$93.6 \pm 0.9^{a}$</td><td style='text-align: center; word-wrap: break-word;'>$83.1 \pm 2.7^{a}$</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>MILES (Chen et al., 2006)</td><td style='text-align: center; word-wrap: break-word;'>$87.2 \pm 1.7^{b}$</td><td style='text-align: center; word-wrap: break-word;'>$73.8 \pm 1.6^{a}$</td><td style='text-align: center; word-wrap: break-word;'>$92.7 \pm 0.7^{a}$</td><td style='text-align: center; word-wrap: break-word;'>$83.3 \pm 2.6^{a}$</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>APR (Dietterich et al., 1997)</td><td style='text-align: center; word-wrap: break-word;'>$77.8 \pm 0.7^{b}$</td><td style='text-align: center; word-wrap: break-word;'>$54.1 \pm 0.9^{b}$</td><td style='text-align: center; word-wrap: break-word;'>$55.0 \pm 1.0^{b}$</td><td style='text-align: center; word-wrap: break-word;'>—</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Citation-kNN (Wang, 2000)</td><td style='text-align: center; word-wrap: break-word;'>$85.5 \pm 0.9^{b}$</td><td style='text-align: center; word-wrap: break-word;'>$63.5 \pm 1.5^{b}$</td><td style='text-align: center; word-wrap: break-word;'>$89.6 \pm 0.9^{b}$</td><td style='text-align: center; word-wrap: break-word;'>$70.6 \pm 3.2^{a}$</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>DD (Maron &amp; Lozano-Pérez, 1998)</td><td style='text-align: center; word-wrap: break-word;'>$84.1^{b}$</td><td style='text-align: center; word-wrap: break-word;'>$63.1^{b}$</td><td style='text-align: center; word-wrap: break-word;'>$90.7^{b}$</td><td style='text-align: center; word-wrap: break-word;'>—</td></tr></table>

<div style="text-align: center;"><div style="text-align: center;">表 1：MIL 数据集 Tiger、Fox、Elephant 和 UCSB 乳腺癌在 AUC 上的结果。除第一行外，所有方法的结果取自  ${}^{a}$（Küçükəscı & Baydoğan, 2018）或  ${}^{b}$（Carbonneau et al., 2016），取决于哪篇报告了更高的 AUC。</div> </div>


实例学习（lMIL）。在真实世界数据集上，DeepRC 取得了  $0.832 \pm 0.022$ 的 AUC，紧随其后的是使用 MinMax 核的 SVM（AUC  $0.825 \pm 0.022$）和负担检验（AUC  $0.699 \pm 0.041$）。在所有数据集上，DeepRC 在平均 AUC 方面优于所有竞争方法（Widrich et al., 2020a; b）。

• MIL 基准数据集。我们将霍普菲尔德层应用于更多 MIL 数据集（Ise et al., 2018; Küçükasıçı & Baydoğan, 2018; Cheplygina et al., 2016）：用于图像标注的 Elephant、Fox 和 Tiger（Andrews et al., 2003）。这些数据集由 Corel 数据集中经过预处理和分割的彩色图像组成。一幅图像由一组片段（或斑块）组成，每个片段由颜色、纹理和形状描述子刻画。这些数据集各有 100 个正例和 100 个负例图像。负例图像是从其他动物的照片池中随机抽取的。Elephant 包含 1,391 个实例和 230 个特征，Fox 包含 1,320 个实例和 230 个特征，Tiger 包含 1,220 个实例和 230 个特征。此外，我们使用 UCSB 乳腺癌分类数据集（Kandemir et al., 2014），它包含 58 个输入对象上的 2,002 个实例。一个实例代表癌变或正常组织的组织病理学图像的一个斑块。这里使用层 HopfieldPooling，它通过提取对两类之一具有指示性的实例的平均值，来计算每个输入对象的表示。层 HopfieldPooling 的输入是嵌入实例的集合 Y。一个可训练但固定的状态（查询）模式 Q 用于对类别指示性实例求平均。这种平均使得可变大小的包（bag）能够被压缩为固定大小的表示，以区分不同的包。更多细节见附录 A.5.2 节。我们的方法创造了新的最优结果，并在数据集 Tiger、Elephant 和 UCSB 乳腺癌上优于其他方法（Küçükası & Baydoğan, 2018; Carbonneau et al., 2016）（见表 1）。

UCI 基准集合。迄今为止，深度学习在小数据集上表现不佳。然而，霍普菲尔德网络在处理小数据集方面很有前景，因为它们可以存储训练数据点或其表示，以执行基于相似度、最近邻或学习向量量化的方法。因此，我们在 UC Irvine（UCI）机器学习库中用于基准测试监督学习方法的小数据集上测试霍普菲尔德层 Hopfield（Fernández-Delgado et al., 2014; Wainberg et al., 2016; Khan et al., 2018），这些数据集也用于测试前馈神经网络（Klambauer et al., 2017a; Wu et al., 2018），我们的霍普菲尔德网络可以在其中利用其记忆。该集合中的全部 121 个数据集在规模、特征数量和难度上差异很大（Fernández-Delgado et al., 2014），因此在 Klambauer et al. (2017a) 中被划分为 75 个样本数少于 1,000 的"小数据集"和 45 个样本数大于等于 1,000 的"大数据集"。在 75 个小数据集上，随机森林（RF）和支持向量机（SVM）精度很高，而在大数据集上，深度学习方法和神经网络处于领先（Klambauer et al., 2017a; Wu et al., 2018）。我们通过层 Hopfield Layer 应用现代霍普菲尔德网络，其中自归一化网络（SNN）将输入向量映射为 Y 和 R。Hopfield Layer 的输出 Z 进入 softmax 输出。我们将现代霍普菲尔德网络与深度



<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>方法</td><td style='text-align: center; word-wrap: break-word;'>平均排名差值</td><td style='text-align: center; word-wrap: break-word;'>p 值</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Hopfield（本文）</td><td style='text-align: center; word-wrap: break-word;'>-3.92</td><td style='text-align: center; word-wrap: break-word;'>—</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>SVM</td><td style='text-align: center; word-wrap: break-word;'>-3.23</td><td style='text-align: center; word-wrap: break-word;'>0.15</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>SNN</td><td style='text-align: center; word-wrap: break-word;'>-2.85</td><td style='text-align: center; word-wrap: break-word;'>0.10</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>RandomForest</td><td style='text-align: center; word-wrap: break-word;'>-2.79</td><td style='text-align: center; word-wrap: break-word;'>0.05</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>...</td><td style='text-align: center; word-wrap: break-word;'>...</td><td style='text-align: center; word-wrap: break-word;'>...</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Stacking</td><td style='text-align: center; word-wrap: break-word;'>8.73</td><td style='text-align: center; word-wrap: break-word;'>1.2e-11</td></tr></table>

<div style="text-align: center;"><div style="text-align: center;">表 2：UCI 基准 75 个小数据集上的结果，以相对于平均排名的差值给出。</div> </div>


方法（如 SNN、resnet）、RF、SVM、boosting、bagging 以及 Fernández-Delgado et al. (2014) 的许多其他机器学习方法进行了比较。由于每种方法都包含多个变体和实现，我们采用了 Klambauer et al. (2017a) 定义的方法组和代表方法。对每个数据集计算了方法的排名，见表 2。我们发现霍普菲尔德网络在小数据集上优于所有其他方法，在 10 个数据集上创造了新的最优结果。除前三种紧随其后的方法外，差异均显著（Wilcoxon 符号秩检验）。详见附录 A.5.3 节。

药物设计基准数据集。我们在四个药物设计数据集上测试霍普菲尔德层 HopfieldLayer。这些数据集代表了药物设计中建模任务的四个主要领域，具体是为预测以下内容开发精确模型：a) 新的抗病毒药物（HIV），由药物治疗项目（DTP）AIDS 抗病毒筛选；b) 新的蛋白质抑制剂，具体为 Subramanian et al. (2016) 的人源  $\beta$-分泌酶（BACE）抑制剂；c) 代谢效应，如血脑屏障通透性（BBBP）（Martins et al., 2012）；d) 来自副作用资源库（SIDER）的化合物副作用（Kuhn et al., 2016）。我们应用霍普菲尔德层 HopfieldLayer，其中训练数据用作存储的模式 Y，输入向量用作状态模式 R，相应的训练标签用于投影霍普菲尔德层的输出  $YW_V$。我们的 Hopfield Layer 架构在预测 SIDER 上的副作用  $0.672 \pm 0.019$ 以及预测  $\beta$-分泌酶 BACE  $0.902 \pm 0.023$ 方面达到了最优性能。详情见附录表 A.5。

结论。我们提出了一种具有连续状态的现代霍普菲尔德网络及其相应的新更新规则。该网络可以存储指数级多的模式，一次更新即可检索出模式，并且检索误差呈指数级小。我们分析了 BERT 模型的注意力头。新的现代霍普菲尔德网络已作为层集成到深度学习架构中，以支持对原始输入数据、中间结果或学习到的原型的存储与访问。这些霍普菲尔德层开启了超越全连接、卷积或循环网络的新深度学习方式，并提供了池化、记忆、联想和注意力机制。为神经网络层配备记忆的霍普菲尔德层，在所考虑的四项多示例学习问题中的三项、免疫组库分类以及两个药物设计数据集上刷新了最优结果。在 UCI 基准集合的小规模分类任务上，它们在不同机器学习方法中取得了最好的结果。

## 致谢

ELLIS Unit Linz、LIT AI Lab 和机器学习研究所得到了 Land Oberösterreich 的资助，LIT 项目 DeepToxGen（LIT-2017-3-YOU-003）和 AI-SNN（LIT-2018-6-YOU-214）、Medical Cognitive Computing Center（MC3）、Janssen Pharmaceutica、UCB Biopharma、Merck Group、Audi.JKU Deep Learning Center、Audi Electronic Venture GmbH、TGW、Primal、S3AI（FFG-872172）、Silicon Austria Labs（SAL）、Anyline、FILL、EnliteAI、Google Brain、ZF Friedrichshafen AG、Robert Bosch GmbH、TÜV Austria、DCS 以及 NVIDIA Corporation 的支持。IARAI 由 Here Technologies 支持。


