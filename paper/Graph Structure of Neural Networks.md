# 神经网络的图结构

Jiaxuan You $^{1}$ Jure Leskovec $^{1}$ Kaiming He $^{2}$ Saining Xie $^{2}$

## 摘要

神经网络通常被表示为神经元之间连接的图。然而，尽管神经网络被广泛使用，目前人们对神经网络的图结构与其预测性能之间的关系仍知之甚少。本文系统性地研究了神经网络的图结构如何影响其预测性能。为此，我们提出了一种全新的、基于图的神经网络表示方法，称为关系图（relational graph），其中神经网络的每一层计算对应于沿图结构进行的一轮消息交换。借助这一表示，我们表明：(1) 关系图存在一个“甜点区（sweet spot）”，处于该区域的神经网络具有显著提升的预测性能；(2) 神经网络性能近似是其关系图的聚类系数与平均路径长度的平滑函数；(3) 我们的发现在许多不同的任务与数据集上保持一致；(4) 该甜点区可以被高效地识别；(5) 性能最优的神经网络，其图结构与真实生物神经网络的结构惊人地相似。我们的工作为神经网络架构的设计以及从总体上理解神经网络开辟了新的方向。

## 1. 引言

深度神经网络由组织成层（layer）的神经元以及它们之间的连接构成。神经网络的架构可以用其“计算图（computational graph）”来刻画：其中神经元表示为节点，有向边连接不同层中的神经元。这种图表示展示了网络如何将信息从其输入神经元、经由隐藏层一路传递并变换到输出神经元（McClelland et al., 1986）。



尽管人们已普遍观察到神经网络的性能依赖于其架构（LeCun et al., 1998; Krizhevsky et al., 2012; Simonyan & Zisserman, 2015; Szegedy et al., 2015; He et al., 2016），但目前仍缺乏关于神经网络准确率与其底层图结构之间关系的系统性理解。这一点对神经架构搜索（neural architecture search）尤为重要，因为如今该领域要在所有可能的连接模式上进行穷举搜索（Ying et al., 2019）。从这个角度看，若干开放性问题随之产生：网络结构与其预测性能之间是否存在系统性的联系？性能良好的神经网络具有哪些结构特征？这些结构特征如何跨任务与数据集泛化？是否存在一种高效的方法来判断某个给定的神经网络是否有前景？

建立这样的联系在科学上和实践上都很重要，因为它将直接影响设计更高效、更准确的架构，也将为执行神经网络的新的硬件架构设计提供指导。理解支撑神经网络的图结构同样会推动深度学习科学的发展。

然而，建立网络架构与其准确率之间的关系并非易事，因为如何将一个神经网络映射为图（以及反过来）尚不明确。自然的选择是使用计算图表示，但它有许多局限：(1) 缺乏通用性：计算图受限于其所允许的图性质，例如这些图必须是有向无环的（DAG）、在层级别上是二分的（bipartite）、在网络级别上是单入单出（single-in-single-out）的（Xie et al., 2019）。这限制了为一般图所开发的丰富工具的使用。(2) 与生物学/神经科学脱节：生物神经网络具有远为丰富、更少模板化的结构（Fornito et al., 2013）。在大脑网络中，存在的是信息交换，而不仅仅是单向流动（Stringer et al., 2018）。这类生物学或神经科学模型无法简单地用有向无环图来表示。

在此，我们系统地研究了神经网络的图结构与其预测性能之间的关系。我们提出了一种将神经网络表示为图的新方法，我们称之为关系图（relational graph）。<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//2fda93d7-0c3c-4581-9c88-8b31a3dfb5fb/markdown_1/imgs/img_in_image_box_107_131_1085_417.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-10-06T16%3A51%3A43Z%2F-1%2F%2Fad7d3914da0a1641d49836323902a9da7070d768a013ea5779711b79410a4a84" alt="Image" width="79%" /></div>

<div style="text-align: center;"><div style="text-align: center;">图 1：我们方法的概览。(a) 神经网络的一层可以视为一个关系图，我们将相互交换消息的节点连接起来。(b) 神经网络层与关系图的更多示例。(c) 我们依据图的度量（包括平均路径长度和聚类系数）来探索关系图的设计空间，其中完全图对应于全连接层。(d) 我们将这些关系图转化为神经网络，并研究其预测性能如何依赖于相应关系图的图度量。</div> </div>


我们的关键洞见在于关注消息交换（message exchange），而不仅仅是有向数据流。举一个简单的例子，对于一个固定宽度的全连接层，我们可以将一个输入通道和一个输出通道共同表示为一个节点，而关系图中的一条边表示这两个节点之间的消息交换（图 1(a)）。在这一表述下，通过恰当定义消息交换，我们表明关系图可以表示许多类型的神经网络层（全连接层、卷积层等），同时摆脱计算图的许多约束（如有向、无环、二分、单入单出）。神经网络的一层对应于在关系图上进行的一轮消息交换；要获得深层网络，我们在同一张图上进行若干轮消息交换。我们的新表示使我们能够构建更丰富、更多样的神经网络，并使用网络科学中成熟的工具对它们进行分析（Barabási & Psfai, 2016）。

随后我们设计了一个名为 WS-flex 的图生成器，使我们能够系统地探索神经网络（即关系图）的设计空间。基于神经科学的洞见，我们用关系图的聚类系数和平均路径长度来刻画神经网络（图 1(c)）。此外，我们的框架灵活且通用，因为我们可以将关系图转化为多种神经网络架构，包括多层感知机（MLP）、卷积神经网络（CNN）、ResNet 等，并控制计算预算（图 1(d)）。

使用标准图像分类数据集 CIFAR-10 和 ImageNet，我们对神经网络的架构如何影响其预测性能进行了系统性研究。我们做出了若干重要的实证观察：

- 在受控计算预算下，关系图的一个“甜点区（sweet spot）”使神经网络具有显著提升的性能（5.1 节）。

- 神经网络的性能近似是其关系图的聚类系数与平均路径长度的平滑函数（5.2 节）。

- 我们的发现在许多架构（MLP、CNN、ResNet、EfficientNet）与任务（CIFAR-10、ImageNet）上保持一致（5.3 节）。

- 该甜点区可以被高效地识别。识别一个甜点区只需少量关系图样本以及少量训练轮次（5.4 节）。

- 性能良好的神经网络，其图结构与真实生物神经网络的结构惊人地相似（5.5 节）。

我们的结果对设计神经网络架构、推进深度学习科学以及从总体上增进对神经网络的理解具有启示意义 $^{1}$。

## 2. 作为关系图的神经网络

为了探索神经网络的图结构，我们首先介绍关系图表示的概念及其具体实例。我们展示我们的表示如何在统一的框架下刻画多种神经网络架构。在深度学习的语境中使用图的语言，有助于将这两个世界联系在一起，并为我们的研究奠定基础。

### 2.1. 图上的消息交换

|                    |       固定宽度 MLP        |       可变宽度 MLP        |       ResNet-34       |           ResNet-34-sep            |           ResNet-50            |
| :----------------: | :-------------------: | :-------------------: | :-------------------: | :--------------------------------: | :----------------------------: |
|     节点特征 $x_i$     |   标量：<br>数据的 1 个维度    |      向量：数据的多个维度       |      张量：数据的多个通道       |             张量：数据的多个通道             |           张量：数据的多个通道           |
| 消息函数 $f_i(\cdot)$  |         标量乘法          |     （非方阵）<br>矩阵乘法     |    $3 \times 3$ 卷积    | $3 \times 3$ 深度卷积与 $1 \times 1$ 卷积 | $3 \times 3$ 与 $1 \times 1$ 卷积 |
| 聚合函数 AGG( $\cdot$) | $\sigma(\sum(\cdot))$ | $\sigma(\sum(\cdot))$ | $\sigma(\sum(\cdot))$ |       $\sigma(\sum(\cdot))$        |     $\sigma(\sum(\cdot))$      |
|       轮数 $R$       |        每层 1 轮         |        每层 1 轮         |      带残差连接的 34 轮      |            带残差连接的 34 轮             |          带残差连接的 50 轮           |

<div style="text-align: center;"><div style="text-align: center;">表 1：用关系图语言表述的多种神经网络架构。这些架构通常被实现为完全关系图，而我们为这些架构系统地探索了更多的图结构。</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//2fda93d7-0c3c-4581-9c88-8b31a3dfb5fb/markdown_2/imgs/img_in_image_box_107_419_576_559.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-10-06T16%3A51%3A46Z%2F-1%2F%2F480f1ffc3e369fdfcd93f8ca878f82eb828aa07b000d59863511f73e275a3a68" alt="Image" width="38%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图 2：将一个 4 节点关系图转化为 4 层 65 维 MLP 的示例。我们高亮展示了节点 $x_{1}$ 的消息交换。使用 $x_{i}, f_{i}(\cdot)$、AGG($\cdot$) 与 $R$ 的不同定义（表 1 中所定义的），关系图可以被转化为多种神经网络架构。</div> </div>


我们首先从图的角度重新审视神经网络的定义。我们定义图 $G =$
 $(\mathcal{V}, \mathcal{E})$，其节点集为 $\mathcal{V} = \{v_1, ..., v_n\}$，边集为 $\mathcal{E} \subseteq \{(v_i, v_j) | v_i, v_j \in \mathcal{V}\}$。我们假设每个节点 $v$ 具有一个节点特征标量/向量 $x_v$。

当一个图 G 与神经元之间的消息交换相关联时，我们称之为关系图。具体而言，一次消息交换由一个消息函数（其输入为节点的特征，输出为一条消息）和一个聚合函数（其输入为一组消息，输出为更新后的节点特征）来定义。在每一轮消息交换中，每个节点向其邻居发送消息，并聚合来自邻居的入消息。每条消息在每条边上通过消息函数 $f(\cdot)$ 进行变换，随后在每个节点处通过聚合函数 $AGG(\cdot)$ 进行聚合。假设我们进行 R 轮消息交换，那么节点 v 的第 r 轮消息交换可以描述为

$$  \mathbf{x}_{v}^{(r+1)}=\operatorname{A G G}^{(r)}(\{f_{v}^{(r)}(\mathbf{x}_{u}^{(r)}),\forall u\in N(v)\})   \tag*{(1)}$$

其中 $u, v$ 是 $G$ 中的节点，$N(v) = \{u|v \vee (u, v) \in \mathcal{E}\}$ 是节点 $v$ 的邻域（我们假设所有节点都含有自环），$\mathbf{x}_v^{(r)}$ 是输入节点特征，$\mathbf{x}_v^{(r+1)}$ 是节点 $v$ 的输出节点特征。注意，这种消息交换可以定义在任意图 $G$ 上；为简单起见，本文只考虑无向图。

式 1 给出了消息交换的一般定义。在本节的余下部分，我们讨论如何将这一一般性的消息交换定义实例化为不同的神经网络架构。我们在表 1 中总结了不同的实例，并在图 2 中给出了一个将其实例化为 4 层 65 维 MLP 的具体示例。



### 2.2. 作为关系图的固定宽度 MLP

多层感知机（MLP）由若干层计算单元（神经元）组成，其中每个神经元对标量输入执行加权求和并输出，随后施加某种非线性。假设 MLP 的第 $r$ 层以 $\mathbf{x}^{(r)}$ 为输入、以 $\mathbf{x}^{(r+1)}$ 为输出，则一个神经元计算：

$$  x_{i}^{(r+1)}=\sigma(\sum_{j}w_{ij}^{(r)}x_{j}^{(r)})   \tag*{(2)}$$

其中 $w_{ij}^{(r)}$ 是可训练权重，$x_j^{(r)}$ 是输入 $\mathbf{x}^{(r)}$ 的第 $j$ 个维度（$i.e., \mathbf{x}^{(r)} = (x_1^{(r)}, ..., x_n^{(r)})$），$x_i^{(r+1)}$ 是输出 $\mathbf{x}^{(r+1)}$ 的第 $i$ 个维度，$\sigma$ 是非线性函数。

让我们考虑所有层的输入与输出 $x^{(r)}$（$1 \leq r \leq R$）都具有相同特征维度的特殊情况。在这种情况下，一个全连接的固定宽度 MLP 层可以用一个完全关系图来表示，其中每个节点 $x_i$ 连接到所有其他节点 $\{x_1, \ldots, x_n\}$，即每个节点 $v$ 的邻域集为 $N(v) = \mathcal{V}$。此外，全连接的固定宽度 MLP 层具有一种特殊的消息交换定义，其中消息函数为 $f_i(x_j) = w_{ij}x_i$，聚合函数为 $\text{AGG}(\{x_i\}) = \sigma(\sum\{x_i\})$。

上述讨论表明，固定宽度 MLP 可以被视为一个带有特殊消息交换函数的完全关系图。因此，固定宽度 MLP 只是一个远为一般的模型族中的一个特例，在这个更一般的模型族中，消息函数、聚合函数，以及最重要的是关系图结构都可以变化。

这一洞见使我们能够将固定宽度 MLP 从使用完全关系图推广到使用任意一般的关系图 G。基于式 1 中消息交换的一般定义，我们有：

$$  x_{i}^{(r+1)}=\sigma(\sum_{j\in N(i)}w_{i j}^{(r)}x_{j}^{(r)})   \tag*{(3)}$$

其中 $i, j$ 是 $G$ 中的节点，$N(i)$ 由 $G$ 定义。### 2.3. 作为关系图的一般神经网络

式 3 中的图视角为将固定宽度 MLP 表示为关系图奠定了基础。在本节中，我们讨论如何进一步将关系图推广到一般神经网络。

**作为关系图的可变宽度 MLP。** 一般神经网络的一个重要设计考量是层宽度在整个网络中往往会变化。例如，在 CNN 中，一种常见做法是在空间下采样后将层宽度（特征通道数）加倍。为了表示层宽度可变的神经网络，我们将节点特征从标量 $x_i^{(r)}$ 推广为向量 $\mathbf{x}_i^{(r)}$，它由 MLP 输入 $\mathbf{x}^{(r)}$ 的若干维度组成，即 $\mathbf{x}^{(r)} = \text{CONCAT}(\mathbf{x}_1^{(r)}, ..., \mathbf{x}_n^{(r)})$；并将消息函数 $f_i(\cdot)$ 从标量乘法推广为矩阵乘法：

$$  \mathbf{x}_{i}^{(r+1)}=\sigma(\sum_{j\in N(i)}\mathbf{W}_{ij}^{(r)}\mathbf{x}_{j}^{(r)})   \tag*{(4)}$$

其中 $\mathbf{W}_{ij}^{(r)}$ 是权重矩阵。此外，我们允许：(1) 同一节点在不同层中，$\mathbf{x}_i^{(r)}$ 与 $\mathbf{x}_i^{(r+1)}$ 可以具有不同的维度；(2) 在同一层内，图中不同的节点 $\mathbf{x}_i^{(r)}$ 与 $\mathbf{x}_j^{(r)}$ 可以具有不同的维度。这一推广后的定义导出了神经网络的灵活图表示，因为我们可以在不同层之间复用同一张关系图，且层宽度可以任意。假设我们用一个 $n$ 节点关系图来表示一个 $m$ 维的层，则 $m \bmod n$ 个节点各有 $\lfloor m/n \rfloor + 1$ 个维度，其余 $n - (m \bmod n)$ 个节点各有 $\lfloor m/n \rfloor$ 个维度。例如，如果我们用一个 4 节点关系图来表示一个 2 层神经网络。若第一层的宽度为 5，而第二层的宽度为 9，则这 4 个节点在第一层中的维度为 $\{2, 1, 1, 1\}$，在第二层中为 $\{3, 2, 2, 2\}$。

注意，在此定义下，关系图的节点数上限受限于相应神经网络中最窄层的宽度（因为每个节点的特征维度至少为 1）。

**作为关系图的 CNN。** 我们进一步使关系图适用于 CNN，此时输入变为图像张量 $\mathbf{X}^{(r)}$。我们将节点特征的定义从向量 $\mathbf{x}_{i}^{(r)}$ 推广为张量 $\mathbf{X}_{i}^{(r)}$，它由输入图像 $\mathbf{X}^{(r)} = \text{CONCAT}(\mathbf{X}_{1}^{(r)}, ..., \mathbf{X}_{n}^{(r)})$ 的若干通道组成。随后我们用卷积算子推广消息交换的定义。具体而言，

$$  \mathbf{X}_{i}^{(r+1)}=\sigma(\sum_{j\in N(i)}\mathbf{W}_{ij}^{(r)}*\mathbf{X}_{j}^{(r)})   \tag*{(5)}$$

其中 $*$ 是卷积算子，$\mathbf{W}_{ij}^{(r)}$ 是卷积滤波器。在此定义下，广泛使用的稠密卷积同样被表示为完全图。

**作为关系图的现代神经网络架构。** 最后，我们将关系图推广到表示具有更精巧设计的现代神经网络架构。例如，为了表示 ResNet（He et al., 2016），我们保持层间的残差连接不变。为了表示带有瓶颈变换（bottleneck transform）的神经网络（He et al., 2016），关系图交替地施加带有 $3 \times 3$ 与 $1 \times 1$ 卷积的消息交换；类似地，在高效计算的设定中，广泛使用的可分离卷积（Howard et al., 2017; Chollet, 2017）可以视为交替地施加带有 $3 \times 3$ 深度卷积与 $1 \times 1$ 卷积的消息交换。



总体而言，关系图为神经网络提供了一种通用的表示。通过恰当地定义节点特征与消息交换，关系图可以表示多种神经网络架构，如表 1 所总结。

## 3. 探索关系图

在本节中，我们详细描述如何设计并探索第 2 节所定义的关系图空间，以研究神经网络的图结构与其预测性能之间的关系。要取得进展需要三个主要组成部分：(1) 刻画图结构性质的图度量；(2) 能够生成多样图的图生成器；(3) 一种控制计算预算的方法，从而使不同神经网络性能上的差异源于其各异的关系图结构。

### 3.1. 图度量的选择

鉴于图结构的复杂性，图度量常被用于刻画图。在本文中，我们关注一个全局图度量——平均路径长度，以及一个局部图度量——聚类系数。值得注意的是，这两个度量在网络科学（Watts & Strogatz, 1998）和神经科学（Sporns, 2003; Bassett & Bullmore, 2006）中被广泛使用。具体而言，平均路径长度度量任意一对节点之间的平均最短路径距离；聚类系数度量给定节点邻域内节点之间实际存在的边数占它们之间可能存在的边数的比例，并在所有节点上取平均。还有其他可用于分析的图度量，收录于附录中。

### 3.2. 图生成器的设计

给定所选定的图度量，我们旨在使用一个图生成器来生成能够覆盖图度量较大范围的多样图。然而，这样的目标需要精心设计生成器：经典的图生成器只能生成有限类别的图，而近期基于学习的图生成器则被设计用于模仿给定的范例图（Kipf & Welling, 2017; Li et al., 2018; You et al., 2018a;b; 2019a）。

**现有图生成器的局限。** 为说明现有图生成器的局限，我们考察以下经典图生成器：(1) Erdős-Rényi（ER）模型，可以在给定节点数与边数的情况下均匀随机地采样图（Erdős & Rényi, 1960）；(2) Watts-Strogatz（WS）模型，可以生成具有小世界性质的图（Watts & Strogatz, 1998）；(3) Barabási-Albert（BA）模型，可以生成无标度图（Albert & Barabási, 2002）；(4) Harary 模型，可以生成具有最大连通度的图（Harary, 1962）；(5) 规则环形格点图（ring graphs）；(6) 完全图。对于所有类型的图生成器，我们将节点数固定为 64，枚举所有可能的离散参数，并对图生成器的所有连续参数进行网格搜索。在每种参数设置下，我们用不同的随机种子生成 30 张随机图。总计我们生成了 486,000 张 WS 图、53,000 张 ER 图、8,000 张 BA 图、1,800 张 Harary 图、54 张环形图以及 1 张完全图（更多细节见附录）。在图 3 中，我们可以观察到这些经典图生成器所生成的图在平均路径长度与聚类系数空间中的覆盖范围是有限的。

**WS-flex 图生成器。** 在此我们提出 WS-flex 图生成器，它可以生成在图度量上具有宽广覆盖范围的图；值得注意的是，如图 3 所示，WS-flex 图几乎涵盖了上述经典随机生成器所生成的所有图。WS-flex 生成器推广了 WS 模型，其方式是放宽“在随机重连之前所有节点具有相同度”这一约束。具体而言，WS-flex 生成器由节点数 n、平均度 k 和重连概率 p 参数化。边数确定为 $e = \lfloor n \times k/2 \rfloor$。具体地，WS-flex 生成器首先创建一个环形图，其中每个节点连接到 $\lfloor e/n \rfloor$ 个相邻节点；随后生成器随机选取 $e \mod n$ 个节点，并将每个节点连接到一个最近的相邻节点；最后，所有边以概率 p 随机重连。我们使用 WS-flex 生成器在聚类系数与平均路径长度的空间中进行平滑采样，然后为我们的实验子采样了 3942 张图，如图 1(c) 所示。

### 3.3. 控制计算预算

为了比较这些多样图所转化得到的神经网络，重要的是确保所有网络具有大致相同的复杂度，从而使其性能差异源于各自的关系图结构。我们使用 FLOPS（乘加运算次数）作为度量。我们首先计算基线网络实例（即完全关系图）的 FLOPS，并在每个实验中将其作为参考

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//eb664520-0023-4863-993f-6ff797cea504/markdown_1/imgs/img_in_chart_box_614_129_848_297.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-10-06T16%3A51%3A44Z%2F-1%2F%2F269e09f2c28164d1bf789e568d05bd16649a5eea4e957124d3680458c0073e87" alt="Image" width="19%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//eb664520-0023-4863-993f-6ff797cea504/markdown_1/imgs/img_in_chart_box_848_129_1082_295.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-10-06T16%3A51%3A44Z%2F-1%2F%2Ff1d81bb43781a0ba888ddbd1306bfd693c38443c4fe0618155510ca67b93acc7" alt="Image" width="19%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图 3：不同图生成器所生成的图。所提出的图生成器 WS-flex 能够覆盖远为广阔的图设计空间区域。WS（Watts-Strogatz）、BA（Barabási-Albert）、ER（Erdős-Rényi）。</div> </div>


复杂度。如 2.3 节所述，通过将维度或通道划分到互不重叠的节点特征集合中，关系图结构可以被实例化为宽度可变的神经网络。因此，我们可以在不改变关系图结构的情况下，方便地调整神经网络的宽度以匹配参考复杂度（在基线 FLOPS 的 0.5% 以内）。我们在附录中提供了更多细节。

## 4. 实验设置

考虑到我们想要探索的候选图数量庞大（共 3942 张），我们首先在 CIFAR-10 数据集（Krizhevsky, 2009）上研究 MLP 的图结构，该数据集有 5 万张训练图像和 1 万张验证图像。随后我们进一步研究规模更大、更复杂的 ImageNet 分类任务（Russakovsky et al., 2015），它包含 1000 个图像类别、128 万张训练图像和 5 万张验证图像。

### 4.1. 基础架构

对于 CIFAR-10 实验，我们使用一个具有 512 个隐藏单元的 5 层 MLP 作为基础架构。MLP 的输入是 $(32 \times 32 \times 3)$ 图像拉平后的 3072 维向量，输出是 10 维预测。每个 MLP 层都有一个 ReLU 非线性和一个 BatchNorm 层（Ioffe & Szegedy, 2015）。我们使用批大小 128 训练模型 200 个 epoch，采用余弦学习率调度（Loshchilov & Hutter, 2016），初始学习率为 0.1（退火至 0，不重启）。我们用 5 个不同的随机种子训练所有 MLP 模型，并报告平均结果。

对于 ImageNet 实验，我们使用三个 ResNet 系列架构，包括：(1) ResNet-34，它仅由 $3 \times 3$ 卷积的基本块（basic block）组成（He et al., 2016）；(2) ResNet-34-sep，这是我们将 ResNet-34 中所有 $3 \times 3$ 稠密卷积替换为 $3 \times 3$ 可分离卷积的变体（Chollet, 2017）；(3) ResNet-50，它由 $1 \times 1$、$3 \times 3$、$1 \times 1$ 卷积的瓶颈块（bottleneck block）组成（He et al., 2016）。此外，我们使用在<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//eb664520-0023-4863-993f-6ff797cea504/markdown_2/imgs/img_in_chart_box_106_123_324_379.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-10-06T16%3A51%3A46Z%2F-1%2F%2Fa9c718f3f0c21382a884e58603b8382169dcca061aeb568bfc3b17b0c9252a23" alt="Image" width="17%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//eb664520-0023-4863-993f-6ff797cea504/markdown_2/imgs/img_in_chart_box_316_121_447_327.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-10-06T16%3A51%3A46Z%2F-1%2F%2Fcc3688a9448a4bd181bd67f400961dd85bdca939d544a6e3c95ed3ae632b51da" alt="Image" width="10%" /></div>


完全图 Top-1 错误率：$33.34 \pm 0.36$

最佳图 Top-1 错误率：$32.05 \pm 0.14$

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//eb664520-0023-4863-993f-6ff797cea504/markdown_2/imgs/img_in_chart_box_454_127_671_378.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-10-06T16%3A51%3A46Z%2F-1%2F%2F0bd6db6473ade17b2384a3c0156a7e6f989bcb0c459101b1e990487ce8516fd1" alt="Image" width="17%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//eb664520-0023-4863-993f-6ff797cea504/markdown_2/imgs/img_in_chart_box_668_121_796_326.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-10-06T16%3A51%3A46Z%2F-1%2F%2Fe531a24c97e67c9208277c652c2bdb70640fb243995fd52f6ed2a4a5b3823c90" alt="Image" width="10%" /></div>


<div style="text-align: center;"><div style="text-align: center;">e 跨神经网络架构的相关性</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//eb664520-0023-4863-993f-6ff797cea504/markdown_2/imgs/img_in_chart_box_808_128_945_255.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-10-06T16%3A51%3A47Z%2F-1%2F%2F8bded7df0b3716b6cd1f46c0ce1af8959a6d9d672208b44a22a078e084140f7d" alt="Image" width="11%" /></div>


完全图 Top-1 错误率：$25.73 \pm 0.07$

最佳图 Top-1 错误率：$24.57 \pm 0.10$

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//eb664520-0023-4863-993f-6ff797cea504/markdown_2/imgs/img_in_chart_box_950_135_1085_255.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-10-06T16%3A51%3A47Z%2F-1%2F%2F3d30759ebbdcff1639933e6ce4328f6137a553fb911d51fd059b1fbd2fcd64dc" alt="Image" width="11%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//eb664520-0023-4863-993f-6ff797cea504/markdown_2/imgs/img_in_chart_box_807_258_946_378.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-10-06T16%3A51%3A47Z%2F-1%2F%2F9e0a697620fc98f9aa03fbd06d1e87f7572bbcd4cc4dd2f7d6c398e7936f1e5b" alt="Image" width="11%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//eb664520-0023-4863-993f-6ff797cea504/markdown_2/imgs/img_in_chart_box_949_259_1085_379.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-10-06T16%3A51%3A47Z%2F-1%2F%2Fe87b9d55e74741b84ca833c8a1fa1b4b84f053434b4fea544be3e9078a98f360" alt="Image" width="11%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//eb664520-0023-4863-993f-6ff797cea504/markdown_2/imgs/img_in_chart_box_110_391_277_601.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-10-06T16%3A51%3A48Z%2F-1%2F%2F6ddb5050a978b230c4230aecef75604c9e7f4c77c4233fe500077e9f40dcb82e" alt="Image" width="13%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//eb664520-0023-4863-993f-6ff797cea504/markdown_2/imgs/img_in_chart_box_276_400_441_600.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-10-06T16%3A51%3A48Z%2F-1%2F%2F73d1666e245a70dc31015438970f3042b978497af31f222a45d3697059f80ae9" alt="Image" width="13%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//eb664520-0023-4863-993f-6ff797cea504/markdown_2/imgs/img_in_chart_box_438_401_601_600.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-10-06T16%3A51%3A48Z%2F-1%2F%2Fac5fad397681fec36a373b77f4d7b83235131a3731248a58bb18ab4b25766d7d" alt="Image" width="13%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//eb664520-0023-4863-993f-6ff797cea504/markdown_2/imgs/img_in_chart_box_600_402_764_601.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-10-06T16%3A51%3A48Z%2F-1%2F%2Faa253109edd1f1794691ce4cd4bea5f46f5a4414d3ae5a543cf190e191a5dfca" alt="Image" width="13%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//eb664520-0023-4863-993f-6ff797cea504/markdown_2/imgs/img_in_chart_box_761_402_922_600.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-10-06T16%3A51%3A49Z%2F-1%2F%2Fe350b449a89b47fd9ae7479373a12107688a3b19f5774b458a29d516af2519c7" alt="Image" width="13%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//eb664520-0023-4863-993f-6ff797cea504/markdown_2/imgs/img_in_chart_box_922_414_1083_599.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-10-06T16%3A51%3A49Z%2F-1%2F%2F2633df0ce8523e80036bca1c49bd41e960c21e54a3faa8c38248e3de09fa92cb" alt="Image" width="13%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图 4：关键结果。所有实验的计算预算都经过严格控制。每个可视化结果都在至少 3 个随机种子上取平均。具有 $C=1$ 与 $L=1$ 的完全图（右下角）被视为基线。(a)(c) 图度量与神经网络性能的关系。最佳图显著优于基线完全图。(b)(d) 单一图度量与神经网络性能的关系。落入给定范围内的关系图以灰点显示。整体平滑函数由蓝色回归线表示。(e) 跨架构的一致性。图中显示了同一组 52 个关系图在转化为不同神经网络架构时性能的相关性。(f) 所有实验的总结。在不同设置下，最佳关系图（红色叉号）始终优于基线完全图。此外，我们高亮了“甜点区”（红色矩形区域），在其中关系图在统计上不劣于最佳关系图（带有红色叉号的分箱）。CIFAR-10 上 5 层 MLP 的分箱值为所有 $C$ 与 $L$ 落入给定分箱的关系图的平均值。</div> </div>


小计算量机制下取得良好性能的 EfficientNet-B0 架构（Tan & Le, 2019）。最后，我们使用一个具有 $3 \times 3$ 卷积的简单 8 层 CNN。该模型有 3 个阶段，隐藏单元数为 [64, 128, 256]。下采样使用步幅为 2 的卷积。主干（stem）与头部（head）层与 ResNet 相同。我们使用余弦学习率调度、初始学习率 0.1 训练所有 ImageNet 模型 100 个 epoch。ResNet 系列模型的批大小为 256，EfficientNet-B0 为 512。我们用 3 个随机种子训练所有 ImageNet 模型，并报告平均性能。所有基础架构都具有完全关系图结构。参考计算复杂度为：MLP 为 2.89e6 FLOPS，ResNet-34 为 3.66e9 FLOPS，ResNet-34-sep 为 0.55e9 FLOPS，ResNet-50 为 4.09e9 FLOPS，EfficientNet-B0 为 0.39e9 FLOPS，8 层 CNN 为 0.17e9 FLOPS。在 NVIDIA Tesla V100 GPU 上训练一个 MLP 模型大约需要 5 分钟，而在 8 块 Tesla V100 GPU 上以数据并行方式训练一个 ImageNet 上的 ResNet 模型大约需要一天。我们在附录中提供了更多细节。

### 4.2. 使用关系图进行探索

对于所有架构，我们按照表 1 中概述的相应定义，将每张采样得到的关系图实例化为一个神经网络。具体而言，我们将所有稠密层（线性层、$3 \times 3$ 与 $1 \times 1$ 卷积层）替换为其关系图对应物。我们保持输入层与输出层不变，并保持所有其他设计（如下采样、跳跃连接等）原样不动。随后，我们为所有模型匹配参考计算复杂度，如 3.3 节所讨论。

对于 CIFAR-10 的 MLP 实验，我们研究 3.2 节所述的 3942 张 64 节点采样关系图。对于 ImageNet 实验，由于计算成本高昂，我们从这 3942 张图中均匀地子采样了 52 张图。由于 EfficientNet-B0 是一个小模型，其某一层仅有 16 个通道，我们无法复用为其他设置采样的 64 节点图。我们按照第 3 节中相同的流程重新采样了 48 张 16 节点关系图。## 5. 结果

在本节中，我们总结实验结果并讨论我们的关键发现。我们收集所有采样关系图在不同任务与架构上的 top-1 错误率，并记录每张采样图的图度量（平均路径长度 L 与聚类系数 C）。我们将这些结果以图度量与预测性能的热力图形式呈现（图 4(a)(c)(f)）。

### 5.1. 顶级神经网络的甜点区

总体而言，图度量与预测性能的热力图（图 4(f)）表明，存在能够超越完全图（右下角像素）基线的图结构。性能最佳的关系图在 CIFAR-10 上可以比完全图基线高出 1.4% 的 top-1 错误率，在 ImageNet 上的模型上则可高出 0.5% 到 1.2%。

值得注意的是，我们发现性能最优的图倾向于聚集在由 $C$ 与 $L$ 定义的空间中的一个甜点区内（图 4(f) 中的红色矩形）。我们按照以下步骤来识别甜点区：(1) 我们将图 4(a) 中的 3942 张图下采样并聚合为 52 个分箱的粗分辨率，每个分箱记录落入该分箱的图的性能；(2) 我们找出平均性能最佳的分箱（图 4(f) 中的红色叉号）；(3) 我们对每个分箱针对性能最佳分箱进行单尾 $t$ 检验，并记录在统计上不显著劣于最佳分箱的分箱（以 $p$ 值 0.05 为阈值）。覆盖这些分箱的最小面积矩形即被可视化为甜点区。对于 CIFAR-10 上的 5 层 MLP，甜点区为 $C \in [0.10, 0.50], L \in [1.82, 2.75]$。

### 5.2. 神经网络性能作为图度量上的平滑函数

在图 4(f) 中，我们观察到神经网络的预测性能近似是其关系图的聚类系数与平均路径长度的平滑函数。将其中一个图度量固定在一个小范围内（$C \in [0.4, 0.6]$，$L \in [2, 2.5]$），我们将网络性能对另一个度量进行可视化（如图 4(b)(d) 所示）。我们使用二次多项式回归来可视化整体趋势。我们观察到，聚类系数和平均路径长度都能指示神经网络的性能，呈现出平滑的 U 形相关性。

### 5.3. 跨架构的一致性

鉴于关系图定义了跨多种神经网络架构的共享设计空间，我们观察到具有某些图度量的关系图无论以何种方式实例化，都可能始终表现良好。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//4ac8762c-10c6-4197-a69b-f7e561a80d48/markdown_0/imgs/img_in_chart_box_616_134_1080_420.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-10-06T16%3A51%3A43Z%2F-1%2F%2F0396d0dcff61dbfb98985de854493208b6355b7dd02e1e7eb95da1dd8d29f39e" alt="Image" width="37%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图 5：快速识别甜点区。左：使用较少关系图样本识别出的甜点区与使用全部 3942 张图识别出的甜点区之间的相关性。右：在中间训练轮次识别出的甜点区与最终轮次（100 个 epoch）识别出的甜点区之间的相关性。</div> </div>


**定性一致性。** 我们在图 4(f) 中直观地观察到，甜点区在不同架构之间大致一致。具体而言，如果我们取各架构甜点区的并集，我们得到 $C \in [0.43, 0.50]$、$L \in [1.82, 2.28]$，这就是跨架构的一致甜点区。此外，图 4(b)(d) 中所示图度量与相应神经网络性能之间的 U 形趋势在视觉上也是一致的。

**定量一致性。** 为进一步量化这种跨任务与架构的一致性，我们选取图 4(f) 热力图中的 52 个分箱，其中分箱值表示图度量落入该分箱范围内的关系图的平均性能。我们绘制这 52 个分箱值在不同任务对之间的相关性，如图 4(e) 所示。我们观察到，具有某些图度量的关系图，其性能在不同任务与架构之间具有相关性。例如，尽管 ResNet-34 的复杂度远高于 5 层 MLP，且 ImageNet 是比 CIFAR-10 更具挑战性的数据集，但一组固定的关系图在两种设置下都会表现相似，其 Pearson 相关系数为 0.658（$p$ 值 < $10^{-8}$）。

### 5.4. 快速识别甜点区

将成千上万张关系图训练至收敛可能在计算上难以承受。因此，我们定量地表明，甜点区可以在远为更低的计算成本下被识别，例如通过采样更少的图并训练更少的轮次。

**需要多少张图？** 以 CIFAR-10 上的 5 层 MLP 为例，我们考虑图 4(f) 中 52 个分箱上的热力图，它是使用 3942 个图样本计算得到的。我们研究是否可以用远为更少的图样本

| 图 | 路径长度 (L) | 聚类系数 (C) | CIFAR-10 误差 (%) |
| :---: | :---: | :---: | :---: |
| 完全图 | 1.00 | 1.00 | 33.34 $\pm$ 0.36 |
| 猫皮层 | 1.81 | 0.55 | 33.01 $\pm$ 0.22 |
| 猕猴视觉皮层 | 1.73 | 0.53 | 32.78 $\pm$ 0.21 |
| 猕猴全皮层 | 2.38 | 0.46 | 32.77 $\pm$ 0.14 |
| 跨神经网络架构的一致甜点区 | 1.82-2.28 | 0.43-0.50 | 32.50 $\pm$ 0.33 |
| 最佳 5 层 MLP | 2.48 | 0.45 | 32.05 $\pm$ 0.14 |

<div style="text-align: center;"><div style="text-align: center;">表 2：顶级人工神经网络可以与生物神经网络相似（Bassett & Bullmore, 2006）。</div> </div>


生物神经网络：猕猴全皮层

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//4ac8762c-10c6-4197-a69b-f7e561a80d48/markdown_1/imgs/img_in_image_box_111_400_578_546.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-10-06T16%3A51%3A44Z%2F-1%2F%2Fcf52e3c7286d4d093ff3536c3b5f2c3da59ead7fa227b47e80ddda2b8d3dba99" alt="Image" width="38%" /></div>


人工神经网络：最佳 5 层 MLP

<div style="text-align: center;"><div style="text-align: center;">图 6：生物（左）与人工（右）神经网络图结构的可视化。</div> </div>


生成一张类似的热力图。具体而言，我们在每个分箱内对图进行子采样，同时确保每个分箱至少有一张图。随后我们计算使用全部 3942 张图与使用子采样的更少图所得到的 52 个分箱值之间的相关性，如图 5（左）所示。我们可以看到，仅使用 52 个样本计算得到的分箱值与使用全部 3942 个图样本计算得到的分箱值之间具有 0.90 的高 Pearson 相关性。这一发现表明，在实践中，进行类似分析所需的图要少得多。

**需要多少个训练轮次？** 以 ImageNet 上的 ResNet-34 为例，我们计算在 52 张采样关系图上，部分训练模型的验证 top-1 错误率与训练完整 100 个 epoch 模型的验证 top-1 错误率之间的相关性，如图 5（右）所示。令人惊讶的是，仅训练 3 个 epoch 后的模型就已具有很高的相关性（0.93），这意味着好的关系图即使在训练早期也能表现良好。这一发现同样重要，因为它表明判断一张关系图是否有前景所需的计算成本可以被大幅降低。

### 5.5. 网络科学与神经科学的联系

**网络科学。** 我们所度量的平均路径长度刻画了信息在整个网络中被交换的效率（Latora & Marchiori, 2001），这与我们对由多轮消息交换构成的关系图的定义相吻合。因此，图 4(b)(d) 中的 U 形相关性可能表明消息交换效率（Sengupta et al., 2013）与学习分布式表示的能力（Hinton, 1984）之间存在一种权衡。

**神经科学。** 我们发现性能最佳的关系图惊人地类似于生物神经网络，如表 2 与图 6 所示。相似性体现在两个方面：(1) 顶级人工神经网络的图度量（$L$ 与 $C$）与生物神经网络高度相似；(2) 借助关系图表示，我们可以将生物神经网络转化为 5 层 MLP，并发现这些网络同样优于基线完全图。虽然我们的发现仍是初步的，但我们的方法为网络科学、神经科学与深度学习之间的跨学科研究开辟了新的可能性。



## 6. 相关工作

**神经网络连接性。** 神经网络连接模式的设计一直聚焦于不同粒度上的计算图：宏观结构，即跨层的连接性（LeCun et al., 1998; Krizhevsky et al., 2012; Simonyan & Zisserman, 2015; Szegedy et al., 2015; He et al., 2016; Huang et al., 2017; Tan & Le, 2019），以及微观结构，即层内的连接性（LeCun et al., 1998; Xie et al., 2017; Zhang et al., 2018; Howard et al., 2017; Dao et al., 2019; Alizadeh et al., 2019）。我们当前的探索聚焦于后者，但同样的方法可以推广到宏观空间。Deep Expander Networks（Prabhu et al., 2018）采用扩展图（expander graph）来生成二分结构。RandWire（Xie et al., 2019）使用现有的图生成器生成宏观结构。然而，这些工作并未探索图结构度量与网络预测性能之间的统计关系。另一项相关工作为 Cross-channel Communication Networks（Yang et al., 2019），其旨在通过消息传递来促进神经元之间的通信，但其中只考虑了完全图结构。

**神经架构搜索。** 在微观层面（Ahmed & Torresani, 2018; Wortsman et al., 2019; Yang et al., 2018）或宏观层面（Zoph & Le, 2017; Zoph et al., 2018）学习连接模式的工作，大多聚焦于改进学习/搜索算法（Liu et al., 2018; Pham et al., 2018; Real et al., 2019; Liu et al., 2019）。NAS-Bench101（Ying et al., 2019）通过枚举规模受限的 DAG（$\leq$ 7 个节点，相比之下我们的工作使用 64 节点图）来定义一个图搜索空间。我们的工作指向了一条新路径：与其在所有可能的连接模式上进行穷举搜索，某些图生成器和图度量可以定义一个平滑的空间，在其中搜索成本可以被显著降低。

## 7. 讨论

**神经网络的分层图结构。** 作为这一方向的第一步，我们的工作聚焦于层级别的图结构。神经网络本质上是分层图（从神经元之间的连接到层、块乃至网络之间的连接），它们构成了比本文所考虑的更为复杂的设计空间。在该空间中进行广泛探索在计算上将难以承受，但我们预期我们的方法与发现能够推广。

**高效实现。** 我们当前的实现使用标准 CUDA 内核，因而依赖权重掩码（weight masking），这导致与基线完全图相比墙钟时间性能更差。然而，我们发现的成果的实际应用并非遥不可及。与我们的工作互补，目前有诸如块稀疏内核（block-sparse kernels）（Gray et al., 2017）与快速稀疏卷积网络（fast sparse ConvNets）（Elsen et al., 2019）等持续的努力，它们可以弥合理论 FLOPS 与实际收益之间的差距。我们的工作也可能为新的硬件架构设计提供指导，例如具有脉冲模式的、受生物启发的架构（Pei et al., 2019）。

**先验 vs. 学习。** 我们目前将关系图表示用作一种结构先验，即在训练全程将图结构硬连线（hard-wire）在神经网络上。已有研究表明，深度 ReLU 神经网络能够自动学习稀疏表示（Glorot et al., 2011）。一个更进一步的问题随之而来：在不施加图先验的情况下，训练一个（全连接）神经网络是否会涌现出某种图结构？

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//4ac8762c-10c6-4197-a69b-f7e561a80d48/markdown_2/imgs/img_in_chart_box_194_804_495_1033.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-10-06T16%3A51%3A46Z%2F-1%2F%2Ff973308b178b465ea7b6ace968cd9944427bc12d24ac69355a3703ba792e0d17" alt="Image" width="24%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图 7：先验 vs. 学习。CIFAR-10 上 5 层 MLP 的结果。我们高亮了作为结构先验时性能最佳的图。此外，我们训练了一个全连接 MLP，并将其学习到的权重可视化为关系图（不同的点是在不同阈值下的图）。学习到的图结构在训练后向“甜点区”移动，但并未弥合差距。</div> </div>


作为一项初步探索，我们“逆向工程”一个训练好的神经网络，并研究其中涌现出的关系图结构。具体而言，我们在 CIFAR-10 上训练一个全连接的 5 层 MLP（与之前实验相同的设置）。随后我们尝试通过以下步骤推断该网络底层的关系图结构：(1) 为了得到关系图中的节点，我们按照 2.2 节所述的流程，将所有隐藏层的权重堆叠起来并将它们分成 64 个节点；(2) 为了得到无向边，权重与其转置相加；(3) 我们计算权重的 Frobenius 范数作为边值；(4) 我们通过以某个阈值对边值进行二值化，得到一个稀疏图结构。



我们在图 7 中展示了不同阈值下提取出的图。正如预期，初始化时提取出的图遵循 E-R 图的模式（图 3（左）），因为权重矩阵是随机 i.i.d. 初始化的。有趣的是，在训练至收敛后，提取出的图不再是 E-R 随机图，而是向我们在第 5 节发现的甜点区移动。注意，这些学习到的图与作为结构先验施加的最佳性能图之间仍然存在差距，这或许可以解释为什么全连接 MLP 的性能较差。

在我们的实验中，我们还发现存在少数特殊情况，此时学习图结构可以更优（即当任务简单且网络容量充裕时）。我们在附录中提供了更多讨论。总体而言，这些结果进一步表明，研究神经网络的图结构对于理解其预测性能至关重要。

**图神经网络（GNN）与一般神经网络架构的统一视角。** 我们将神经网络定义为图上消息交换函数的方式，部分受到 GNN 的启发（Kipf & Welling, 2017; Hamilton et al., 2017; Veličković et al., 2018）。在关系图表示下，我们指出 GNN 是一类特殊的一般神经网络架构，其中：(1) 图结构被视为输入，而非神经网络架构的一部分；因此，(2) 消息函数在所有边上共享，以尊重输入图的不变性。具体而言，回想我们如何将一般神经网络定义为关系图：

$$  \mathbf{x}_{v}^{(r+1)}=\operatorname{A G G}^{(r)}(\{f_{v}^{(r)}(\mathbf{x}_{u}^{(r)}),\forall u\in N(v)\})   \tag*{(6)}$$

对于 GNN，$f_{v}^{(r)}$ 在所有边 $(u, v)$ 上共享，且 $N(v)$ 由输入图数据集提供，而一般神经网络架构则没有这样的约束。

因此，我们的工作为 GNN 与一般神经网络架构设计提供了一个统一的视角，我们希望它能连接这两个社区并激发新的创新。一方面，一般神经网络架构中的成功技术可以自然地引入到 GNN 的设计中，例如可分离卷积（Howard et al., 2017）、组归一化（group normalization）（Wu & He, 2018）以及 Squeeze-and-Excitation 模块（Hu et al., 2018）；另一方面，超越常用范式（即式 6）的新型 GNN 架构（You et al., 2019b; Chen et al., 2019）可能启发更先进的神经架构设计。## 8. 结论

总而言之，我们提出了一个使用关系图表示来分析和理解神经网络的新视角。我们的工作暗示了从研究传统计算架构到研究神经网络图结构的一次新转变。我们表明，其他科学学科（网络科学、神经科学等）所提供的成熟图技术与方法论，可以为理解和设计深度神经网络做出贡献。我们相信，这将是一条富有成果的未来研究途径，能够处理更复杂的情形。

## 致谢

本工作是在 Jiaxuan You 于 Facebook AI Research 实习期间完成的。Jure Leskovec 是 Chan Zuckerberg Biohub 研究员。作者感谢 Alexander Kirillov、Ross Girshick、Jonathan Gomes Selman、Pan Li 的有益讨论。

## 参考文献

Ahmed, K. and Torresani, L. Maskconnect: Connectivity learning by gradient descent. In ECCV, 2018.

Albert, R. and Barabási, A.-L. Statistical mechanics of complex networks. Reviews of modern physics, 74(1):47, 2002.

Alizadeh, K., Farhadi, A., and Rastegari, M. Butterfly transform: An efficient fft based neural architecture design. arXiv preprint arXiv:1906.02256, 2019.

Barabási, A.-L. and Psfai, M. Network science. Cambridge University Press, Cambridge, 2016.

Bassett, D. S. and Bullmore, E. Small-world brain networks. The neuroscientist, 12(6):512–523, 2006.

Chen, Z., Villar, S., Chen, L., and Bruna, J. On the equivalence between graph isomorphism testing and function approximation with gnns. In NeurIPS, 2019.

Chollet, F. Xception: Deep learning with depthwise separable convolutions. In CVPR, pp. 1251–1258, 2017.

Dao, T., Gu, A., Eichhorn, M., Rudra, A., and Ré, C. Learning fast algorithms for linear transforms using butterfly factorizations. In ICML, 2019.

Elsen, E., Dukhan, M., Gale, T., and Simonyan, K. Fast sparse convnets. arXiv preprint arXiv:1911.09723, 2019.

Erdős, P. and Rényi, A. On the evolution of random graphs. Publ. Math. Inst. Hung. Acad. Sci, 5(1):17–60, 1960.

Fornito, A., Zalesky, A., and Breakspear, M. Graph analysis of the human connectome: promise, progress, and pitfalls. Neuroimage, 80:426–444, 2013.

Glorot, X., Bordes, A., and Bengio, Y. Deep sparse rectifier neural networks. In AISTATS, pp. 315–323, 2011.

Hamilton, W., Ying, Z., and Leskovec, J. Inductive representation learning on large graphs. In NeurIPS, 2017.

Gray, S., Radford, A., and Kingma, D. P. Gpu kernels for block-sparse weights. arXiv preprint arXiv:1711.09224, 2017.

Harary, F. The maximum connectivity of a graph. Proceedings of the National Academy of Sciences of the United States of America, 48(7):1142, 1962.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In CVPR, 2016.

Hinton, G. E. Distributed representations. 1984.

Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., and Adam, H. Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861, 2017.

Hu, J., Shen, L., and Sun, G. Squeeze-and-excitation networks. In CVPR, 2018.

Huang, G., Liu, Z., van der Maaten, L., and Weinberger, K. Q. Densely connected convolutional networks. In CVPR, 2017.

Ioffe, S. and Szegedy, C. Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167, 2015.

Kipf, T. N. and Welling, M. Semi-supervised classification with graph convolutional networks. In ICLR, 2017.

Krizhevsky, A. Learning multiple layers of features from tiny images. Technical report, 2009.

Krizhevsky, A., Sutskever, I., and Hinton, G. E. Imagenet classification with deep convolutional neural networks. In NeurIPS, 2012.

Latora, V. and Marchiori, M. Efficient behavior of small-world networks. Physical review letters, 87(19):198701, 2001.

LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.

Li, Y., Vinyals, O., Dyer, C., Pascanu, R., and Battaglia, P. Learning deep generative models of graphs. In ICML, 2018.
