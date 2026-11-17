# 基于动态可扩展网络的终身学习

Jaehong Yoon $^{1,3*}$, Eunho Yang $^{1,3}$, Jeongtae Lee $^{2}$, Sung Ju Hwang $^{1,3}$

KAIST $^{1}$，韩国大田；UNIST $^{2}$，韩国蔚山；AITrics $^{3}$，韩国首尔

{jaehong.yoon, eunhoy, sjhwang82}@kaist.ac.kr, jtlee@unist.ac.kr

## 摘要

我们提出了一种用于终身学习的新型深度网络架构，称之为动态可扩展网络（Dynamically Expandable Network，DEN）。它能够在按顺序训练一系列任务的过程中动态决定自身的网络容量，从而学习到一种紧凑的、任务间相互重叠的知识共享结构。DEN 通过选择性再训练（selective retraining）以在线方式高效训练：每到来一个新任务时仅用必要数量的单元动态扩展网络容量，并通过拆分/复制单元（splitting/duplicating units）并为其打时间戳（timestamping）来有效防止语义漂移（semantic drift）。我们在终身学习场景下的多个公开数据集上验证了 DEN，它不仅显著优于现有的深度网络终身学习方法，而且在参数量大幅减少的情况下达到了与批处理（batch）模型相当的性能。此外，将学到的网络在所有任务上微调后，其性能显著优于批处理模型，这表明即使一开始所有任务都已知，DEN 也可用于估计最优网络结构。

## 1 引言

终身学习（Thrun, 1995），即任务按顺序到来的持续学习问题，是迁移学习中的一个重要课题。终身学习的主要目标是利用早期任务的知识，以便在后续任务的模型上获得更好的性能，或更快的收敛/训练速度。尽管存在许多不同的方法来解决这一问题，我们考虑深度学习下的终身学习，以发挥深度神经网络的能力。幸运的是，对于深度学习而言，知识的存储与迁移可以通过已学得的网络权重以直接的方式完成。学得的权重可以作为已有任务的知识，而新任务只需共享这些权重即可加以利用。

因此，对于深度神经网络，我们可以简单地将终身学习视为在线学习或增量学习的一种特例。执行这种增量学习有多种方式（Rusu et al., 2016; Zhou et al., 2012）。最简单的方式是用新训练数据继续训练网络，从而将网络增量地微调到新任务上。然而，这种简单的网络再训练会使新任务和旧任务的性能都退化。如果新任务与旧任务差异很大，例如先前任务是分类动物图像、而新任务是分类汽车图像，那么在先前任务上学到的特征可能对新任务没有用处。同时，为新任务重新训练的表示可能对旧任务产生不利影响，因为它们可能已经偏离了原本的含义、不再对旧任务最优。例如，描述斑马条纹图案的特征，对于后续诸如条纹 T 恤或围栏等类别的分类任务，其含义可能发生改变，这些类别可能与该特征相匹配并大幅改变其含义。

那么，在深度神经网络的在线/增量学习中，我们如何确保通过网络进行的知识共享对所有任务都有益？近期工作建议，要么使用正则项防止参数值发生剧烈变化、同时仍能为新任务找到好解（Kirkpatrick et al., 2017），要么阻断对旧任务参数的任何修改（Rusu et al., 2016）。我们的策略与这两种方法都不同，因为我们会在每个任务 $t$ 上对网络进行再训练，使得每个新任务只利用并改变先前训练网络中相关的部分，同时在必要时仍允许扩展网络容量。这样，每个任务 $t$ 将使用与先前任务不同的子网络，同时仍与它们共享相当一部分子网络。图 1 展示了我们的模型与现有深度终身学习方法的对比。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//dab9434e-6266-450e-8eb5-a73999577f99/markdown_1/imgs/img_in_image_box_285_122_396_268.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A49%3A14Z%2F-1%2F%2F590277015666704984fcb1dfebaa7828e451823e5114262d58e1fc26979d0fd4" alt="Image" width="9%" /></div>

<div style="text-align: center;"><div style="text-align: center;">(a) 无扩展的再训练</div> </div>

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//dab9434e-6266-450e-8eb5-a73999577f99/markdown_1/imgs/img_in_image_box_520_122_658_267.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A49%3A14Z%2F-1%2F%2F082ae01b331e802c73661863cabc66aab7da1035ae56d0d8ae224c32bfea4d09" alt="Image" width="11%" /></div>

<div style="text-align: center;"><div style="text-align: center;">(b) 无再训练的扩展</div> </div>

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//dab9434e-6266-450e-8eb5-a73999577f99/markdown_1/imgs/img_in_image_box_786_124_900_267.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A49%3A15Z%2F-1%2F%2Ff78aa9a17bb128bca89e4940ab8a6dd0d9231c8fee125a4fca8f0d963a812bea" alt="Image" width="9%" /></div>

<div style="text-align: center;"><div style="text-align: center;">(c) 带扩展的部分再训练</div> </div>

<div style="text-align: center;"><div style="text-align: center;">图 1：概念：(a) 诸如弹性权重巩固 Kirkpatrick et al. (2017) 之类的再训练模型会对先前任务上学到的整个网络进行再训练，同时对其进行正则化以防止偏离原始模型过大。用红色标注的单元和权重表示被再训练的，黑色表示保持固定的。(b) 诸如渐进网络（Rusu et al., 2016）之类的无再训练模型会为新任务 t 扩展网络，同时不对先前任务的网络权重做修改。(c) 我们的 DEN 有选择地对旧网络进行再训练，在必要时扩展其容量，从而在训练过程中动态决定其最优容量。</div> </div>

对于这种具有选择性参数共享和动态层扩展的增量深度学习场景，有若干挑战需要解决。

1) 实现训练的可扩展性与高效性：如果网络容量增长，每个任务的训练成本也会随之增长，因为后续任务将与更大的网络建立连接。因此，我们需要一种方法来保持再训练的计算开销较低。

2) 决定何时扩展网络，以及增加多少神经元：如果旧网络已能充分解释新任务，网络可能不需要扩展规模。另一方面，如果任务与现有任务差异很大，可能需要加入许多神经元。因此，模型需要动态地只加入必要数量的神经元。

3) 防止语义漂移（即灾难性遗忘），即网络在训练过程中偏离初始配置，从而在早期样本/任务上表现出退化的性能。由于我们的方法会对网络进行再训练（即便是部分再训练）以拟合后续学到的任务，并且加入的新神经元可能通过与旧子网络建立连接而对先前任务产生负面影响，因此我们需要一种机制来防止潜在的语义漂移。

为克服这些挑战，我们提出了一种新颖的深度网络模型以及一种高效且有效的增量学习算法，我们将其命名为动态可扩展网络（Dynamically Expandable Networks，DEN）。在终身学习场景中，DEN 最大程度地利用在所有先前任务上学到的网络来高效地学习对新任务的预测，同时在必要时通过增加或拆分/复制神经元来动态地扩展网络容量。我们的方法适用于任何通用的深度网络，包括卷积网络。

我们在多个公开数据集上验证了用于终身学习的增量深度神经网络，其性能与为每个任务单独训练一个网络的模型相当或更好，而只使用了后者 11.9%p – 60.3%p 的参数。此外，将学到的网络在所有任务上微调可获得更好的性能，比批处理模型高出多达 0.05%p – 4.8%p。因此，我们的模型还可用于结构估计，即使在批处理训练可行的更一般设置下，也能在给定网络容量下获得最优性能。

## 2 相关工作

终身学习 终身学习（Thrun, 1995）是持续学习的学习范式，模型从一系列任务中学习，同时将早期任务中获得的知识迁移到后续任务。自 Thrun（1995）提出这一思想以来，由于其数据以流的形式到达的场景（如自动驾驶或机器人智能体的学习）中的实用性，它已被广泛研究。终身学习常被作为在线多任务学习问题来处理，其关注点既在于高效训练，也在于知识迁移。Eaton & Ruvolo（2013）提出了一个在线终身学习框架（ELLA），它基于现有的多任务学习公式（Kumar & Daume III, 2012），通过为每个任务预测器的学习去除对先前任务的依赖并防止再训练先前任务的预测器，从而高效地更新一系列任务的隐参数基。最近，终身学习在深度学习框架中得到研究；由于深度网络的终身学习可以通过简单的再训练直接完成，研究的主要焦点在于克服灾难性遗忘（Kirkpatrick et al., 2017; Rusu et al., 2016; Zenke et al., 2017; Lee et al., 2017）。

防止灾难性遗忘 深度网络的增量学习或终身学习会导致所谓的灾难性遗忘问题，它描述的是为新任务再训练网络致使网络遗忘先前任务所学内容的情况。该问题的一种解决方法是使用正则项，防止新模型偏离旧模型太多，例如 $\ell_2$ 正则项。然而，使用简单的 $\ell_2$ 正则项会阻止模型为新任务学习新知识，从而导致后续任务性能次优。为克服这一局限，Kirkpatrick et al.（2017）提出了弹性权重巩固（Elastic Weight Consolidation，EWC），它通过当前任务的 Fisher 信息矩阵，用先前迭代时的模型参数来正则化每一步的模型参数，从而能为两个任务都找到好解。Zenke et al.（2017）提出了类似方法，但他们的方法在线计算每个突触的巩固程度，并考虑整个学习轨迹而非最终参数值。防止灾难性遗忘的另一种方式是像 Rusu et al.（2016）那样完全阻断对先前网络的任何修改，即在每个学习阶段，用容量固定的子网络扩展网络，该子网络用来自原始网络的输入权重训练，但不向原始网络反向传播。

动态网络扩展 现有工作中很少有探索能在训练过程中动态增加容量的神经网络。Zhou et al.（2012）提出通过为一组损失较高的困难样本加入新神经元来增量训练去噪自编码器，之后再与其他神经元合并以防止冗余。最近，Philipp & Carbonell（2017）提出了一种非参数神经网络模型，它不仅学习最小化损失，还寻找每层能降低损失的最小维度，其假设是每层有无限多个神经元。Cortes et al.（2016）也基于提升（boosting）理论提出了一种能自适应地学习结构和权重以最小化给定损失的网络。然而，这些工作都没有考虑多任务设置，且都涉及迭代地加入神经元（或神经元集合）的过程，而我们的方法只需为每个任务训练网络一次，即可决定加入多少神经元。Xiao et al.（2014）提出了一种为多类分类增量训练网络的方法，其中网络不仅容量增长，还随着新类别的到来形成层次结构。然而，该模型只增长和分叉最顶层的层，而我们的方法可以在任意层增加神经元数量。

## 3 动态可扩展网络的增量学习

我们考虑终身学习场景下深度神经网络的增量训练问题，其中未知数量、训练数据分布未知的任务按顺序到达模型。具体而言，我们的目标是学习 $T$ 个任务序列的模型，$t = 1, \ldots, t, \ldots, T$，其中 $T$ 无界，时刻 $t$ 的任务带有训练数据 $\mathcal{D}_t = \{x_i, y_i\}_{i=1}^N$。注意每个任务 $t$ 可以是单个任务，也可以由一组子任务构成。虽然我们的方法对任何类型的任务都通用，但为简化起见，我们只考虑二分类任务，即对输入特征 $\boldsymbol{x} \in \mathbb{R}^d$ 有 $y \in \{0, 1\}$。终身学习设置中的主要挑战在于，当前时刻 $t$ 无法获得截至 $t - 1$ 的所有先前训练数据集（如果有的话，只能访问先前任务的模型参数）。时刻 $t$ 的终身学习智能体旨在通过求解以下问题来学习模型参数 $\boldsymbol{W}^t$：

$$  \underset{W^{t}}{\mathrm{m i n i m i z e}}\mathcal{L}(\mathbf{W}^{t};\mathbf{W}^{t-1},\mathcal{D}_{t})+\lambda\Omega(\mathbf{W}^{t}),\quad t=1,\ldots.   \tag*{(1)}$$

其中 $\mathcal{L}$ 是任务特定的损失函数，$W^t$ 是任务 $t$ 的参数，$\Omega(W^t)$ 是正则项（例如逐元素 $\ell_2$ 范数）以适当约束我们的模型 $W^t$。对于我们主要关注的神经网络，$W^t = \{W_l\}_{l=1}^L$ 是权重张量。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//dab9434e-6266-450e-8eb5-a73999577f99/markdown_3/imgs/img_in_image_box_301_167_919_320.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A49%3A17Z%2F-1%2F%2Fa092cb192467a078909e29b4e5604f5a24ed332afb57003974c0dae96681faea" alt="Image" width="50%" /></div>

<div style="text-align: center;"><div style="text-align: center;">图 2：动态可扩展网络的增量学习：左：选择性再训练。DEN 首先识别与新任务相关的神经元，并有选择地再训练与它们关联的网络参数。中：动态网络扩展。如果选择性再训练未能将损失降到设定阈值以下，我们以自顶向下的方式扩展网络容量，同时利用组稀疏正则化消除任何不必要的神经元。右：网络拆分/复制。DEN 为每个单元计算漂移量 $\rho_{i}^{t}$，以识别在训练中偏离其原始值过大的单元并将其复制。</div> </div>

为应对终身学习的这些挑战，我们让网络最大程度地利用从先前任务获得的知识，同时在累积的知识本身不足以充分解释新任务时允许其动态扩展容量。图 2 和算法 1 描述了我们的增量学习过程。

算法 1 动态可扩展网络的增量学习

输入：数据集 $\mathcal{D} = (\mathcal{D}_1, \ldots, \mathcal{D}_T)$，阈值 $\tau$、$\sigma$

输出：$W^T$

for $t = 1, \ldots, T$ do

if $t = 1$ then

使用式 2 训练网络权重 $W^1$

else

$W^t = \text{SelectiveRetraining}(W^{t-1})$ {使用算法 2 对先前网络进行选择性再训练}

if $\mathcal{L}_t > \tau$ then

$W^t = \text{DynamicExpansion}(W^t)$ {使用算法 3 扩展网络容量}

$W^t = \text{Split}(W^t)$ {使用算法 4 拆分并复制单元}

在以下小节中，我们详细介绍增量学习算法的各个组成部分：1) 选择性再训练，2) 动态网络扩展，以及 3) 网络拆分/复制。

选择性再训练（Selective Retraining）。 为一系列任务训练模型最朴素的方式是每当新任务到来时再训练整个模型。然而，对于深度神经网络，这种再训练代价非常高昂。因此，我们建议对模型进行选择性再训练，即只再训练受新任务影响的权重。最初（t=1），我们用 $\ell_1$ 正则化训练网络，以促进权重的稀疏性，使每个神经元只与下一层的少数神经元相连：

$$  \underset{\pmb{W}^{t=1}}{\mathrm{m i n i m i z e}}\mathcal{L}(\pmb{W}^{t=1};\mathcal{D}_{t})+\mu\sum_{l=1}^{L}\|\pmb{W}_{l}^{t=1}\|_{1}   \tag*{(2)}$$

其中 $1 \leq l \leq L$ 表示网络的第 $l_{th}$ 层，$W_l^t$ 是第 $l$ 层的网络参数，$\mu$ 是对 $W$ 施加稀疏性的逐元素 $\ell_1$ 范数的正则化参数。对于卷积层，我们对滤波器施加 (2, 1)-范数，以只从上一层选择少数滤波器。

在我们整个增量学习过程中，我们保持 $W^{t-1}$ 稀疏，因此如果将计算集中在与新任务相关的子网络上，就能大幅减少计算开销。为此，当新任务 t 到达模型时，我们首先通过求解以下问题，用神经网络的最高层隐单元拟合一个稀疏线性模型来预测任务 t：

$$  \underset{\boldsymbol{W}_{L,t}^{t}}{\mathrm{minimize}}\mathcal{L}(\boldsymbol{W}_{L,t}^{t};\boldsymbol{W}_{1:L-1}^{t-1},\mathcal{D}_{t})+\mu\|\boldsymbol{W}_{L,t}^{t}\|_{1}   \tag*{(3)}$$

其中 $W_{1:L-1}^{t-1}$ 表示除 $W_{L,t}^{t}$ 之外的所有其他参数的集合。也就是说，我们求解该优化以获得输出单元 $o_{t}$ 与第 L-1 层隐单元之间的连接（将第 $L-1$ 层之前的所有其他参数固定为 $\boldsymbol{W}^{t-1}$）。一旦我们在该层建立起稀疏连接，就可以识别网络中所有受训练影响的单元和权重，同时保持网络中未连接到 $o_{t}$ 的部分不变。具体而言，我们从那些被选中的节点出发在网络上进行广度优先搜索，以识别所有有路径通向 $o_{t}$ 的单元（以及输入特征）。然后，我们只训练所选子网络 $S$ 的权重，记为 $\boldsymbol{W}_{S}^{t}$。

$$  \underset{W_{S}^{t}}{\mathrm{m i n i m i z e}}\mathcal{L}(\boldsymbol{W}_{S}^{t};\boldsymbol{W}_{S^{c}}^{t-1},\mathcal{D}_{t})+\mu\|\boldsymbol{W}_{S}^{t}\|_{2}   \tag*{(4)}$$

由于稀疏连接已经建立$^{1}$，我们使用逐元素 $\ell_{2}$ 正则项。这种部分再训练将带来较低的计算开销，也有助于避免负迁移，因为未被选中的神经元不会受再训练过程的影响。算法 2 描述了选择性再训练过程。

算法 2 选择性再训练

输入：数据集 $\mathcal{D}_t$，先前参数 $W^{t-1}$

输出：网络参数 $W^t$

初始化 $l \leftarrow L - 1$，$S = \{o_t\}$

求解式 3 以获得 $W_L, t^2$

若 $W_{L,t}^t$ 中 $i$ 与 $o_t$ 之间的权重非零，则将神经元 $i$ 加入 $S$。

for $l = L - 1, \ldots, 1$ do

若存在某个神经元 $j \in S$ 使得 $W_{L,ij}^{t-1} \neq 0$，则将神经元 $i$ 加入 $S$。

求解式 4 以获得 $W_S^t$

动态网络扩展（Dynamic Network Expansion）。 当新任务与旧任务高度相关，或从每个任务聚合得到的部分知识足以解释新任务时，仅靠选择性再训练就足以应对新任务。然而，当学到的特征无法准确表示新任务时，需要向网络引入额外的神经元，以容纳新任务所需的特征。一些现有工作（Zhou et al., 2012; Rusu et al., 2016）基于类似思想。然而，它们要么因需要重复前向传播的迭代训练而效率低下（Zhou et al., 2012），要么在每个任务 t 加入固定数量的单元而不考虑任务难度（Rusu et al., 2016），因而在性能和网络容量利用率上都是次优的。

为克服这些局限，我们改为提出一种高效的方法，利用组稀疏正则化（group sparse regularization）为每个任务动态决定在哪些层增加多少神经元，而无需为每个单元重复再训练网络。假设我们用固定数量（记为 $k$）的单元扩展网络的第 $l_{th}$ 层，这会引起两个参数矩阵的扩展：$\boldsymbol{W}_l^t = [\boldsymbol{W}_l^{t-1}; \boldsymbol{W}_l^{\mathcal{N}}]$ 和 $\boldsymbol{W}_{l-1}^t = [\boldsymbol{W}_{l-1}^{t-1}; \boldsymbol{W}_{l-1}^{\mathcal{N}}]$，分别对应输出层和输入层，其中 $\boldsymbol{W}^{\mathcal{N}}$ 是与新增神经元有关的扩展权重矩阵。由于我们并非总是想加入全部 $k$ 个单元（取决于新任务与旧任务的相关性），我们对新增参数执行组稀疏正则化如下：

$$  \underset{\boldsymbol{W}_{l}^{\mathcal{N}}}{\mathrm{minimize}}\mathcal{L}(\boldsymbol{W}_{l}^{\mathcal{N}};\boldsymbol{W}_{l}^{t-1},\mathcal{D}_{t})+\mu\|\boldsymbol{W}_{l}^{\mathcal{N}}\|_{1}+\gamma\sum_{g}\|\boldsymbol{W}_{l,g}^{\mathcal{N}}\|_{2}   \tag*{(5)}$$

其中 $g \in \mathcal{G}$ 是在每个神经元输入权重上定义的组。对于卷积层，我们将每组定义为每个卷积滤波器的激活图。这种组稀疏正则化曾在 Wen et al. (2016) 和 Alvarez & Salzmann (2016) 中用于为整个网络寻找合适的神经元数量，而我们将其应用于部分网络。算法 3 描述了扩展的细节。

在选择性再训练完成后，网络检查损失是否低于某个阈值。若否，则我们在每一层用 $k$ 个神经元扩展其容量并求解式 5。由于式 5 中的组稀疏正则化，从训练中被认为不必要的隐单元（或卷积滤波器）会被整体丢弃。我们期望通过这一动态网络扩展过程，模型能捕捉到先前未被 $W_l^{t-1}$ 表示的、用于最小化残差误差的新特征，同时通过避免加入过多单元来最大化网络容量的利用率。

算法 3 动态网络扩展

输入：数据集 $\mathcal{D}_t$，阈值 $\tau$

执行算法 2 并计算 $\mathcal{L}$

if $\mathcal{L} > \tau$ then

在所有层加入 $k$ 个单元 $h^{\mathcal{N}}$

在所有层求解式 5

for $l = L - 1, \ldots, 1$ do

移除 $h_l^{\mathcal{N}}$ 中无用的单元

网络拆分/复制（Network Split/Duplication）。 终身学习中的一个关键挑战是语义漂移（或灾难性遗忘）问题，它描述的是模型逐渐拟合后续学到的任务、从而遗忘先前任务所学内容，导致这些任务性能退化的问题。防止语义漂移最流行却又最简单的方法是使用 $\ell_{2}$ 正则化，让参数不过度偏离其原始值，如下：

$$  \underset{\pmb{W}^{t}}{\mathrm{m i n i m i z e}}\mathcal{L}(\pmb{W}^{t};\mathcal{D}_{t})+\lambda\|\pmb{W}^{t}-\pmb{W}^{t-1}\|_{2}^{2}   \tag*{(6)}$$

其中 $t$ 是当前任务，$W^{t-1}$ 是为任务 $\{1, \ldots, t-1\}$ 训练的网络权重张量，$\lambda$ 是正则化参数。这种 $\ell_2$ 正则化将通过 $\lambda$ 给定的程度，强制解 $W^t$ 靠近 $W^{t-1}$；若 $\lambda$ 小，则网络会更多地反映新任务而遗忘旧任务；若 $\lambda$ 大，则 $W^t$ 会尽可能保留先前任务学到的知识。除了施加简单的 $\ell_2$ 正则化，也可以用 Fisher 信息对每个元素加权（Kirkpatrick et al., 2017）。尽管如此，若任务数量很大，或后续任务与先前任务在语义上差异很大，可能难以同时为先前任务和新任务找到好解。

在这种情况下，更好的解决方案是拆分神经元，使我们拥有对两个不同任务都最优的特征。在执行式 6 之后，我们测量每个隐单元 $i$ 的语义漂移量 $\rho_{i}^{t}$，即其在 t-1 时刻与 t 时刻输入权重之间的 $\ell_{2}$ 距离。然后，若 $\rho_{i}^{t} > \sigma$，我们认为该特征的含义在训练中发生了显著变化，并将该神经元 i 拆分为两个副本（适当地引入指向和来自该副本的新边）。此操作可以并行地对所有隐单元执行。在神经元复制之后，由于拆分改变了整体结构，网络需要再次求解式 6 以重新训练权重。然而在实践中，由于初始训练给出了合理的参数初始化，这次二次训练通常收敛很快。算法 4 描述了拆分操作的算法。

算法 4 网络拆分/复制

输入：权重 $W^{t-1}$，阈值 $\sigma$

执行式 6 以获得 $\widetilde{W}^{t}$

for 所有隐单元 i do

$\rho_{i}^{t}=\|w_{i}^{t}-w_{i}^{t-1}\|_{2}$

if $\rho_{i}^{t}>\sigma$ then

将 i 复制为 $i^{\prime}$（为 $i^{\prime}$ 引入新边 $w^{\prime}$）

以 $\widetilde{W}^{t}$ 为初始化执行式 6 以获得 $W^{t}$

时间戳推理（Timestamped Inference）。 在网络扩展和网络拆分过程中，我们通过设置 $\{z\}_{j} = t$ 为每个新加入的单元 $j$ 打上时间戳，以记录它被加入网络时的训练阶段 $t$，从而进一步防止由引入新隐单元引起的语义漂移。在推理时，每个任务只使用截至阶段 $t$ 引入的参数，以防止旧任务使用训练过程中加入的新隐单元。这比 Rusu et al. (2016) 中固定截至每个学习阶段所学权重的策略更为灵活，因为早期任务仍可通过被进一步训练（但未被拆分）的单元从后续任务的学习中受益。

## 4 实验

基线和我们的模型。 1) DNN-STL。基础深度神经网络，前馈或卷积，为每个任务单独训练。

2) DNN-MTL。为所有任务一次性训练的基础 DNN。

3) DNN。基础 DNN。所有增量模型都使用 $\ell_{2}$ 正则化。

4) DNN-L2。基础 DNN，其中在每个任务 $t$，$W^{t}$ 初始化为 $W^{t-1}$，并在 $W^{t}$ 与 $W^{t-1}$ 之间用 $\ell_{2}$ 正则化持续训练。

5) DNN-EWC。用弹性权重巩固（Kirkpatrick et al., 2017）正则化训练的深度网络。

6) DNN-Progressive。我们对渐进网络（Rusu et al., 2016）的实现，其每个任务的网络权重对后续到达的任务保持固定。

7) DEN。我们的动态可扩展网络。

基础网络设置。 1) 前馈网络：我们使用一个两层网络，含 312-128 个神经元，采用 ReLU 激活。2) 卷积网络。在 CIFAR-100 数据集上的实验中，我们使用 AlexNet（Krizhevsky et al., 2012）的修改版本，它有五个卷积层（深度 64-128-256-256-128，滤波器尺寸 $5 \times 5$）和三个全连接层（每层 384-192-100 个神经元）。

所有模型和算法均使用 Tensorflow（Abadi et al., 2016）库实现。我们将在论文被接收后公开代码，以供复现结果。

数据集。 1) MNIST-Variation。该数据集由 62,000 张 0 到 9 的手写数字图像构成。与 MNIST 不同，手写数字被旋转到任意角度，且背景含有噪声，这使得预测任务更具挑战性。我们对每个类别使用 1,000/200/5,000 张图像作训练/验证/测试划分。我们将每个任务构造成一对多（one-versus-rest）二分类。

2) CIFAR-100。该数据集由 100 个通用物体类别的 60,000 张图像构成（Krizhevsky & Hinton, 2009）。每个类别有 500 张训练图像和 100 张测试图像。我们使用 CNN 作为该数据集实验的基础网络，以表明我们的方法适用于 CNN。此外，我们将每个任务视为一组 10 个子任务，每个子任务是对每个类别的二分类任务。

3) AWA（Animals with Attributes）。该数据集由 50 种动物的 30, 475 张图像构成（Lampert et al., 2009）。对于特征，我们使用随数据集提供的 DECAF 特征，其维度通过 PCA 降至 500。我们使用 30/30/30 张图像的随机划分作训练/验证/测试。

### 4.1 定量评估

我们从预测准确率和效率两方面验证模型，其中效率由训练结束时的网络规模和训练时间衡量。我们首先在图 3 顶行报告基线和我们模型的平均每任务性能。DNN-STL 在 AWA 和 CIFAR-100 数据集上表现出最佳性能；它们表现好是意料之中的，因为它们被训练为对每个任务最优，而所有其他模型都是在线训练的，可能引起语义漂移。当任务数量较少时，MTL 通过多任务学习的知识共享表现最好；但当任务数量较大时，STL 表现更好，因为它比 MTL 具有更大的学习容量。我们的模型 DEN 的表现与这些批处理模型几乎相同，甚至在 MNIST-Variation 数据集上优于它们。结合正则化的再训练模型（如 L2 和 EWC）表现不佳，尽管后者优于前者。这是意料之中的，因为这两个模型无法动态增加容量。渐进网络比这两者表现更好，但在所有数据集上都逊于 DEN。性能差距在 AWA 上最为显著，因为任务数量较多（$T = 50$）可能使其更难找到合适的网络容量。如果网络太小，它将没有足够的学习容量来表示新任务；如果太大，则容易过拟合。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//a796d80b-2b68-4e45-8585-fbc9d4925853/markdown_3/imgs/img_in_chart_box_202_170_454_387.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A49%3A31Z%2F-1%2F%2F16b1169090284d5c839814154e0ae30ec7f872bfddc47408c609a15969a26c4f" alt="Image" width="20%" /></div>

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//a796d80b-2b68-4e45-8585-fbc9d4925853/markdown_3/imgs/img_in_chart_box_474_167_719_388.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A49%3A32Z%2F-1%2F%2Facad6d5402a42843ec152b1b81ca8bca3fd8d97e2b9849f337f609ca502c97f7" alt="Image" width="20%" /></div>

AWA-Class 上的平均每任务性能

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//a796d80b-2b68-4e45-8585-fbc9d4925853/markdown_3/imgs/img_in_chart_box_747_179_989_382.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A49%3A33Z%2F-1%2F%2Fc52e20ab3d70546aeebf51c6620b603d63e1e2b077c879158e536396e24f39f6" alt="Image" width="19%" /></div>

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//a796d80b-2b68-4e45-8585-fbc9d4925853/markdown_3/imgs/img_in_chart_box_207_398_461_618.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A49%3A33Z%2F-1%2F%2F7387844d11bbe1192b9d2d5ef52d54638fb0f3e6a593b673659104a7f1cbfa8a" alt="Image" width="20%" /></div>

<div style="text-align: center;"><div style="text-align: center;">(a) MNIST-Variation（T=10）</div> </div>

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//a796d80b-2b68-4e45-8585-fbc9d4925853/markdown_3/imgs/img_in_chart_box_478_402_722_618.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A49%3A34Z%2F-1%2F%2F3cdba638f5952937fac224059da409c2669607aa8f3236cbeb317aacc8271945" alt="Image" width="19%" /></div>

<div style="text-align: center;"><div style="text-align: center;">(b) CIFAR-100（T=10）</div> </div>

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//a796d80b-2b68-4e45-8585-fbc9d4925853/markdown_3/imgs/img_in_chart_box_747_403_994_617.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A49%3A34Z%2F-1%2F%2F89b564015736d556712159df141eb3947b95e96960f161bfa345574ec81bbeba" alt="Image" width="20%" /></div>

<div style="text-align: center;"><div style="text-align: center;">(c) AWA（T=50）</div> </div>

<div style="text-align: center;"><div style="text-align: center;">图 3：顶行：模型随任务数 $t$ 的平均每任务性能，在五次随机划分上取平均。图例中的数字表示模型完成学习后（$t = T$）的平均每任务性能。底行：准确率随网络容量的变化。网络容量相对于 MTL 的容量给出，我们将后者视为 100%。</div> </div>

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//a796d80b-2b68-4e45-8585-fbc9d4925853/markdown_3/imgs/img_in_chart_box_225_786_404_930.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A49%3A34Z%2F-1%2F%2F6fef53bdf684fda398d739cef16ad0b1c26fb2ce8296a14cc62b9f5816b69547" alt="Image" width="14%" /></div>

<div style="text-align: center;"><div style="text-align: center;">(a) AUROC 随时间变化</div> </div>

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//a796d80b-2b68-4e45-8585-fbc9d4925853/markdown_3/imgs/img_in_chart_box_428_786_601_930.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A49%3A34Z%2F-1%2F%2F9e91c174ebbbf8d7aff98cb1e4dd1bae028c146166c90381efa54a3e0082a212" alt="Image" width="14%" /></div>

<div style="text-align: center;"><div style="text-align: center;">(b) 选中的神经元</div> </div>

<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>模型</td><td style='text-align: center; word-wrap: break-word;'>AUROC</td><td style='text-align: center; word-wrap: break-word;'>容量</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>DNN-L2</td><td style='text-align: center; word-wrap: break-word;'>0.7440\pm0.01</td><td style='text-align: center; word-wrap: break-word;'>1</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>DNN-L2-Selective</td><td style='text-align: center; word-wrap: break-word;'>0.7499\pm0.01</td><td style='text-align: center; word-wrap: break-word;'>1</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>DEN-Constant (k=13)</td><td style='text-align: center; word-wrap: break-word;'>0.7611\pm0.01</td><td style='text-align: center; word-wrap: break-word;'>1.55</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>DEN-Constant (k=20)</td><td style='text-align: center; word-wrap: break-word;'>0.7580\pm0.01</td><td style='text-align: center; word-wrap: break-word;'>1.89</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>DEN-Dynamic</td><td style='text-align: center; word-wrap: break-word;'>0.7684\pm0.01</td><td style='text-align: center; word-wrap: break-word;'>1.56\pm0.01</td></tr></table>

<div style="text-align: center;"><div style="text-align: center;">(c) 扩展性能</div> </div>

<div style="text-align: center;"><div style="text-align: center;">图 4：选择性再训练的效果。(a) 显示 AUROC 随实际训练时间的变化，(b) 显示选择性再训练选中的神经元数量。(c) 扩展性能。我们在 MNIST-Variance 数据集上报告预测 AUROC 和以相对 DNN-MTL 参数数量衡量的网络容量。报告的数字为五次随机划分的均值与标准误。</div> </div>

我们进一步在图 3（底行）报告每个模型在相对 MTL 衡量的网络容量下的性能。对于基线，我们报告不同网络容量的多个模型的性能。DEN 以远少于渐进网络的参数获得更好的性能，或以相近的参数数量获得显著更好的性能。在 MNIST-Variation、CIFAR-100 和 AWA 上，DEN 仅用 STL 的 18.0%、60.3% 和 11.9% 容量就达到了与 STL 相同的性能水平。这也显示了 DEN 的主要优势，即能够动态找到其最优容量，因为它在 MNIST-Variation 上学到一个非常紧凑的模型，而在 CIFAR-100 上学到一个大得多的网络。进一步在所有任务上微调 DEN（DEN-Finetune）在所有数据集上都获得性能最好的模型，这表明 DEN 不仅对终身学习有用，而且在所有任务一开始都已知时也可用于网络容量估计。

选择性再训练的效果。 我们通过测量 MNIST-Variation 数据集上的训练速度和 ROC 曲线下面积（AUROC），进一步考察选择性训练的高效性和有效性。为此，我们将不含网络扩展的模型（我们称为 DNN-Selective）与在 DNN-L2 和 DNN-L1（式 (2)）上的再训练在准确率和效率两方面进行比较。图 4(a)

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//a320b3e0-5bf1-4180-ba22-11a51fb4c1e5/markdown_0/imgs/img_in_chart_box_204_136_457_363.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A49%3A24Z%2F-1%2F%2F16735f965fe47aa9003456191b75108a109a7f19ba95352afa3060f46190f1d1" alt="Image" width="20%" /></div>

<div style="text-align: center;"><div style="text-align: center;">(a) 任务 1</div> </div>

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//a320b3e0-5bf1-4180-ba22-11a51fb4c1e5/markdown_0/imgs/img_in_chart_box_479_137_730_364.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A49%3A24Z%2F-1%2F%2F3ce659e1635ab90139aa6e59a140973d9d4cc957d460803fe2f25a5e5e1ca0d8" alt="Image" width="20%" /></div>

<div style="text-align: center;"><div style="text-align: center;">(b) 任务 4</div> </div>

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//a320b3e0-5bf1-4180-ba22-11a51fb4c1e5/markdown_0/imgs/img_in_chart_box_750_136_998_364.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A49%3A25Z%2F-1%2F%2F1f3faba198543aff2fbee87a72a5923bebbe0b9e0a68e611249a6b3da7bfd8f9" alt="Image" width="20%" /></div>

<div style="text-align: center;"><div style="text-align: center;">(c) 任务 7</div> </div>

<div style="text-align: center;"><div style="text-align: center;">图 5：MNIST-Variation 数据集上的语义漂移实验。我们在每个训练阶段报告不同模型在 $t = 1$、t = 4 和 t = 7 上的 AUROC，以观察这些任务的模型性能随时间如何变化。报告的 AUROC 为五次随机划分的平均。</div> </div>

显示了每个模型的准确率随训练时间（以 GPU 计算实际耗时衡量）的变化。我们观察到，选择性再训练所需时间显著少于对网络的完整再训练，甚至少于带稀疏网络权重的 DNN-L1。此外，虽然 DNN-L1 的准确率略低于 DNN-L2，但 DNN-Selective 将准确率较基础网络提高了 2%。这一准确率提升可能源于对灾难性遗忘的抑制，因为 DEN 对每个新引入的任务只训练部分子网络。图 4(b) 显示了选择性再训练下每层选中的神经元数量。注意 DNN-selective 大多选择较少部分的上层单元（这些单元更任务特定），而选择较大比例的更通用的下层单元。

网络扩展的效果。 我们还将网络扩展的有效性与我们模型的一个变体进行比较，该变体执行选择性再训练和层扩展，但不进行网络拆分。我们将该模型称为 DNN-Dynamic。我们在 MNIST-Variation 数据集上将 DNN-Dynamic 与主实验中使用的 DNN-L2 以及 DNN-Constant 进行比较，后者是我们模型的一个版本，在每层用固定数量的单元扩展容量。图 4(c) 显示了实验结果。DNN-Dynamic 获得最佳的平均 AU-ROC，显著优于包括 DNN-Constant 在内的所有模型，同时网络规模的增长显著小于 DNN-Constant（k=20）。这可能是因为参数数量较少不仅在训练效率上有益，也有助于防止模型过拟合。我们可以将 DNN-Constant 的网络容量设置为相近（k=13）以获得更好的准确率，但它仍然逊于能在每层动态调整神经元数量的 DEN。

网络拆分/复制和时间戳推理的效果。 为观察网络拆分/复制和单元时间戳如何帮助防止语义漂移（或灾难性遗忘），同时允许在后续任务上获得良好性能，我们在不同学习阶段将我们模型的性能与基线以及我们 DEN 的一个不含时间戳推理的变体（DEN-No-Stamp）进行比较。图 5 的 (a)、(b) 和 (c) 各图中显示了模型性能在每个训练阶段 t 上如何变化，分别针对任务 t=1、t=4 和 t=7。

我们观察到，DNN-L2 防止了早期阶段学到的模型的语义漂移，但导致后续任务（t=4, 7）的性能越来越差。另一方面，DNN-EWC 在后续任务上的性能优于 DNN-L2，如 Kirkpatrick et al. (2017) 所报告。然而，它的性能显著低于 DNN-Progressive 和我们的模型，这可能是因为它无法增加网络容量，导致表达能力受限。DNN-Progressive 在旧任务上没有语义漂移，这是意料之中的，因为它不为旧任务再训练参数。不含时间戳的 DEN 在后续任务上比 DNN-Progressive 表现更好，但性能随时间略有退化。最后，我们带有时间戳推理的完整模型 DEN 在任何学习阶段都没有明显的性能退化迹象，同时显著优于 DNN-Progressive。这些结果表明 DEN 在防止语义漂移方面也非常有效。

## 5 结论

我们提出了一种用于终身学习的新型深度神经网络——动态可扩展网络（DEN）。DEN 通过利用任务相关性对在旧任务上训练的网络进行部分再训练，同时在必要时增加容量以容纳新任务所需的新知识，从而为自身找到最优容量，同时有效防止语义漂移。我们实现了 DEN 的前馈和卷积神经网络两种版本，并在终身学习场景下的多个分类数据集上验证，它们显著优于现有的终身学习方法，在只使用 11.9%p - 60.3%p 容量的情况下达到与批处理训练的网络几乎相同的性能。进一步在所有任务上微调模型，得到的模型性能优于批处理模型，这表明 DEN 对网络结构估计同样有用。

致谢 本研究由韩国科学与信息通信技术部及未来创造科学部通过韩国国家研究基金会资助的下一代信息计算发展计划（NRF-2016M3C4A7952600）、三星电子三星研究资助中心（SRFC-IT150203）以及 MSIP/IITP 的 ICT 研发计划（2016-0-00563，面向智能自主数字伙伴的自适应机器学习技术开发研究，以及 2017-0-00537，面向太空智能的自主物联网协作框架开发）资助。

## 参考文献

Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, et al. Tensorflow: Large-scale Machine Learning on Heterogeneous Distributed Systems. arXiv:1603.04467, 2016.

Jose M Alvarez and Mathieu Salzmann. Learning the number of neurons in deep networks. In Advances in Neural Information Processing Systems, pp. 2262–2270, 2016.

Corinna Cortes, Xavi Gonzalvo, Vitaly Kuznetsov, Mehryar Mohri, and Scott Yang. Adanet: Adaptive structural learning of artificial neural networks. arXiv preprint arXiv:1607.01097, 2016.

Eric Eaton and Paul L. Ruvolo. ELLA: An efficient lifelong learning algorithm. In Sanjoy Dasgupta and David Mcallester (eds.), ICML, volume 28, pp. 507–515. JMLR Workshop and Conference Proceedings, 2013.

James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, pp. 201611835, 2017.

Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. 2009

Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. ImageNet Classification with Deep Convolutional Neural Networks. In NIPS, 2012.

Abhishek Kumar and Hal Daume III. Learning task grouping and overlap in multi-task learning. In ICML, 2012.

Christoph Lampert, Hannes Nickisch, and Stefan Harmeling. Learning to Detect Unseen Object Classes by Between-Class Attribute Transfer. In CVPR, 2009.

Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based Learning Applied to Document Recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.

Sang-Woo Lee, Jin-Hwa Kim, Jung-Woo Ha, and Byoung-Tak Zhang. Overcoming catastrophic forgetting by incremental moment matching. arXiv preprint arXiv:1703.08475, 2017.

George Philipp and Jaime G. Carbonell. Nonparametric neural networks. In ICLR, 2017.

Andrei Rusu, Neil Rabinowitz, Guillaume Desjardins, Hubert Soyer, James Kirkpatrick, Koray Kavukcuoglu, Razvan Pascanu, and Raia Hadsell. Progressive neural networks. arXiv preprint arXiv:1606.04671, 2016.

S. Thrun. A lifelong learning perspective for mobile robot control. In V. Graefe (ed.), Intelligent Robots and Systems. Elsevier, 1995.

Wei Wen, Chunpeng Wu, Yandan Wang, Yiran Chen, and Hai Li. Learning structured sparsity in deep neural networks. In NIPS, pp. 2074–2082, 2016.

Tianjun Xiao, Jiaxing Zhang, Kuiyuan Yang, Yuxin Peng, and Zheng Zhang. Error-driven incremental learning in deep convolutional neural network for large-scale image classification. In Proceedings of the 22nd ACM international conference on Multimedia, pp. 177–186. ACM, 2014.

Friedemann Zenke, Ben Poole, and Surya Ganguli. Continual learning through synaptic intelligence. In ICML, pp. 3987–3995, 2017.

Guanyu Zhou, Kihyuk Sohn, and Honglak Lee. Online incremental feature learning with denoising autoencoders. In International Conference on Artificial Intelligence and Statistics, pp. 1453–1461, 2012.

#### 附录 A

### A.1 置换 MNIST 上的结果

我们提供在置换 MNIST（Permuted MNIST）数据集（Lecun et al., 1998）上的额外实验结果。该数据集由 70,000 张 0 到 9 的手写数字图像构成，其中 60,000 张用于训练，10,000 张用于测试。该数据集与原始 MNIST 的不同之处在于，十个任务中的每一个都是对输入像素不同随机置换的多类分类。图 6 显示了该实验的结果。我们的 DEN 优于所有终身学习基线，而只使用了基础网络容量的 1.39 倍。此外，DEN-Finetune 在所有模型（包括 DNN-STL 和 DNN-MTL）中取得最佳 AUROC。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//a320b3e0-5bf1-4180-ba22-11a51fb4c1e5/markdown_2/imgs/img_in_chart_box_256_891_587_1182.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A49%3A30Z%2F-1%2F%2Fafcb516ea67887545d3cf4de02da426d26ad52f0d1ca0722ab28049efd96a198" alt="Image" width="27%" /></div>

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//a320b3e0-5bf1-4180-ba22-11a51fb4c1e5/markdown_2/imgs/img_in_chart_box_613_892_934_1181.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A49%3A30Z%2F-1%2F%2F2dc863b9368e12389358d62dc04675bfcddf77caef0feda72554494315b7ef3c" alt="Image" width="26%" /></div>

<div style="text-align: center;"><div style="text-align: center;">图 6：置换 MNIST 上的结果。所有模型在置换 MNIST 上相对 MTL 的平均每任务 AUROC 和网络容量。</div> </div>
