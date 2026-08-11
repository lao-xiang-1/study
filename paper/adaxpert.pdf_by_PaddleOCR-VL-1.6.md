arXiv:2107.00254v1 [cs.LG] 1 Jul 2021

# AdaXpert：为增长数据自适应调整神经网络架构

Shuaicheng Niu $^{*12}$ Jiaxiang Wu $^{*3}$ Guanghui Xu $^{1}$ Yifan Zhang $^{4}$

Yong Guo $^{1}$ Peilin Zhao $^{3}$ Peng Wang $^{5}$ Mingkui Tan $^{16}$

## 摘要

在实际应用中，数据通常以增长的方式到来，其中数据量和类别数可能动态增加。这给学习带来了一个关键挑战：面对不断增加的数据量或类别数，必须即时调整神经模型的容量以获得良好的性能。现有方法要么忽略了数据的增长特性，要么试图为给定数据集独立搜索最优架构，因此无法及时针对变化的数据调整架构。为此，我们提出了一种神经架构自适应方法，即Adaptation eXpert（AdaXpert），能够在增长数据上高效地调整之前的架构。具体来说，我们引入了一个架构调整器（architecture adjuster），根据之前的架构以及当前与之前数据分布之间的差异程度，为每个数据快照生成合适的架构。此外，我们提出了一种自适应条件（adaptation condition）来判断是否需要调整，从而避免不必要且耗时的调整。在两种增长场景（数据量增加和类别数增加）上的大量实验证明了所提出方法的有效性。

## 1. 引言

深度神经网络（DNNs）已在许多具有挑战性的任务中取得了最先进的成果，包括图像分类（Hu et al., 2018; Lu et al., 2021）、自然语言处理（Devlin et al., 2019; Brown et al., 2020）以及许多其他领域（Cao et al., 2019; Zhang et al., 2020; Guo et al., 2020a; Zeng et al., 2020）。DNN成功的关键因素之一在于有效神经架构的设计，包括：1）手动设计的架构，如ResNet（He et al., 2016）和MobileNet（Howard et al., 2017）；2）自动设计的架构，如（Zoph et al., 2018; Cai et al., 2019; Tan et al., 2019）。然而，这些方法通常为特定任务/数据集设计固定的架构。

在实际应用中，数据通常以增长的方式到来。例如，智能边缘设备（如数十亿部手机和监控摄像头）和医学成像设备每天都在持续收集新数据（Grantz et al., 2020; Liang et al., 2019）。具体来说，新收集的数据有以下两种类型：1）数据量增加：新数据的标签已在之前的数据中出现，增长不改变数据的标签空间；2）类别数增加：新到达的数据具有与之前数据不同的标签，因此数据的标签空间在增长。在这两种场景下，数据分布都可能动态变化。由于最优网络架构在不同的数据分布下可能不同（Zoph & Le, 2017），在将DNN应用于增长数据时，可以（也应该）动态调整架构以获得更好的性能（见图1）。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//4ab30407-501c-46a3-af34-55a63f92d4b7/markdown_0/imgs/img_in_chart_box_627_372_837_539.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-11T08%3A45%3A45Z%2F-1%2F%2Fb6a765ab290d885937ccf94a166ea76a347cdb037c1cfc11c2ee54265ded7999" alt="Image" width="17%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//4ab30407-501c-46a3-af34-55a63f92d4b7/markdown_0/imgs/img_in_image_box_874_374_1053_534.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-11T08%3A45%3A45Z%2F-1%2F%2Fe232c61f5d1409d13feb89cf1bc7f013001454018e3b902ba2d7773b64876197" alt="Image" width="14%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图1. 架构自适应的动机。左图：在不同大小CIFAR100子集上训练的ResNet性能比较。最优架构在不同子集大小之间有所不同。右图：由于数据以增长方式到来且数据分布可能动态变化，应根据数据分布的偏移来调整模型架构。</div> </div>


为了实现上述目标，一个直观的解决方案是在新数据到达时重新设计网络架构。然而，有效神经架构的设计在很大程度上依赖于人类的专业知识。此外，人工设计无法充分探索完整的架构空间，导致产生次优架构（Zoph & Le, 2017）。除了手动设计之外，还可以借助自动神经架构搜索（NAS）技术（Cai et al., 2019; Tan et al., 2019）。然而，这些方法为每次数据增长独立地从头设计新架构，忽略了之前的架构是可迁移的，导致设计效率低下。此外，无论是手动设计还是自动设计，都没有考虑架构调整的必要性以进一步提高自适应效率。直观地说，如果新到达的数据与之前的数据非常相似，则无需进行自适应调整。

 $^{*}$同等贡献。本工作完成于Shuaicheng Niu在腾讯AI实验室实习期间。 $^{1}$华南理工大学软件工程学院，中国 $^{2}$大数据与智能机器人教育部重点实验室，中国 $^{3}$腾讯AI实验室，中国 $^{4}$新加坡国立大学，新加坡 $^{5}$西北工业大学，中国 $^{6}$琶洲实验室，中国。通讯作者：Mingkui Tan <mingkuitan@scut.edu.cn>。

发表于第 $38^{th}$ 届国际机器学习大会论文集，PMLR 139，2021年。版权归作者所有。

为了解决上述局限性，我们提出了一种神经架构自适应方法，称为AdaXpert（Adaptation eXpert），由一个架构调整器和一个自适应条件组成。具体来说，我们首先采用Wasserstein距离来定量度量当前数据与之前数据之间的差异。然后，调整器以之前的架构和数据差异作为输入，为当前数据生成合适的架构。接着，调整器根据此次调整获得一个奖励，该奖励旨在在准确率和计算效率之间取得平衡。最终调整后的架构由训练好的调整器生成。值得注意的是，我们的方法旨在为当前数据找到新的最优架构，而不是简单地将之前的架构扩展为更大的架构。架构调整是弹性的，例如，调整可能会从之前的架构中移除冗余层并增加某些层的容量（卷积核大小和/或通道数），如图4所示。此外，我们提出了一种自适应条件来判断架构调整的必要性。通过这种方式，当新到达的数据与之前的数据高度相似时，我们避免了不必要的调整，从而进一步提高了调整效率。基于上述考虑，我们的AdaXpert能够自动调整架构，在尽可能小的计算成本下获得当前数据上更好的性能。我们的主要贡献总结如下：

- 我们提出了一种面向增长数据的网络自适应方法。通过考虑当前数据与之前数据之间的差异，我们的方法自适应地调整模型架构，在保持较小计算成本的同时实现更好的性能。

- 我们提出了一种自适应条件来判断架构调整的必要性。借助该条件，我们的方法避免了对高度相似数据进行不必要的自适应调整。

- 在两种数据增长场景（即数据量增加和类别数增加）上的实验证明了所提出方法的有效性和优越性。

## 2. 相关工作

神经架构搜索（NAS）在自动设计有效架构方面受到了越来越多的关注。经典的NAS问题（Zoph & Le, 2017）利用强化学习（RL）范式来生成DNN的模型描述。基于RL的方法（Pham et al., 2018; Tan et al., 2019; Zoph et al., 2018）旨在学习一个具有策略的控制器来生成架构。除了RL之外，基于进化的方法（Real et al., 2019; Piergiovanni et al., 2019）和基于梯度的算法（Liu et al., 2019; Xu et al., 2020）也发现了性能优异的新架构。近年来，基于元学习的方法（Lian et al., 2020; Wang et al., 2020）关注少样本问题，并自动学习一个旨在快速适应新任务的元架构。与NAS设计固定架构不同，我们动态调整网络架构以处理增长数据的问题。

持续学习（CL）旨在将之前任务中学到的知识迁移到未来的场景中。为了解决新任务，基于回放的方法（Rebuffi et al., 2017; Chaudhry et al., 2019; Rolnick et al., 2019）选择性地存储之前任务的样本用于新任务的训练。为了缓解灾难性遗忘问题，基于正则化的方法（Kirkpatrick et al., 2017; Liu et al., 2018; Lange et al., 2020）在损失函数中引入正则化项，要求模型不改变之前任务的重要参数。另一个并行任务是在线学习（OL）（Hoi et al., 2018; Zhang et al., 2019），旨在基于一系列训练样本学习一个性能良好的模型。随着数据的增长，CL旨在克服之前数据上的遗忘问题并进行调整以在新数据上获得更好的性能，而OL则关注特定模型的参数学习。在本工作中，我们旨在每次数据增长时在整个数据集上即时调整架构。

渐进式神经网络。为了提高模型容量，CL方法（Rusu et al., 2016; Xu & Zhu, 2018; Rosenfeld & Tsotsos, 2020）提出动态扩展其网络架构。这些方法固定之前任务的层并为新任务生长分支。此外，DEN（Yoon et al., 2018）首先为新任务将架构扩展到较大尺寸，然后使用剪枝方法移除不重要的权重。最近，Gao et al., 2020和Li et al., 2019结合NAS技术为每个任务设计架构以实现CL的目标。然而，这些方法忽略了当前数据与之前数据之间的分布差异，因此难以确定调整后架构的合适模型大小。在本工作中，我们基于之前的架构和增长数据的特性来动态调整架构。此外，除了扩展之外，我们的调整还可能移除冗余层或添加新层。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//4ab30407-501c-46a3-af34-55a63f92d4b7/markdown_2/imgs/img_in_image_box_114_139_1074_364.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-11T08%3A45%3A48Z%2F-1%2F%2Fbfc16245636cead34fad311557f2f3ddbd3862e325387d47f20acf24e670ad67" alt="Image" width="78%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图2. 我们提出的AdaXpert示意图。(a) 在时间步 $t$，给定新输入数据 $\mathcal{D}_t^{new}$ 和之前的模型 $\alpha_{t-1}$，我们首先判断是否需要调整架构。如果需要，将 $\alpha_{t-1}$ 输入NAA模块进行架构自适应。(b) 我们的控制器以 $\alpha_{t-1}$ 的架构以及当前数据 $\mathcal{D}_{t-1} \cup \mathcal{D}_t^{new}$ 与之前数据 $\mathcal{D}_{t-1}$ 之间的分布距离作为输入，输出调整后的架构。控制器随后获得一个奖励，因此可以通过策略梯度方法进行训练。最后，我们采用训练好的NAA生成最终调整后的架构 $\alpha_t$。</div> </div>


## 3. 所提出的方法

### 3.1. 问题定义

在本文中，我们旨在随着数据的增长动态调整神经网络架构。形式化地，我们将一系列新输入数据记为 $\mathcal{D}_t^{new}=\{(x_i^t, y_i^t)\}_{i=1}^{n_t}$， $t=1,...,T$，其中 $x_i^t \in \mathcal{X}$， $y_i^t \in \{1,...,C_t\}$， $\mathcal{X}$ 为输入图像空间，$n_t$ 和 $C_t$ 分别为图像数量和类别数。此外，我们将时间步 $t$ 的所有累积数据记为 $\mathcal{D}_t = \mathcal{D}_1^{new} \cup \cdots \cup \mathcal{D}_t^{new}$。

对于上述增长数据集 $\{\mathcal{D}_t\}_{t=1}^T$，其相应的数据分布可能动态变化，因此不同 $\mathcal{D}_t$ 的最优架构也可能不同。然而，现有方法通常为特定任务/数据集设计固定架构，而单一固定网络架构可能对所有 $\{\mathcal{D}_t\}_{t=1}^T$ 都不是最优的。为了获得更好的性能，应该为不同的 $\mathcal{D}_t$ 设计不同的架构，即随着数据的增长动态调整模型架构。

为了实现上述目标，一种直观的方法是对每个 $\mathcal{D}_t$ 分别进行神经架构搜索，以获得相应的架构 $\alpha_t$。然而，这种方法忽略了之前的 $\alpha_{t-1}$ 对于 $\mathcal{D}_t$ 是可迁移的，这可以提高 $\alpha_t$ 的搜索效率。此外，当新输入数据与之前的数据非常相似时，避免不必要的搜索过程也很重要。

为了解决上述挑战，我们提出了一种神经架构自适应方法，即Adaptation eXpert（AdaXpert），旨在以最少的专家干预自动设计动态网络。AdaXpert主要由两个关键组件组成：1）基于强化学习的神经架构调整器（见第3.2节），根据新输入数据 $\mathcal{D}_t^{new}$ 的特性将之前的模型架构 $\alpha_{t-1}$ 自适应地调整为新的 $\alpha_t$；2）判断架构调整必要性的自适应条件（见第3.3节），从而避免不必要的调整。AdaXpert的详细流程总结在算法1中。

### 3.2. 动态神经架构自适应

给定之前的深度模型和新输入数据，我们旨在自动调整模型架构以获得更好的性能，同时保持较小的模型计算成本（如MAdds）。为此，我们设计了一个网络架构调整器（NAA）算法，旨在根据 $\mathcal{D}_{t-1}$ 和 $\mathcal{D}_t$ 之间的分布差异执行不同的调整策略。具体来说，如果新数据与之前的数据非常相似，我们只需进行轻微调整，即保持调整后架构的MAdds增长较小。否则，我们允许调整有相对较大的MAdds增长。

在下文中，我们解决架构自适应的两个关键问题：1）如何度量当前数据 $\mathcal{D}_t$ 与之前数据 $\mathcal{D}_{t-1}$ 之间的差异；2）如何设计架构调整器。

数据差异的定量度量。应根据当前数据与之前数据之间的差异程度执行不同的架构自适应策略。为了量化这种差异，我们计算当前数据与之前数据之间的分布距离如下。

形式化地，给定当前数据集 $\mathcal{D}_t$ 和之前的数据集 $\mathcal{D}_{t-1}$，我们首先将这两个数据集输入之前的模型 $\alpha_{t-1}$，分别获得它们的特征嵌入 $\mathbf{M}_t \in \mathbb{R}^{m \times q}$ 和 $\mathbf{M}_{t-1} \in \mathbb{R}^{n \times q}$。其中，$m$ 和 $n$ 分别表示 $\mathcal{D}_t$ 和 $\mathcal{D}_{t-1}$ 中的样本数量，$q$ 表示特征维度。然后，$\mathbf{M}_t$ 和 $\mathbf{M}_{t-1}$ 可以被视为从两个未知分布 $\mathbb{P}_t$ 和 $\mathbb{P}_{t-1}$ 中采样的两个样本矩阵。为了计算 $\mathbb{P}_t$ 和 $\mathbb{P}_{t-1}$ 之间的距离，可以使用非参数

算法1 AdaXpert的整体算法。

输入：输入数据集 $\{\mathcal{D}_t^{new}\}_{t=1}^{T}$；针对 $\mathcal{D}_1^{new}$ 训练好的模型 $\alpha_1$；超网络 $\mathcal{N}_1$ 和控制器 $\pi(\cdot; \theta_1)$；阈值 $\epsilon$。

1: 令 $\mathcal{D}_1 = \mathcal{D}_1^{new}$。

2: for $t=2,\ldots,T$ do

3: 令 $\mathcal{D}_t = \mathcal{D}_{t-1} \cup \mathcal{D}_t^{new}$，$\mathcal{N}_t = \mathcal{N}_{t-1}$，$\theta_t = \theta_{t-1}$。

4: 使用公式(4)计算准确率差异 $H_t$。

5: if $H_t > \epsilon$ then

6: 使用算法2在 $\mathcal{D}_t$ 上更新 $\mathcal{N}_t$ 和 $\pi(\cdot; \theta_t)$。

7: 生成调整后的架构 $\alpha_t \sim \pi(\cdot; \theta_t)$。

8: 在 $\mathcal{D}_t$ 上重新训练调整后的架构 $\alpha_t$。

9: else

10: 令 $\alpha_t = \alpha_{t-1}$。

11: end if

12: end for

输出：调整后的架构 $\{\alpha_t\}_{t=1}^T$。

估计方法来计算Kullback-Leibler（KL）散度（Nguyen et al., 2007）或Wasserstein距离（WD）（Sriperumbudur et al., 2010）。

此处也可以使用其他指标来计算分布距离，如KL散度和Jensen-Shannon散度（Fuglede & Topsoe, 2004）。关于WD的更多讨论见补充材料。基于 $\mathcal{W}(\mathcal{D}_{t}, \mathcal{D}_{t-1})$，我们设计了一个数据差异感知的控制器来执行不同的架构调整。由于我们的调整问题具有高度非凸性质，我们将其转化为马尔可夫决策过程（MDP），然后使用强化学习方法训练控制器。

然而，上述非参数估计方法可能计算代价高昂。例如，计算KL散度需要求解一个二次规划问题。幸运的是，我们的初步研究表明，样本矩阵 $\mathbf{M}_t$ 和 $\mathbf{M}_{t-1}$ 近似满足多元高斯分布（更多细节见补充材料）。因此，在本文中，我们假设 $\mathbb{P}_t$ 和 $\mathbb{P}_{t-1}$ 是两个多元高斯分布，并使用最大似然估计方法获得其分布参数，即 $\mathbb{P}_t \sim \mathcal{N}(\mu_t, \Sigma_t)$ 和 $\mathbb{P}_{t-1} \sim \mathcal{N}(\mu_{t-1}, \Sigma_{t-1})$。然后，我们计算Wasserstein距离（Takatsu et al., 2011）如下：

算法2 网络架构调整器的训练。

输入：数据集 $\{\mathcal{D}_{t-1}, \mathcal{D}_t\}$；之前的架构 $\alpha_{t-1}$，超网络 $\mathcal{N}_t$ 和控制器 $\pi(\cdot)$ 及其参数 $\theta_t$；超参数 $\eta$ 和 $M$。

1: 将 $\mathcal{D}_t$ 分为训练集和验证集 $\{\mathcal{D}_{train}, \mathcal{D}_{val}\}$。

2: 在 $\mathcal{D}_{train}$ 上微调 $\mathcal{N}_t$。

3: 使用公式(1)计算 $\mathcal{D}_{t-1}$ 和 $\mathcal{D}_t$ 之间的WD $(d_t)$；4: // 训练控制器模型

5: for $i=1,...,M$ do

6: 采样 $\alpha_t' \sim \pi(\alpha_{t-1}, d_t; \theta_t)$。

7: 从 $\mathcal{D}_{val}$ 中采样一批数据。

8: 基于 $\mathcal{N}_t$ 使用公式(5)计算奖励 $\mathcal{R}(\alpha_t')$。

9: 更新 $\theta_t \leftarrow \theta_t + \eta \mathcal{R}(\alpha_t') \nabla_{\theta_t} \log \pi(\cdot)$。

10: end for

输出：超网络 $\mathcal{N}_t$ 和控制器 $\pi(\cdot; \theta_t)$。

 $$ \begin{align*}\mathcal{W}(\mathcal{D}_{t},\mathcal{D}_{t-1})=&||\mu_{t}-\mu_{t-1}||_{2}^{2}+\\&\operatorname{tr}\Big(\Sigma_{t}+\Sigma_{t-1}-2(\Sigma_{t-1}^{1/2}\Sigma_{t}\Sigma_{t-1}^{1/2})^{1/2}\Big).\end{align*} $$

网络架构调整器（NAA）的MDP重构。由于架构调整过程本质上是一个多步决策过程，我们将调整过程形式化为MDP。形式化地，MDP可以定义为一个元组 $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R})$，其中 $\mathcal{S}$ 是有限状态集，$\mathcal{A}$ 是有限动作集，$\mathcal{P} : \mathcal{S} \times \mathcal{A} \to \mathcal{S}$ 是状态转移分布，$\mathcal{R} : \mathcal{S} \times \mathcal{A} \to \mathcal{R}$ 是奖励函数。此外，策略 $\pi_\theta$ 在给定当前状态下决定动作。在NAA的上下文中，如图2中时间步 $t$ 所示，我们将状态记为 $s = [\alpha_{t-1}, d_t] \in \mathcal{S}$，其中 $d_t = \mathcal{W}(\mathcal{D}_t, \mathcal{D}_{t-1})$ 是当前数据与之前数据之间的分布差异。给定这样一个状态 $s$，策略（控制器）执行一系列动作 $a = \pi_\theta(s) \in \mathcal{A}$ 来确定调整后架构 $\alpha_t^\prime$ 每一层的操作。形式化地，动作空间根据不同的架构搜索空间类型来定义。然后，控制器收到一个奖励 $r = \mathcal{R}(\alpha_t^\prime)$。关于奖励设计的更多细节见第3.4节。

NAA的训练。我们NAA的目标是最大化期望奖励 $\mathbb{E}[\mathcal{R}(\alpha)]$，表示为求解以下优化问题：

 $$ \max_{\theta}\mathbb{E}_{\pi_{\theta}}[\mathcal{R}(\alpha)]. $$

按照策略梯度方法（Williams, 1992; Schulman et al., 2017），我们通过上升梯度来更新 $\theta$：

 $$ \theta\leftarrow\theta+\eta\mathcal{R}(\alpha)\nabla_{\theta}\mathrm{log}\pi_{\theta}(\alpha). $$

NAA的训练细节总结在算法2中。

### 3.3. 何时自适应网络架构

给定之前的模型 $\alpha_{t-1}$ 和当前数据 $\mathcal{D}_t$，需要考虑模型架构是否需要调整。例如，给定一个在MNIST数据集上搜索到的最优架构 $\alpha$，即使我们收到更多输入的MNIST图像，也无需调整 $\alpha$，因为它已经是最优的了。这表明在执行架构自适应时有必要考虑之前架构的可行性。因此，如图2(a)所示，每次新数据到达时，我们使用一个自适应条件来判断调整的必要性，从而提高自适应效率。

形式化地，在时间步 $t$，给定之前的模型 $\alpha_{t-1}$、当前数据 $D_t$ 和之前的数据 $D_{t-1}$，我们计算以下准确率差异以进行进一步决策：

 $$ H_{t}=\Phi(\mathcal{D}_{t-1};\alpha_{t-1})-\Phi(\mathcal{D}_{t};\alpha_{t-1}), $$

其中 $\Phi(\mathcal{D};\alpha)$ 是模型 $\alpha$ 在数据集 $\mathcal{D}$ 上的某个性能指标。对于分类模型，我们选择top-1准确率作为指标。注意，对于具有新类别标签的增长数据，之前的模型会做出错误预测，因为分类器无法预测新标签。基于上述准确率差异，可以判断是否需要调整之前的架构 $\alpha_{t-1}$。具体来说，给定 $H_t$ 及其阈值 $\epsilon$，我们仅在 $H_t > \epsilon$ 时调整模型架构。

在我们的方法中，我们利用WD来度量数据差异的程度，并采用准确率差异作为架构调整的自适应条件。原因如下：1）对于自适应条件，准确率差异对人类来说更直观。相比之下，由于WD是分布距离，很难为WD设置合适的阈值来判断调整的必要性。2）为了识别数据差异程度，WD比准确率差异具有更强的区分能力。具体来说，对于变化的标签空间，准确率差异更多地由新数据的数量决定，而WD是根据数据本身的底层特性计算的。

### 3.4. NAA的奖励设计

奖励函数 $\mathcal{R}(\cdot)$ 对于训练我们的NAA模型非常重要。在本小节中，我们将给出我们的奖励函数。为简单起见，我们仅说明单轮NAA的训练，在其他轮次中，NAA可以用同样的方式训练。给定当前和之前的数据集 $\mathcal{D}_t$ 和 $\mathcal{D}_{t-1}$，我们将它们之间的WD记为 $d_t = \mathcal{W}(\mathcal{D}_t, \mathcal{D}_{t-1})$。然后，对于当前状态 $s$，NAA执行一系列动作以获得调整后的架构 $\alpha'_t$。最终奖励计算如下：

 $$ \mathcal{R}(\alpha_{t}^{\prime}){=}\mathcal{V}(\alpha_{t}^{\prime}){-}\mathcal{V}(\alpha_{t-1}){-}\frac{\lambda}{d_{t}}\big(\mathcal{C}(\alpha_{t}^{\prime}){-}\mathcal{C}(\alpha_{t-1})\big), $$

其中 $\mathcal{V}(\alpha)$ 和 $\mathcal{C}(\alpha)$ 分别表示模型 $\alpha$ 的验证准确率和计算复杂度，$\lambda$ 是一个权衡参数。我们采用MAdds作为度量 $\alpha$ 计算复杂度的指标，也可以使用其他指标，如推理延迟。

为了获得验证准确率 $\mathcal{V}(\alpha)$，可以从头训练它然后在验证集上验证。然而，这将导致难以承受的计算负担。在本文中，我们利用权重共享技术（Pham et al., 2018）来构建一个超网络，即一个大型计算图，其中每个网络架构共享参数。这样，一旦超网络训练完成，所有架构直接从超网络继承其权重，然后使用这些权重进行进一步评估。对于公式(5)中的第一项 $\mathcal{V}(\alpha_t') - \mathcal{V}(\alpha_{t-1})$，我们希望调整后的架构 $\alpha_t'$ 能比原始的 $\alpha_{t-1}$ 获得更高的验证准确率。对于公式(5)中的第二项 $\frac{\lambda}{d_t} \times (\mathcal{C}(\alpha_t') - \mathcal{C}(\alpha_{t-1}))$，计算成本过高的架构将受到惩罚，而 $d_t$ 用于自适应地正则化调整后架构的规模。如果 $d_t$ 较小，即输入数据与原始数据非常相似，奖励函数将更多地关注约束调整后架构的计算复杂度。

## 4. 实验

在本节中，我们针对两种数据增长场景评估我们的AdaXpert，即相同标签空间内的数据量增长（场景I）和标签空间增加（场景II）。之后，我们进行消融实验以验证方法中各组件的有效性。最后，我们将自适应过程获得的架构与现有方法获得的架构进行比较。代码可在 https://github.com/mr-eggplant/adaxpert0 获取。

### 4.1. 实验设置

数据集：我们在ImageNet这一大规模图像分类数据集（Deng et al., 2009）上进行实验。基于ImageNet，我们模拟了两种数据增长场景以验证所提出方法的有效性。为方便起见，我们将ImageNet-#记为ImageNet的子集，其中"#"表示类别数。例如，ImageNet-100包含整个ImageNet前100个类别的样本。我们也以类似的方式命名动态调整的架构，例如，AdaXpert-20表示在ImageNet-20上获得的架构。

架构自适应的搜索空间：此处，我们考虑基于倒置Mobile Block（Howard et al., 2019）的架构空间。具体来说，模型被分为5个单元（unit），特征图空间大小逐渐减小，通道数逐渐增加。每个单元最多包含4层，其中仅当特征图大小减小时第一层的步长为2，其他所有层的步长为1。在我们的实验中，我们搜索每个单元中的层数（从 $\{2, 3, 4\}$ 中选择）、每层中的卷积核大小（从 $\{3, 5, 7\}$ 中选择）以及每层中的宽度扩展比（从 $\{3, 4, 6\}$ 中选择）。

对比方法：我们将AdaXpert与三类方法进行比较。1）手动设计的网络，包括MobileNetV2、MobileNetV2 $1.4\times$（Howard et al., 2017）、ResNet18和ResNet50（He et al., 2016）。在整个数据增长过程中，这些模型使用相同的固定架构进行训练和评估。2）神经架构搜索（NAS）方法。EfficientNet（Tan & Le, 2019）和MnasNet（Tan et al., 2019）在

<div style="text-align: center;"><div style="text-align: center;">表1. 场景I：在不同大小训练集的ImageNet-100上的比较。我们报告准确率 $\%$（$\uparrow$）和MAdds（百万，$\downarrow$）。</div> </div>




<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td rowspan="2">方法</td><td colspan="2">10%训练集</td><td colspan="2">20%训练集</td><td colspan="2">40%训练集</td><td colspan="2">80%训练集</td><td colspan="2">100%训练集</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>准确率</td><td style='text-align: center; word-wrap: break-word;'>MAdds</td><td style='text-align: center; word-wrap: break-word;'>准确率</td><td style='text-align: center; word-wrap: break-word;'>MAdds</td><td style='text-align: center; word-wrap: break-word;'>准确率</td><td style='text-align: center; word-wrap: break-word;'>MAdds</td><td style='text-align: center; word-wrap: break-word;'>准确率</td><td style='text-align: center; word-wrap: break-word;'>MAdds</td><td style='text-align: center; word-wrap: break-word;'>准确率</td><td style='text-align: center; word-wrap: break-word;'>MAdds</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>MobileNetV2</td><td style='text-align: center; word-wrap: break-word;'>52.72</td><td style='text-align: center; word-wrap: break-word;'>300</td><td style='text-align: center; word-wrap: break-word;'>64.02</td><td style='text-align: center; word-wrap: break-word;'>300</td><td style='text-align: center; word-wrap: break-word;'>72.08</td><td style='text-align: center; word-wrap: break-word;'>300</td><td style='text-align: center; word-wrap: break-word;'>78.36</td><td style='text-align: center; word-wrap: break-word;'>300</td><td style='text-align: center; word-wrap: break-word;'>79.60</td><td style='text-align: center; word-wrap: break-word;'>300</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>MobileNetV2 (1.4 $\times$)</td><td style='text-align: center; word-wrap: break-word;'>54.36</td><td style='text-align: center; word-wrap: break-word;'>560</td><td style='text-align: center; word-wrap: break-word;'>64.82</td><td style='text-align: center; word-wrap: break-word;'>560</td><td style='text-align: center; word-wrap: break-word;'>73.02</td><td style='text-align: center; word-wrap: break-word;'>560</td><td style='text-align: center; word-wrap: break-word;'>78.88</td><td style='text-align: center; word-wrap: break-word;'>560</td><td style='text-align: center; word-wrap: break-word;'>80.76</td><td style='text-align: center; word-wrap: break-word;'>560</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>ResNet18</td><td style='text-align: center; word-wrap: break-word;'>52.72</td><td style='text-align: center; word-wrap: break-word;'>1,814</td><td style='text-align: center; word-wrap: break-word;'>62.74</td><td style='text-align: center; word-wrap: break-word;'>1,814</td><td style='text-align: center; word-wrap: break-word;'>71.54</td><td style='text-align: center; word-wrap: break-word;'>1,814</td><td style='text-align: center; word-wrap: break-word;'>77.54</td><td style='text-align: center; word-wrap: break-word;'>1,814</td><td style='text-align: center; word-wrap: break-word;'>79.30</td><td style='text-align: center; word-wrap: break-word;'>1,814</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>ResNet50</td><td style='text-align: center; word-wrap: break-word;'>38.86</td><td style='text-align: center; word-wrap: break-word;'>4,087</td><td style='text-align: center; word-wrap: break-word;'>53.26</td><td style='text-align: center; word-wrap: break-word;'>4,087</td><td style='text-align: center; word-wrap: break-word;'>66.76</td><td style='text-align: center; word-wrap: break-word;'>4,087</td><td style='text-align: center; word-wrap: break-word;'>78.62</td><td style='text-align: center; word-wrap: break-word;'>4,087</td><td style='text-align: center; word-wrap: break-word;'>80.30</td><td style='text-align: center; word-wrap: break-word;'>4,087</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>MnasNet-A1</td><td style='text-align: center; word-wrap: break-word;'>50.30</td><td style='text-align: center; word-wrap: break-word;'>323</td><td style='text-align: center; word-wrap: break-word;'>61.14</td><td style='text-align: center; word-wrap: break-word;'>323</td><td style='text-align: center; word-wrap: break-word;'>71.46</td><td style='text-align: center; word-wrap: break-word;'>323</td><td style='text-align: center; word-wrap: break-word;'>78.82</td><td style='text-align: center; word-wrap: break-word;'>323</td><td style='text-align: center; word-wrap: break-word;'>79.66</td><td style='text-align: center; word-wrap: break-word;'>323</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>EfficientNet-B0</td><td style='text-align: center; word-wrap: break-word;'>50.56</td><td style='text-align: center; word-wrap: break-word;'>398</td><td style='text-align: center; word-wrap: break-word;'>62.90</td><td style='text-align: center; word-wrap: break-word;'>398</td><td style='text-align: center; word-wrap: break-word;'>72.32</td><td style='text-align: center; word-wrap: break-word;'>398</td><td style='text-align: center; word-wrap: break-word;'>79.42</td><td style='text-align: center; word-wrap: break-word;'>398</td><td style='text-align: center; word-wrap: break-word;'>80.38</td><td style='text-align: center; word-wrap: break-word;'>398</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Meta-NAS</td><td style='text-align: center; word-wrap: break-word;'>51.00</td><td style='text-align: center; word-wrap: break-word;'>559</td><td style='text-align: center; word-wrap: break-word;'>60.00</td><td style='text-align: center; word-wrap: break-word;'>559</td><td style='text-align: center; word-wrap: break-word;'>69.30</td><td style='text-align: center; word-wrap: break-word;'>559</td><td style='text-align: center; word-wrap: break-word;'>77.08</td><td style='text-align: center; word-wrap: break-word;'>559</td><td style='text-align: center; word-wrap: break-word;'>77.48</td><td style='text-align: center; word-wrap: break-word;'>559</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>D-EfficientNets</td><td style='text-align: center; word-wrap: break-word;'>49.08</td><td style='text-align: center; word-wrap: break-word;'>145</td><td style='text-align: center; word-wrap: break-word;'>61.42</td><td style='text-align: center; word-wrap: break-word;'>178</td><td style='text-align: center; word-wrap: break-word;'>71.28</td><td style='text-align: center; word-wrap: break-word;'>203</td><td style='text-align: center; word-wrap: break-word;'>78.26</td><td style='text-align: center; word-wrap: break-word;'>229</td><td style='text-align: center; word-wrap: break-word;'>80.44</td><td style='text-align: center; word-wrap: break-word;'>278</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Progressive NN</td><td style='text-align: center; word-wrap: break-word;'>54.60</td><td style='text-align: center; word-wrap: break-word;'>149</td><td style='text-align: center; word-wrap: break-word;'>61.72</td><td style='text-align: center; word-wrap: break-word;'>181</td><td style='text-align: center; word-wrap: break-word;'>71.38</td><td style='text-align: center; word-wrap: break-word;'>203</td><td style='text-align: center; word-wrap: break-word;'>78.78</td><td style='text-align: center; word-wrap: break-word;'>244</td><td style='text-align: center; word-wrap: break-word;'>80.02</td><td style='text-align: center; word-wrap: break-word;'>261</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>DEN</td><td style='text-align: center; word-wrap: break-word;'>54.60</td><td style='text-align: center; word-wrap: break-word;'>149</td><td style='text-align: center; word-wrap: break-word;'>63.50</td><td style='text-align: center; word-wrap: break-word;'>258</td><td style='text-align: center; word-wrap: break-word;'>72.20</td><td style='text-align: center; word-wrap: break-word;'>315</td><td style='text-align: center; word-wrap: break-word;'>78.85</td><td style='text-align: center; word-wrap: break-word;'>439</td><td style='text-align: center; word-wrap: break-word;'>80.84</td><td style='text-align: center; word-wrap: break-word;'>515</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>AdaXpert (ours)</td><td style='text-align: center; word-wrap: break-word;'>54.60</td><td style='text-align: center; word-wrap: break-word;'>149</td><td style='text-align: center; word-wrap: break-word;'>64.90</td><td style='text-align: center; word-wrap: break-word;'>171</td><td style='text-align: center; word-wrap: break-word;'>73.28</td><td style='text-align: center; word-wrap: break-word;'>199</td><td style='text-align: center; word-wrap: break-word;'>79.28</td><td style='text-align: center; word-wrap: break-word;'>232</td><td style='text-align: center; word-wrap: break-word;'>80.74</td><td style='text-align: center; word-wrap: break-word;'>252</td></tr></table>


倒置Mobile Block搜索空间（与我们相同）中搜索并达到了最先进的性能。我们还将方法与Meta-NAS（Shaw et al., 2019）进行比较，该方法首先在多个任务上搜索元架构，然后将其自适应到ImageNet。3）动态神经网络。Progressive NN（Rusu et al., 2016）和DEN（Yoon et al., 2018）是两种随着数据增长动态调整网络架构从小到大 的方法。D-EfficientNets是一种宽度乘数方法，其中网络以EfficientNet-B0（Tan & Le, 2019）的不同宽度重新缩放以适应相应的数据。更多实现细节请参阅补充材料。

### 4.2. 场景I：相同标签空间下的增长数据

在本节中，我们在数据量增长而标签空间保持不变的数据增长场景上进行实验。

增长数据的模拟：由于在ImageNet-1000上评估所有考虑的架构计算成本高昂，我们在ImageNet-100上模拟数据量增长场景。具体来说，数据以不同比例到来，即 $\{10\%, 20\%, 40\%, 80\%, 100\%\}$，每次数据增长场景的类别数保持不变。这里，较小比例的数据集是较大比例数据集的子集。

与最先进方法的比较。如表1所示，我们的方法在所有情况下均取得了最好或相当的准确率，表明了其有效性。更关键的是，我们模型的计算成本（即MAdds）显著低于其他方法，验证了所提出的奖励函数能够兼顾模型效率。具体来说，我们的模型以与DEN相当的准确率（80.74 vs. 80.84），但计算成本要低得多（252M vs. 515M）。类似的现象在表1中被广泛观察到，表明我们的方法比其他最先进方法取得了更好的准确率/计算效率权衡。

### 4.3. 场景II：标签空间增加下的增长数据

在本节中，我们在标签空间增长的数据增长场景上进行实验，即新数据比之前的数据有更多的类别。这个场景更具挑战性，因为新数据的分布可能与之前的数据有显著差异（即新类别）。实验数据集构建如下。

增长数据的模拟：在此实验中，我们使用整个ImageNet-1000来构建增长数据集。与场景I类似，数据集增长五次，每个子集分别包含整个ImageNet的前 $\{10, 20, 40, 80, 100, 200, 1000\}$ 个类别。同样，后一种情况的数据包含前一种情况的所有数据。

与最先进方法的比较：如表2所示，我们的方法能够在计算成本低得多的情况下取得相当的性能。具体来说，对于ImageNet-1000，我们的方法在准确率方面优于所有基线方法。对于ImageNet-100，我们的方法优于手动设计的网络如ResNet50，同时所需MAdds减少了15.7倍（257M vs. 4087M）。值得注意的是，当类别数较少时，ResNet18明显优于ResNet50。然而，随着类别数的增加，情况发生逆转，这验证了我们的动机，即在不同的数据分布下最优架构可能不同。

### 4.4. 消融实验

公式(4)中自适应条件的有效性。我们进行实验以进一步证明架构调整自适应条件的有效性。具体来说，我们首先准备一个基础数据集 $\mathcal{D}_{b}$（场景I中ImageNet-100的20%训练集）和一个在 $\mathcal{D}_{b}$ 上训练的模型。为了证明架构调整的必要性，我们考虑两个不同的数据集，即 $\mathcal{D}_{s}$（与 $\mathcal{D}_{b}$ 差异较小）和 $\mathcal{D}_{l}$（与 $\mathcal{D}_{b}$ 差异较大）。为了构建 $\mathcal{D}_{s}$，我们在基础数据集 $\mathcal{D}_{b}$ 上应用数据增强技术。为了构建 $\mathcal{D}_{l}$，我们使用来自ImageNet-100另外20%训练集的样本。

<div style="text-align: center;"><div style="text-align: center;">表2. 场景II：在不同类别数ImageNet-1000上的比较。我们报告准确率 $\%$（$\uparrow$）和MAdds（百万，$\downarrow$）。</div> </div>




<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td rowspan="2">方法</td><td colspan="2">10类</td><td colspan="2">20类</td><td colspan="2">40类</td><td colspan="2">80类</td><td colspan="2">100类</td><td colspan="2">200类</td><td colspan="2">1000类</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>准确率</td><td style='text-align: center; word-wrap: break-word;'>MAdds</td><td style='text-align: center; word-wrap: break-word;'>准确率</td><td style='text-align: center; word-wrap: break-word;'>MAdds</td><td style='text-align: center; word-wrap: break-word;'>准确率</td><td style='text-align: center; word-wrap: break-word;'>MAdds</td><td style='text-align: center; word-wrap: break-word;'>准确率</td><td style='text-align: center; word-wrap: break-word;'>MAdds</td><td style='text-align: center; word-wrap: break-word;'>准确率</td><td style='text-align: center; word-wrap: break-word;'>MAdds</td><td style='text-align: center; word-wrap: break-word;'>准确率</td><td style='text-align: center; word-wrap: break-word;'>MAdds</td><td style='text-align: center; word-wrap: break-word;'>准确率</td><td style='text-align: center; word-wrap: break-word;'>MAdds</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>MobileNetV2</td><td style='text-align: center; word-wrap: break-word;'>81.80</td><td style='text-align: center; word-wrap: break-word;'>300</td><td style='text-align: center; word-wrap: break-word;'>85.10</td><td style='text-align: center; word-wrap: break-word;'>300</td><td style='text-align: center; word-wrap: break-word;'>81.10</td><td style='text-align: center; word-wrap: break-word;'>300</td><td style='text-align: center; word-wrap: break-word;'>76.92</td><td style='text-align: center; word-wrap: break-word;'>300</td><td style='text-align: center; word-wrap: break-word;'>79.60</td><td style='text-align: center; word-wrap: break-word;'>300</td><td style='text-align: center; word-wrap: break-word;'>80.83</td><td style='text-align: center; word-wrap: break-word;'>300</td><td style='text-align: center; word-wrap: break-word;'>72.00</td><td style='text-align: center; word-wrap: break-word;'>300</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>MobileNetV2 (1.4\times)</td><td style='text-align: center; word-wrap: break-word;'>81.00</td><td style='text-align: center; word-wrap: break-word;'>560</td><td style='text-align: center; word-wrap: break-word;'>85.70</td><td style='text-align: center; word-wrap: break-word;'>560</td><td style='text-align: center; word-wrap: break-word;'>81.30</td><td style='text-align: center; word-wrap: break-word;'>560</td><td style='text-align: center; word-wrap: break-word;'>77.85</td><td style='text-align: center; word-wrap: break-word;'>560</td><td style='text-align: center; word-wrap: break-word;'>80.76</td><td style='text-align: center; word-wrap: break-word;'>560</td><td style='text-align: center; word-wrap: break-word;'>81.90</td><td style='text-align: center; word-wrap: break-word;'>560</td><td style='text-align: center; word-wrap: break-word;'>74.70</td><td style='text-align: center; word-wrap: break-word;'>560</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>ResNet18</td><td style='text-align: center; word-wrap: break-word;'>82.80</td><td style='text-align: center; word-wrap: break-word;'>1,814</td><td style='text-align: center; word-wrap: break-word;'>85.90</td><td style='text-align: center; word-wrap: break-word;'>1,814</td><td style='text-align: center; word-wrap: break-word;'>81.90</td><td style='text-align: center; word-wrap: break-word;'>1,814</td><td style='text-align: center; word-wrap: break-word;'>75.60</td><td style='text-align: center; word-wrap: break-word;'>1,814</td><td style='text-align: center; word-wrap: break-word;'>79.30</td><td style='text-align: center; word-wrap: break-word;'>1,814</td><td style='text-align: center; word-wrap: break-word;'>79.89</td><td style='text-align: center; word-wrap: break-word;'>1,814</td><td style='text-align: center; word-wrap: break-word;'>72.12</td><td style='text-align: center; word-wrap: break-word;'>1,814</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>ResNet50</td><td style='text-align: center; word-wrap: break-word;'>69.60</td><td style='text-align: center; word-wrap: break-word;'>4,087</td><td style='text-align: center; word-wrap: break-word;'>81.80</td><td style='text-align: center; word-wrap: break-word;'>4,087</td><td style='text-align: center; word-wrap: break-word;'>78.85</td><td style='text-align: center; word-wrap: break-word;'>4,087</td><td style='text-align: center; word-wrap: break-word;'>76.52</td><td style='text-align: center; word-wrap: break-word;'>4,087</td><td style='text-align: center; word-wrap: break-word;'>80.30</td><td style='text-align: center; word-wrap: break-word;'>4,087</td><td style='text-align: center; word-wrap: break-word;'>82.89</td><td style='text-align: center; word-wrap: break-word;'>4,087</td><td style='text-align: center; word-wrap: break-word;'>77.15</td><td style='text-align: center; word-wrap: break-word;'>4,087</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>MnasNet-A1</td><td style='text-align: center; word-wrap: break-word;'>80.60</td><td style='text-align: center; word-wrap: break-word;'>323</td><td style='text-align: center; word-wrap: break-word;'>84.60</td><td style='text-align: center; word-wrap: break-word;'>323</td><td style='text-align: center; word-wrap: break-word;'>80.80</td><td style='text-align: center; word-wrap: break-word;'>323</td><td style='text-align: center; word-wrap: break-word;'>77.03</td><td style='text-align: center; word-wrap: break-word;'>323</td><td style='text-align: center; word-wrap: break-word;'>79.66</td><td style='text-align: center; word-wrap: break-word;'>323</td><td style='text-align: center; word-wrap: break-word;'>81.95</td><td style='text-align: center; word-wrap: break-word;'>323</td><td style='text-align: center; word-wrap: break-word;'>75.20</td><td style='text-align: center; word-wrap: break-word;'>323</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>EfficientNet-B0</td><td style='text-align: center; word-wrap: break-word;'>81.40</td><td style='text-align: center; word-wrap: break-word;'>398</td><td style='text-align: center; word-wrap: break-word;'>86.00</td><td style='text-align: center; word-wrap: break-word;'>398</td><td style='text-align: center; word-wrap: break-word;'>82.10</td><td style='text-align: center; word-wrap: break-word;'>398</td><td style='text-align: center; word-wrap: break-word;'>77.70</td><td style='text-align: center; word-wrap: break-word;'>398</td><td style='text-align: center; word-wrap: break-word;'>80.38</td><td style='text-align: center; word-wrap: break-word;'>398</td><td style='text-align: center; word-wrap: break-word;'>82.49</td><td style='text-align: center; word-wrap: break-word;'>398</td><td style='text-align: center; word-wrap: break-word;'>76.30</td><td style='text-align: center; word-wrap: break-word;'>398</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Meta-NAS</td><td style='text-align: center; word-wrap: break-word;'>81.20</td><td style='text-align: center; word-wrap: break-word;'>559</td><td style='text-align: center; word-wrap: break-word;'>85.50</td><td style='text-align: center; word-wrap: break-word;'>559</td><td style='text-align: center; word-wrap: break-word;'>80.75</td><td style='text-align: center; word-wrap: break-word;'>559</td><td style='text-align: center; word-wrap: break-word;'>75.03</td><td style='text-align: center; word-wrap: break-word;'>559</td><td style='text-align: center; word-wrap: break-word;'>77.48</td><td style='text-align: center; word-wrap: break-word;'>559</td><td style='text-align: center; word-wrap: break-word;'>80.53</td><td style='text-align: center; word-wrap: break-word;'>559</td><td style='text-align: center; word-wrap: break-word;'>74.30</td><td style='text-align: center; word-wrap: break-word;'>559</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>D-EfficientNets</td><td style='text-align: center; word-wrap: break-word;'>81.00</td><td style='text-align: center; word-wrap: break-word;'>145</td><td style='text-align: center; word-wrap: break-word;'>84.20</td><td style='text-align: center; word-wrap: break-word;'>178</td><td style='text-align: center; word-wrap: break-word;'>80.40</td><td style='text-align: center; word-wrap: break-word;'>203</td><td style='text-align: center; word-wrap: break-word;'>76.63</td><td style='text-align: center; word-wrap: break-word;'>229</td><td style='text-align: center; word-wrap: break-word;'>80.44</td><td style='text-align: center; word-wrap: break-word;'>278</td><td style='text-align: center; word-wrap: break-word;'>82.03</td><td style='text-align: center; word-wrap: break-word;'>319</td><td style='text-align: center; word-wrap: break-word;'>76.30</td><td style='text-align: center; word-wrap: break-word;'>398</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Progressive NN</td><td style='text-align: center; word-wrap: break-word;'>81.20</td><td style='text-align: center; word-wrap: break-word;'>149</td><td style='text-align: center; word-wrap: break-word;'>86.10</td><td style='text-align: center; word-wrap: break-word;'>181</td><td style='text-align: center; word-wrap: break-word;'>81.10</td><td style='text-align: center; word-wrap: break-word;'>203</td><td style='text-align: center; word-wrap: break-word;'>76.43</td><td style='text-align: center; word-wrap: break-word;'>244</td><td style='text-align: center; word-wrap: break-word;'>80.02</td><td style='text-align: center; word-wrap: break-word;'>261</td><td style='text-align: center; word-wrap: break-word;'>82.20</td><td style='text-align: center; word-wrap: break-word;'>329</td><td style='text-align: center; word-wrap: break-word;'>77.53</td><td style='text-align: center; word-wrap: break-word;'>427</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>DEN</td><td style='text-align: center; word-wrap: break-word;'>81.20</td><td style='text-align: center; word-wrap: break-word;'>149</td><td style='text-align: center; word-wrap: break-word;'>86.10</td><td style='text-align: center; word-wrap: break-word;'>258</td><td style='text-align: center; word-wrap: break-word;'>81.35</td><td style='text-align: center; word-wrap: break-word;'>315</td><td style='text-align: center; word-wrap: break-word;'>77.60</td><td style='text-align: center; word-wrap: break-word;'>439</td><td style='text-align: center; word-wrap: break-word;'>80.84</td><td style='text-align: center; word-wrap: break-word;'>515</td><td style='text-align: center; word-wrap: break-word;'>82.09</td><td style='text-align: center; word-wrap: break-word;'>549</td><td style='text-align: center; word-wrap: break-word;'>72.99</td><td style='text-align: center; word-wrap: break-word;'>672</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>AdaXpert (ours)</td><td style='text-align: center; word-wrap: break-word;'>81.20</td><td style='text-align: center; word-wrap: break-word;'>149</td><td style='text-align: center; word-wrap: break-word;'>86.40</td><td style='text-align: center; word-wrap: break-word;'>176</td><td style='text-align: center; word-wrap: break-word;'>81.90</td><td style='text-align: center; word-wrap: break-word;'>195</td><td style='text-align: center; word-wrap: break-word;'>77.68</td><td style='text-align: center; word-wrap: break-word;'>242</td><td style='text-align: center; word-wrap: break-word;'>80.52</td><td style='text-align: center; word-wrap: break-word;'>257</td><td style='text-align: center; word-wrap: break-word;'>82.12</td><td style='text-align: center; word-wrap: break-word;'>293</td><td style='text-align: center; word-wrap: break-word;'>78.13</td><td style='text-align: center; word-wrap: break-word;'>395</td></tr></table>

<div style="text-align: center;"><div style="text-align: center;">表3. 自适应条件的消融实验。我们分别报告在新当前数据集 $\mathcal{D}_b \cup \mathcal{D}_s$ 和 $\mathcal{D}_b \cup \mathcal{D}_l$ 上未调整模型和调整后模型的准确率。</div> </div>




<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>数据集</td><td style='text-align: center; word-wrap: break-word;'>$H_{t}$ (公式4)</td><td style='text-align: center; word-wrap: break-word;'>未调整 (准确率 %)</td><td style='text-align: center; word-wrap: break-word;'>调整后 (准确率 %)</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>$\mathcal{D}_{b} \cup \mathcal{D}_{s}$</td><td style='text-align: center; word-wrap: break-word;'>0.65</td><td style='text-align: center; word-wrap: break-word;'>64.90</td><td style='text-align: center; word-wrap: break-word;'>65.04 (+0.14)</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>$\mathcal{D}_{b} \cup \mathcal{D}_{l}$</td><td style='text-align: center; word-wrap: break-word;'>7.76</td><td style='text-align: center; word-wrap: break-word;'>72.54</td><td style='text-align: center; word-wrap: break-word;'>73.58 (+1.04)</td></tr></table>

<div style="text-align: center;"><div style="text-align: center;">表4. 在ImageNet-100上与NAS-for-each（NFE）的比较。NFE表示"每次数据增长从头搜索"。</div> </div>




<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>指标</td><td style='text-align: center; word-wrap: break-word;'>方法</td><td style='text-align: center; word-wrap: break-word;'>20%数据</td><td style='text-align: center; word-wrap: break-word;'>40%数据</td><td style='text-align: center; word-wrap: break-word;'>80%数据</td><td style='text-align: center; word-wrap: break-word;'>100%数据</td></tr><tr><td rowspan="2">准确率 (%)</td><td style='text-align: center; word-wrap: break-word;'>NFE</td><td style='text-align: center; word-wrap: break-word;'>64.80</td><td style='text-align: center; word-wrap: break-word;'>73.46</td><td style='text-align: center; word-wrap: break-word;'>78.88</td><td style='text-align: center; word-wrap: break-word;'>80.62</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>AdaXpert (ours)</td><td style='text-align: center; word-wrap: break-word;'>64.90</td><td style='text-align: center; word-wrap: break-word;'>73.28</td><td style='text-align: center; word-wrap: break-word;'>79.28</td><td style='text-align: center; word-wrap: break-word;'>80.74</td></tr><tr><td rowspan="2">MAdds (M)</td><td style='text-align: center; word-wrap: break-word;'>NFE</td><td style='text-align: center; word-wrap: break-word;'>294</td><td style='text-align: center; word-wrap: break-word;'>302</td><td style='text-align: center; word-wrap: break-word;'>313</td><td style='text-align: center; word-wrap: break-word;'>311</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>AdaXpert (ours)</td><td style='text-align: center; word-wrap: break-word;'>171</td><td style='text-align: center; word-wrap: break-word;'>199</td><td style='text-align: center; word-wrap: break-word;'>232</td><td style='text-align: center; word-wrap: break-word;'>252</td></tr><tr><td rowspan="2">搜索成本 (GPU天)</td><td style='text-align: center; word-wrap: break-word;'>NFE</td><td style='text-align: center; word-wrap: break-word;'>0.8</td><td style='text-align: center; word-wrap: break-word;'>1.0</td><td style='text-align: center; word-wrap: break-word;'>1.3</td><td style='text-align: center; word-wrap: break-word;'>1.5</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>AdaXpert (ours)</td><td style='text-align: center; word-wrap: break-word;'>0.8</td><td style='text-align: center; word-wrap: break-word;'>0.6</td><td style='text-align: center; word-wrap: break-word;'>0.6</td><td style='text-align: center; word-wrap: break-word;'>0.7</td></tr></table>

如表3所示，我们报告了新当前数据 $\mathcal{D}_b \cup \mathcal{D}_s$ 和 $\mathcal{D}_b \cup \mathcal{D}_l$ 上的准确率差异（基于公式4）、未调整/调整后模型的准确率。从结果来看，对于相似的新数据，无需调整之前的模型架构，因为提升有限（即准确率：64.90 vs. 65.04）。相比之下，为新的不同数据调整之前的架构能够获得更大的性能提升（即准确率：72.54 vs. 73.58）。从这个意义上说，使用自适应条件来判断新数据是否需要调整是很重要的。

与从头搜索的比较。为了进一步验证AdaXpert的优越性，我们还将其与"每次数据增长从头搜索（即NAS-for-Each，NFE）"进行比较。从表4可以看出，我们的AdaXpert实现了更好的效率。在100%数据快照处，AdaXpert获得的架构性能优于NFE，这主要得益于以下两个方面：1）AdaXpert利用之前学到的知识来指导当前的学习。GAN中类似的思想（如Progressive GAN（Karras et al., 2018））和NAS中（如PNAS（Karras et al., 2018）和CNAS（Guo et al., 2020b））已被证明非常有效。2）AdaXpert考虑了当前数据与之前数据之间的差异程度，从而自适应地控制调整后模型的计算成本。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//54d4b105-ae3b-41ac-a9fc-7a76ee6347b5/markdown_2/imgs/img_in_chart_box_614_516_1074_856.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-11T08%3A45%3A46Z%2F-1%2F%2F0289cdf3b6a3cfbed329242bfe12448276e786a80230eab276a6eda876d75edb" alt="Image" width="37%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图3. AdaXpert与最先进NAS方法在ImageNet上的比较。'AdaXpert-#'表示在ImageNet-#上搜索到的架构。</div> </div>


### 4.5. 在ImageNet-1000上的比较

我们提出的方法也可以被视为一种神经架构搜索（NAS）方法，在增长数据集上渐进式地搜索最优架构。在本节中，我们将中间架构（即AdaXpert-100、AdaXpert-200和AdaXpert-1000）与现有NAS方法进行比较，以进一步验证方法的有效性。此处，AdaXpert-#是在ImageNet-#上搜索得到的。我们在整个ImageNet-1000上重新训练每个AdaXpert-#模型以及基线方法。

<div style="text-align: center;"><div style="text-align: center;">表5. 不同架构在ImageNet-1000上的比较。我们的AdaXpert-#架构在ImageNet的不同子集上搜索，然后在整个ImageNet数据集上评估。"-"表示无法获取的结果。</div> </div>




<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td rowspan="2">架构</td><td colspan="2">测试准确率 (%)</td><td rowspan="2">MAdds (M)</td><td rowspan="2">搜索时间 (GPU天)</td><td rowspan="2">搜索方法</td><td rowspan="2">搜索空间</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Top-1</td><td style='text-align: center; word-wrap: break-word;'>Top-5</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>ResNet-18 (He et al., 2016)</td><td style='text-align: center; word-wrap: break-word;'>69.8</td><td style='text-align: center; word-wrap: break-word;'>89.1</td><td style='text-align: center; word-wrap: break-word;'>1,814</td><td style='text-align: center; word-wrap: break-word;'>-</td><td rowspan="3">手动设计</td><td rowspan="3">-</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>MobileNetV2 $1.4\times$ (Sandler et al., 2018)</td><td style='text-align: center; word-wrap: break-word;'>74.7</td><td style='text-align: center; word-wrap: break-word;'>-</td><td style='text-align: center; word-wrap: break-word;'>585</td><td style='text-align: center; word-wrap: break-word;'>-</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>ShuffleNetV2 $2\times$ (Ma et al., 2018)</td><td style='text-align: center; word-wrap: break-word;'>73.7</td><td style='text-align: center; word-wrap: break-word;'>-</td><td style='text-align: center; word-wrap: break-word;'>524</td><td style='text-align: center; word-wrap: break-word;'>-</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>NASNet-A (Zoph et al., 2018)</td><td style='text-align: center; word-wrap: break-word;'>74.0</td><td style='text-align: center; word-wrap: break-word;'>91.6</td><td style='text-align: center; word-wrap: break-word;'>564</td><td style='text-align: center; word-wrap: break-word;'>1,800</td><td rowspan="2">基于RL的进化</td><td rowspan="2">NASNet</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>AmoebaNet-A (Real et al., 2019)</td><td style='text-align: center; word-wrap: break-word;'>74.5</td><td style='text-align: center; word-wrap: break-word;'>92.0</td><td style='text-align: center; word-wrap: break-word;'>555</td><td style='text-align: center; word-wrap: break-word;'>3,150</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>DARTS (Liu et al., 2019)</td><td style='text-align: center; word-wrap: break-word;'>73.1</td><td style='text-align: center; word-wrap: break-word;'>91.0</td><td style='text-align: center; word-wrap: break-word;'>595</td><td style='text-align: center; word-wrap: break-word;'>4</td><td rowspan="2">基于梯度</td><td rowspan="2">DARTS</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>P-DARTS (Chen et al., 2019)</td><td style='text-align: center; word-wrap: break-word;'>75.6</td><td style='text-align: center; word-wrap: break-word;'>92.6</td><td style='text-align: center; word-wrap: break-word;'>577</td><td style='text-align: center; word-wrap: break-word;'>0.3</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>PC-DARTS (Xu et al., 2020)</td><td style='text-align: center; word-wrap: break-word;'>75.8</td><td style='text-align: center; word-wrap: break-word;'>92.7</td><td style='text-align: center; word-wrap: break-word;'>597</td><td style='text-align: center; word-wrap: break-word;'>3.8</td><td style='text-align: center; word-wrap: break-word;'>基于梯度</td><td style='text-align: center; word-wrap: break-word;'>DARTS</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>MobileNetV3-Large (Howard et al., 2019)</td><td style='text-align: center; word-wrap: break-word;'>75.2</td><td style='text-align: center; word-wrap: break-word;'>-</td><td style='text-align: center; word-wrap: break-word;'>219</td><td style='text-align: center; word-wrap: break-word;'>-</td><td rowspan="2">基于RL/梯度</td><td rowspan="4">Mobile Block</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>FBNet-C (Wu et al., 2019)</td><td style='text-align: center; word-wrap: break-word;'>74.9</td><td style='text-align: center; word-wrap: break-word;'>-</td><td style='text-align: center; word-wrap: break-word;'>375</td><td style='text-align: center; word-wrap: break-word;'>9</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>MnasNet-A3 (Tan et al., 2019)</td><td style='text-align: center; word-wrap: break-word;'>76.7</td><td style='text-align: center; word-wrap: break-word;'>93.3</td><td style='text-align: center; word-wrap: break-word;'>403</td><td style='text-align: center; word-wrap: break-word;'>~3,791</td><td rowspan="2">基于RL/梯度</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>ProxylessNAS (Cai et al., 2019)</td><td style='text-align: center; word-wrap: break-word;'>75.1</td><td style='text-align: center; word-wrap: break-word;'>92.3</td><td style='text-align: center; word-wrap: break-word;'>465</td><td style='text-align: center; word-wrap: break-word;'>8.3</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>SPOS (Guo et al., 2020c)</td><td style='text-align: center; word-wrap: break-word;'>74.4</td><td style='text-align: center; word-wrap: break-word;'>91.8</td><td style='text-align: center; word-wrap: break-word;'>323</td><td style='text-align: center; word-wrap: break-word;'>12</td><td style='text-align: center; word-wrap: break-word;'>进化</td><td rowspan="3">Mobile Block</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>OFA-GPU (Cai et al., 2020)</td><td style='text-align: center; word-wrap: break-word;'>76.4</td><td style='text-align: center; word-wrap: break-word;'>-</td><td style='text-align: center; word-wrap: break-word;'>397</td><td style='text-align: center; word-wrap: break-word;'>51.7</td><td style='text-align: center; word-wrap: break-word;'>进化</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>OFA-CPU (Cai et al., 2020)</td><td style='text-align: center; word-wrap: break-word;'>78.7</td><td style='text-align: center; word-wrap: break-word;'>-</td><td style='text-align: center; word-wrap: break-word;'>356</td><td style='text-align: center; word-wrap: break-word;'>51.7</td><td style='text-align: center; word-wrap: break-word;'>进化</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>AtomNAS (Mei et al., 2020)</td><td style='text-align: center; word-wrap: break-word;'>75.9</td><td style='text-align: center; word-wrap: break-word;'>92.0</td><td style='text-align: center; word-wrap: break-word;'>367</td><td style='text-align: center; word-wrap: break-word;'>-</td><td style='text-align: center; word-wrap: break-word;'>基于梯度</td><td rowspan="3">Mobile Block</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>DNA-c (Li et al., 2020)</td><td style='text-align: center; word-wrap: break-word;'>77.8</td><td style='text-align: center; word-wrap: break-word;'>93.7</td><td style='text-align: center; word-wrap: break-word;'>466</td><td style='text-align: center; word-wrap: break-word;'>25</td><td style='text-align: center; word-wrap: break-word;'>贪心搜索</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>GreedyNAS-A (You et al., 2020)</td><td style='text-align: center; word-wrap: break-word;'>77.1</td><td style='text-align: center; word-wrap: break-word;'>93.3</td><td style='text-align: center; word-wrap: break-word;'>366</td><td style='text-align: center; word-wrap: break-word;'>8</td><td style='text-align: center; word-wrap: break-word;'>贪心搜索</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>AdaXpert-100 (ours)</td><td style='text-align: center; word-wrap: break-word;'>76.1</td><td style='text-align: center; word-wrap: break-word;'>92.7</td><td style='text-align: center; word-wrap: break-word;'>257</td><td style='text-align: center; word-wrap: break-word;'>2.5</td><td rowspan="3">基于RL</td><td rowspan="3">Mobile Block</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>AdaXpert-200 (ours)</td><td style='text-align: center; word-wrap: break-word;'>77.1</td><td style='text-align: center; word-wrap: break-word;'>93.3</td><td style='text-align: center; word-wrap: break-word;'>293</td><td style='text-align: center; word-wrap: break-word;'>3.5</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>AdaXpert-1000 (ours)</td><td style='text-align: center; word-wrap: break-word;'>78.1</td><td style='text-align: center; word-wrap: break-word;'>93.7</td><td style='text-align: center; word-wrap: break-word;'>395</td><td style='text-align: center; word-wrap: break-word;'>7</td></tr></table>

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//54d4b105-ae3b-41ac-a9fc-7a76ee6347b5/markdown_3/imgs/img_in_image_box_113_727_580_1022.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-11T08%3A45%3A47Z%2F-1%2F%2Fd6011234d425a70d8a48fd4e405093e7012756b883cf9b9dcfd14a04f04e69da" alt="Image" width="100%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图4. AdaXpert调整后架构的示意图。K和E分别表示卷积核大小和扩展比。</div> </div>


如表5和图3所示，我们的AdaXpert-1000在top-1准确率方面达到了78.1%，优于现有的手动设计架构和大多数考虑的最先进NAS模型（不同搜索空间）。令人惊讶的是，我们的中间模型AdaXpert-100和AdaXpert-200在top-1准确率方面也取得了与大多数基线方法相当的性能，但MAdds更少。其中一个可能的原因是我们利用了之前的架构并度量了增长数据的数据分布差异，而一般NAS方法从头搜索架构。我们还在图4中提供了调整后架构的可视化。根据每次数据增长的数据差异，我们的AdaXpert采用了不同的架构调整策略。虽然模型容量总体上随着数据增长而增加，但可以观察到一些冗余层可能被移除，而其他一些层的卷积核大小和扩展比可能被减小。

## 5. 结论

在本文中，我们提出了一种新的神经架构自适应方法，能够为增长数据高效地自适应合适的神经架构。与忽略之前架构知识的现有方法不同，我们的方法利用之前的架构以及当前数据与之前数据之间的数据差异程度来实现有效的自适应。此外，我们设计了一种自适应条件来避免不必要的调整，从而进一步提高了网络自适应效率。实验结果表明，我们的方法在两种数据增长场景（数据量增加或类别数增加）中取得了最先进的性能，同时享有较低的计算成本。更关键的是，与现有NAS方法在整个ImageNet数据集上搜索的架构相比，我们的架构能够以更少的训练数据（即ImageNet的子集）实现相当的准确率/计算成本。在未来的工作中，将我们的方法扩展为适应来自不同数据领域的增长数据的神经架构将会很有趣。

致谢。本工作部分得到了国家重点研发计划（No. 2020AAA0106900）、国家自然科学基金（NSFC）62072190、广东省重点领域研发计划（2018B010107001）、广东省引进创新创业团队计划2017ZT07X183、中央高校基本科研业务费专项资金D2191240、腾讯AI实验室犀牛鸟专项研究计划（No. JR201902）的资助。

## 参考文献

Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. In Advances in Neural Information Processing Systems, 2020.

Cai, H., Zhu, L., and Han, S. Proxylessnas: Direct neural architecture search on target task and hardware. In International Conference on Learning Representations, 2019.

Cai, H., Gan, C., Wang, Z., and Han, S. Once for all: Train one network and specialize it for efficient learning.

Cao, J., Mo, L., Zhang, Y., et al. Multi-marginal wasserstein gan. In Advances in Neural Information Processing Systems, pp. 1774–1784, 2019.

Chaudhry, A., Rohrbach, M., Elhoseiny, M., Ajanthan, T., Dokania, P., Torr, P., and Ranzato, M. Continual learning with tiny episodic memories. ArXiv, abs/1902.10486, 2019.

Chen, X., Xie, L., Wu, J., and Tian, Q. Progressive differentiable architecture search: Bridging the depth gap between search and evaluation. In IEEE International Conference on Computer Vision, pp. 1294–1303, 2019.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In IEEE Conference on Computer Vision and Pattern Recognition, pp. 248–255, 2009.

Devlin, J., Chang, M., Lee, K., and Toutanova, K. BERT: pre-training of deep bidirectional transformers for language understanding. In Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics, pp. 4171–4186, 2019.

Fuglede, B. and Topsoe, F. Jensen-shannon divergence and hilbert space embedding. In Proceedings of the IEEE International Symposium on Information Theory, 2004.

Gao, Q., Luo, Z., and Klabjan, D. Efficient architecture search for continual learning. ArXiv, abs/2006.04027, 2020.

Grantz, K., Meredith, H. R., Cummings, D., Metcalf, C., Grenfell, B., Giles, J., Mehta, S., Solomon, S., Labrique, A., Kishore, N., Buckee, C., and Wesolowski, A. The use of mobile phone data to inform analysis of covid-19 pandemic epidemiology. Nature Communications, 11, 2020.

Guo, Y., Chen, J., Wang, J., Chen, Q., Cao, J., Deng, Z., Xu, Y., and Tan, M. Closed-loop matters: Dual regression networks for single image super-resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5407–5416, 2020a.

Guo, Y., Chen, Y., Zheng, Y., Zhao, P., Chen, J., Huang, J., and Tan, M. Breaking the curse of space explosion: Towards efficient nas with curriculum search. In Proceedings of the International Conference on Machine Learning, 2020b.

Guo, Z., Zhang, X., Mu, H., Heng, W., Liu, Z., Wei, Y., and Sun, J. Single path one-shot neural architecture search with uniform sampling. In International Conference on Learning Representations, 2020c.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In IEEE Conference on Computer Vision and Pattern Recognition, pp. 770–778, 2016.

Hoi, S., Sahoo, D., Lu, J., and Zhao, P. Online learning: A comprehensive survey. ArXiv, abs/1802.02871, 2018.

Howard, A., Sandler, M., Chu, G., Chen, L.-C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V., et al. Searching for mobilenetv3. In IEEE International Conference on Computer Vision, pp. 1314–1324, 2019.

Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., and Adam, H. Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861, 2017.

Hu, J., Shen, L., and Sun, G. Squeeze-and-excitation networks. In IEEE Conference on Computer Vision and Pattern Recognition, pp. 7132–7141, 2018.

Karras, T., Aila, T., Laine, S., and Lehtinen, J. Progressive growing of gans for improved quality, stability, and variation. In International Conference on Learning Representations, 2018.

Kirkpatrick, J., Pascanu, R., Rabinowitz, N. C., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis, D., Clopath, C., Kumaran, D., and Hadsell, R. Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114:3521–3526, 2017.

Lange, M. D., Jia, X., Parisot, S., Leonardis, A., Slabaugh, G., and Tuytelears, T. Unsupervised model personalization while preserving privacy and scalability: An open problem. In IEEE Conference on Computer Vision and Pattern Recognition, pp. 14451–14460, 2020.

Langley, P. Crafting papers on machine learning. In Proceedings of the International Conference on Machine Learning, pp. 1207–1216, 2000.

Li, C., Peng, J., Yuan, L., Wang, G., Liang, X., Lin, L., and Chang, X. Block-wisely supervised neural architecture search with knowledge distillation. In IEEE Conference on Computer Vision and Pattern Recognition, pp. 1989–1998, 2020.

Li, X., Zhou, Y., Wu, T., Socher, R., and Xiong, C. Learn to grow: A continual structure learning framework for overcoming catastrophic forgetting. In Proceedings of the International Conference on Machine Learning, pp. 3925–3934, 2019.

Lian, D., Zheng, Y., Xu, Y.-T., Lu, Y., Lin, L., Zhao, P., Huang, J., and Gao, S. Towards fast adaptation of neural architectures with meta learning. In International Conference on Learning Representations, 2020.

Liang, H., Tsui, B., Ni, H., et al. Evaluation and accurate diagnoses of pediatric diseases using artificial intelligence. Nature Medicine, 25:433–438, 2019.

Liu, H., Simonyan, K., and Yang, Y. Darts: Differentiable architecture search. In International Conference on Learning Representations, 2019.

Liu, X., Masana, M., Herranz, L., van de Weijer, J., López, A. M., and Bagdanov, A. D. Rotate your networks: Better weight consolidation and less catastrophic forgetting. In International Conference on Pattern Recognition, pp. 2262–2268, 2018.

Lu, Z., Sreekumar, G., Goodman, E., Banzhaf, W., Deb, K., and Boddeti, V. Neural architecture transfer. IEEE transactions on pattern analysis and machine intelligence, PP, 2021.

Ma, N., Zhang, X., Zheng, H.-T., and Sun, J. ShuffleNet V2: Practical guidelines for efficient CNN architecture design. In European Conference on Computer Vision, pp. 116–131, 2018.

Mei, J., Li, Y., Lian, X., Jin, X., Yang, L., Yuille, A., and Yang, J. Atomnas: Fine-grained end-to-end neural architecture search. In International Conference on Learning Representations, 2020.

Nguyen, X., Wainwright, M. J., and Jordan, M. I. Nonparametric estimation of the likelihood ratio and divergence functionals. In IEEE International Symposium on Information Theory, pp. 2016–2020. IEEE, 2007.

Pham, H., Guan, M. Y., Zoph, B., Le, Q. V., and Dean, J. Efficient neural architecture search via parameter sharing. In Proceedings of the International Conference on Machine Learning, pp. 4092–4101, 2018.

Piergiovanni, A. J., Angelova, A., and Ryoo, M. S. Tiny video networks. arXiv preprint arXiv:1910.06961, 2019.

Real, E., Aggarwal, A., Huang, Y., and Le, Q. V. Regularized evolution for image classifier architecture search. In AAAI Conference on Artificial Intelligence, 2019.

Rebuffi, S.-A., Kolesnikov, A., Sperl, G., and Lampert, C. H. icarl: Incremental classifier and representation learning. In IEEE Conference on Computer Vision and Pattern Recognition, pp. 5533–5542, 2017.

Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T., and Wayne, G. Experience replay for continual learning. In Advances in Neural Information Processing Systems, 2019.

Rosenfeld, A. and Tsotsos, J. K. Incremental learning through deep adaptation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 42:651–663, 2020.

Rusu, A. A., Rabinowitz, N. C., Desjardins, G., Soyer, H., Kirkpatrick, J., Kavukcuoglu, K., Pascanu, R., and Hadsell, R. Progressive neural networks. ArXiv, abs/1606.04671, 2016.

Sandler, M., Howard, A. G., Zhu, M., Zhmoginov, A., and Chen, L. MobileNetV2: Inverted residuals and linear bottlenecks. In IEEE Conference on Computer Vision and Pattern Recognition, pp. 4510–4520, 2018.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Shaw, A., Wei, W., Liu, W., Song, L., and Dai, B. Meta architecture search. In Advances in Neural Information Processing Systems, pp. 11225–11235, 2019.

Sriperumbudur, B. K., Fukumizu, K., Gretton, A., Schölkopf, B., and Lanckriet, G. R. Non-parametric estimation of integral probability metrics. In IEEE International Symposium on Information Theory, pp. 1428–1432. IEEE, 2010.

Takatsu, A. et al. Wasserstein geometry of gaussian measures. Osaka Journal of Mathematics, 48(4):1005–1026, 2011.

Tan, M. and Le, Q. Efficientnet: Rethinking model scaling for convolutional neural networks. In Proceedings of the International Conference on Machine Learning, pp. 6105–6114, 2019.

Tan, M., Chen, B., Pang, R., Vasudevan, V., Sandler, M., Howard, A., and Le, Q. V. Mnasnet: Platform-aware neural architecture search for mobile. In IEEE Conference on Computer Vision and Pattern Recognition, pp. 2820–2828, 2019.

Wang, J., Wu, J., Bai, H., and Cheng, J. M-nas: Meta neural architecture search. In AAAI Conference on Artificial Intelligence, 2020.

Williams, R. J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3-4):229–256, 1992.

Wu, B., Dai, X., Zhang, P., Wang, Y., Sun, F., Wu, Y., Tian, Y., Vajda, P., Jia, Y., and Keutzer, K. FBNet: Hardware-aware efficient convnet design via differentiable neural architecture search. In IEEE Conference on Computer Vision and Pattern Recognition, pp. 10734–10742, 2019.

Xu, J. and Zhu, Z. Reinforced continual learning. In Advances in Neural Information Processing Systems, 2018.

Xu, Y., Xie, L., Zhang, X., Chen, X., Qi, G.-J., Tian, Q., and Xiong, H. PC-DARTS: Partial channel connections for memory-efficient architecture search. In International Conference on Learning Representations, 2020.

Yoon, J., Yang, E., Lee, J., and Hwang, S. J. Lifelong learning with dynamically expandable networks. In International Conference on Learning Representations, 2018.

You, S., Huang, T., Yang, M., Wang, F., Qian, C., and Zhang, C. Greedynas: Towards fast one-shot nas with greedy supernet. In IEEE Conference on Computer Vision and Pattern Recognition, pp. 1999–2008, 2020.

Zeng, R., Xu, H., Huang, W., Chen, P., Tan, M., and Gan, C. Dense regression network for video grounding. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2020.

Zhang, Y., Zhao, P., Niu, S., Wu, Q., et al. Online adaptive asymmetric active learning with limited budgets. IEEE Transactions on Knowledge and Data Engineering, 2019.

Zhang, Y., Wei, Y., Wu, Q., Zhao, P., Niu, S., Huang, J., and Tan, M. Collaborative unsupervised domain adaptation for medical image diagnosis. IEEE Transactions on Image Processing, 29:7834–7844, 2020.

Zoph, B. and Le, Q. V. Neural architecture search with reinforcement learning. In International Conference on Learning Representations, 2017.

Zoph, B., Vasudevan, V., Shlens, J., and Le, Q. V. Learning transferable architectures for scalable image recognition. In IEEE Conference on Computer Vision and Pattern Recognition, pp. 8697–8710, 2018.

arXiv:2107.00254v1 [cs.LG] 1 Jul 2021

# "AdaXpert：为增长数据自适应调整神经网络架构"补充材料

Shuaicheng Niu $^{*12}$ Jiaxiang Wu $^{*3}$ Guanghui Xu $^{1}$ Yifan Zhang $^{4}$

Yong Guo $^{1}$ Peilin Zhao $^{3}$ Peng Wang $^{5}$ Mingkui Tan $^{16}$

在补充材料中，我们提供了所提出的AdaXpert的更多实现细节和更多实验结果。补充材料的组织如下。

- 在A节中，我们提供了所提出的AdaXpert的更多实现细节。

- 在B节中，我们提供了所有比较网络架构的评估细节。

- 在C节中，我们提供了自适应条件阈值选择的更多实验结果。

- 在D节中，我们展示了Wasserstein距离在度量数据差异方面的有效性。

- 在E节中，我们提供了关于权衡参数 $\lambda$（公式(5)中）的敏感性分析。

### A. AdaXpert的实现细节

在整个数据增长过程中，我们维护一个超网络 $\mathcal{N}_t$ 和一个控制器 $\pi(\cdot; \theta_t)$。每次数据增长时，我们首先在当前数据 $\mathcal{D}_t$ 上微调之前的超网络以获得 $\mathcal{N}_t$。之后，利用超网络 $\mathcal{N}_t$ 提供的评估信号来训练控制器 $\pi(\cdot; \theta_t)$。下面介绍它们的训练细节。

超网络：对于增长场景I（主论文中），我们分别对0.2和0.4大小的ImageNet-100更新超网络180个epoch，对0.8和1.0大小的ImageNet-100更新超网络60个epoch。对于增长场景II，我们对ImageNet-20和ImageNet-40更新超网络180个epoch，对其他更新60个epoch。按照Guo et al., 2020的做法，我们通过均匀采样足够的架构并依次训练它们来训练超网络。对于每次数据增长，超网络以0.045的学习率、$5 \times 10^{-5}$ 的权重衰减和0.9的动量进行微调。

控制器：控制器以之前的架构 $\alpha_{t-1}$ 和数据之间的差异程度 $d_{t}$（公式(1)中）作为输入，然后输出调整后的架构 $\alpha_{t}^{\prime}$。下面我们首先介绍控制器的架构设计，然后介绍其训练细节。

对于控制器设计，我们首先使用一个两层全连接网络（FCN）来提取输入架构的特征。同时，为了表示数据差异的程度，按照（Pham et al., 2018）的做法，我们为不同的 $d_{t}$ 构建一个可学习的嵌入向量。然后我们将架构嵌入和数据差异嵌入拼接在一起，并送入控制器模型。我们采用LSTM来构建控制器模型。由于架构可以由一系列token表示（Zoph & Le, 2017; Pham et al., 2018），控制器能够通过依次预测token序列来调整网络架构，包括深度、宽度和卷积核大小。这里，我们将FCN参数和可学习嵌入向量纳入控制器的参数中并联合训练。

对于每次数据增长，我们训练控制器模型6k次迭代。我们使用Adam作为优化器，学习率为 $2 \times 10^{-4}$，权重衰减为 $5 \times 10^{-4}$。我们还将控制器的采样熵添加到奖励中，权重为 $2 \times 10^{-4}$。公式(5)中的权衡参数 $\lambda$ 对场景I和II分别设置为 $0.5 \times 10^{-4}$ 和 $2.5 \times 10^{-4}$。这里，$\lambda$ 的值仅在第一次调整时（例如在ImageNet-20上）调优，然后在所有后续调整中保持固定。通过这种方式，尽管后续调整的 $\lambda$ 可能不是最优的，但实验结果表明这已经取得了良好的性能。我们相信对 $\lambda$ 的仔细和高效调优可能进一步提高AdaXpert的性能，我们将其留作未来的工作。

 $^{*}$同等贡献。本工作完成于Shuaicheng Niu在腾讯AI实验室实习期间。 $^{1}$华南理工大学软件工程学院，中国 $^{2}$大数据与智能机器人教育部重点实验室，中国 $^{3}$腾讯AI实验室，中国 $^{4}$新加坡国立大学，新加坡 $^{5}$西北工业大学，中国 $^{6}$琶洲实验室，中国。通讯作者：Mingkui Tan <mingkuitan@scut.edu.cn>。

### B. 架构评估细节

在ImageNet-1000"子集"上的评估。为了公平比较，我们通过相同的设置在ImageNet-1000的子集上训练所有架构（包括我们的AdaXpert），然后在相应的测试集上测试它们。具体来说，我们训练每个架构180个epoch，批量大小为256。我们应用SGD优化器，权重衰减为 $5 \times 10^{-5}$，动量为0.9。此外，学习率从0.1开始，在第80和第130个epoch除以10。所有架构使用Tesla V100 GPU进行评估。

在"整个"ImageNet-1000上的评估。我们根据原始论文报告所有比较方法在ImageNet-1000上的性能。对于AdaXpert模型，我们使用Lu et al., 2020提供的评估方法进行评估。具体来说，为了加速评估，我们使用预训练的公开可用的once-for-all网络（Cai et al., 2020）初始化模型权重，然后微调85个epoch。微调训练采用RMSProp优化器，衰减为0.9，动量为0.9。我们将批归一化动量设置为0.99，权重衰减设置为 $1e^{-5}$。我们使用512的批量大小和0.012的初始学习率，通过余弦退火调度逐渐降至零。正则化设置与Tan & Le, 2019类似：我们使用数据增强策略（Cubuk et al., 2020）、drop connect比率0.2和dropout比率0.2。

### C. 自适应条件的阈值选择

在本节中，我们展示了自适应条件（公式(4)中）的更多结果，以帮助算法工程师选择合适的阈值 $\epsilon$ 来判断架构调整的必要性。与主论文相同，实验在两种考虑的场景上进行，即数据量增加（场景I）和类别数增加（场景II）。

对于场景I，我们报告了在ImageNet-100的0.2训练集上训练好的ResNet18、MobileNetV2和我们的AdaXpert的 $H_t$ 值（公式(4)）。然后在0.2和 $\{0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0\}$ 大小的ImageNet-100之间计算 $H_t$。对于标签空间增长的场景II，我们报告了在ImageNet-20上训练好的上述模型的 $H_t$。然后在ImageNet-20和ImageNet-$\{30, 40, 50, 60, 70, 80, 90, 100\}$ 之间计算 $H_t$。图I的左图和右图分别展示了场景I和II的结果。随着新数据的增长，之前的模型面临更严重的准确率差异。基于这些结果，可以根据手头的任务选择合适的阈值来决定是否调整模型架构。在本文中，我们将阈值 $\epsilon$ 设置为0.02。

值得一提的是，对于标签空间增加的场景II，模型性能必然会下降。但在这种情况下，用户仍然需要这个自适应条件来度量性能下降是否超过给定阈值 $\epsilon$（公式4中），然后决定是否进行调整。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//9a3a8274-6944-483e-be8a-40b63ca9fd14/markdown_0/imgs/img_in_chart_box_305_1113_584_1330.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-11T08%3A45%3A44Z%2F-1%2F%2F0f290b1ff6e6ab3b39cb994ac617b914b20cea1437de64f939d922b8d46a57b3" alt="Image" width="22%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//9a3a8274-6944-483e-be8a-40b63ca9fd14/markdown_0/imgs/img_in_chart_box_603_1117_881_1330.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-11T08%3A45%3A44Z%2F-1%2F%2F07ccdcd91f1e948e302cdd36eb3a7cc4c3ffecf229e4acd50297fa84cbd0092f" alt="Image" width="22%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图I. 不同增长数据的准确率差异 $H_{t}$（公式(4)中）示意图。左图和右图分别表示数据量增加和类别数增加。</div> </div>


### D. 关于Wasserstein距离的更多讨论

为了计算分布距离，可以使用多种度量，包括Jensen-Shannon（JS）散度（Fuglede & Topsoe, 2004）、Wasserstein距离（WD）（Sriperumbudur et al., 2010）等。在本文中，我们选择WD作为度量，因为1）它是建立几何工具来比较概率分布的有效度量，已被广泛用于深度学习（如GAN）中；2）在我们的场景中，它在识别差异程度方面比JS散度具有稍强的区分能力。在本节中，我们在两种考虑的场景上比较WD和JS，即数据量增加（场景I）和类别数增加（场景II）。1）对于场景I，基于在ImageNet-100的0.2训练集上搜索到的架构，我们计算0.2训练集与 $\{0.4, 0.6, 0.8, 1.0\}$ 训练集之间的距离 $d_t$。2）对于场景II，基于在ImageNet-20上获得的架构，我们计算ImageNet-20与ImageNet-$\{40, 60, 80, 100, 200\}$ 之间的距离。

WD和JS的计算细节。如第3.2节所述，我们首先将当前数据 $\mathcal{D}_t$ 和之前的数据 $\mathcal{D}_{t-1}$ 输入之前的模型 $\alpha_{t-1}$，然后分别获得对应的特征矩阵 $\mathbf{M}_t \in \mathbb{R}^{m \times q}$ 和 $\mathbf{M}_{t-1} \in \mathbb{R}^{n \times q}$。其中，$m$ 和 $n$ 分别表示 $\mathcal{D}_t$ 和 $\mathcal{D}_{t-1}$ 中的样本数量，$q$ 表示特征维度。我们假设 $\mathbf{M}_t$ 和 $\mathbf{M}_{t-1}$ 来自两个多元高斯分布 $\mathbb{P}_t$ 和 $\mathbb{P}_{t-1}$，并使用最大似然估计方法来估计其分布参数，即 $\mathbb{P}_t \sim \mathcal{N}(\mu_t, \Sigma_t)$ 和 $\mathbb{P}_{t-1} \sim \mathcal{N}(\mu_{t-1}, \Sigma_{t-1})$。基于上述假设，$\mathcal{D}_t$ 和 $\mathcal{D}_{t-1}$ 之间的WD（Takatsu et al., 2011）计算如下：

 $$ \begin{array}{r}{\mathcal{W}(\mathcal{D}_{t},\mathcal{D}_{t-1})=||\mu_{t}-\mu_{t-1}||_{2}^{2}+\mathrm{tr}\Big(\Sigma_{t}{+}\Sigma_{t-1}{-}2(\Sigma_{t-1}^{1/2}\Sigma_{t}\Sigma_{t-1}^{1/2})^{1/2}\Big),}\end{array} $$

JS散度（Fuglede & Topsoe, 2004）计算如下：

 $$ \begin{align*}\mathrm{JSD}(\mathcal{D}_{t},\mathcal{D}_{t-1})&=\frac{1}{2}\Big(\mathrm{KL}(\mathrm{P}_{t}||\frac{\mathrm{P}_{t}+\mathrm{P}_{t-1}}{2})+\mathrm{KL}(\mathrm{P}_{t-1}||\frac{\mathrm{P}_{t}+\mathrm{P}_{t-1}}{2})\Big),\\\mathrm{where}\quad&\mathrm{KL}(\mathrm{P}_{t}||\mathrm{P}_{t-1})=\int_{-\infty}^{+\infty}\mathrm{P}_{t}(x)\log(\frac{\mathrm{P}_{t}(x)}{\mathrm{P}_{t-1}(x)})dx.\end{align*} $$

此处，$P_{t}$ 是 $P_{t}$ 的概率密度函数。

WD和JS的比较。如图II所示，WD和JS都能够识别当前数据与之前数据之间的差异。一般来说，当前数据与之前数据的差异越大，WD和JS就越大。对于标签空间增长的场景II，WD表现出比JS更强的区分能力（见图II（右图））。具体来说，ImageNet-20和ImageNet-{100, 200}之间的JS接近1，因此无法很好地识别它们之间的差异。相比之下，WD仍然能够识别ImageNet-100和ImageNet-200之间的差异。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//9a3a8274-6944-483e-be8a-40b63ca9fd14/markdown_1/imgs/img_in_chart_box_257_994_593_1223.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-11T08%3A45%3A46Z%2F-1%2F%2Fe47d6a969b62184fec5bb256785f6c0009b637d4fb25c2fbf77a61bcccf543e6" alt="Image" width="27%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//9a3a8274-6944-483e-be8a-40b63ca9fd14/markdown_1/imgs/img_in_chart_box_613_996_952_1222.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-11T08%3A45%3A46Z%2F-1%2F%2F581de7caac35a79ac16032dcd9863344f08ce6a930ab4ccc65f4c53e506bf4bf" alt="Image" width="27%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图II. WD和JS散度的比较。</div> </div>


Wasserstein距离在AdaXpert中的有效性。在我们的方法中，我们根据当前数据与之前数据之间的Wasserstein距离（WD）执行不同的架构调整策略。这使我们的调整能够以适度的计算开销获得更好的模型精度。我们将方法与AdaXpert w/o WD进行比较，即奖励函数的目标仅为最大化模型精度。具体来说，我们在场景I的ImageNet-100上进行实验。如表A所示，随着数据的增长，'AdaXpert w/o WD'倾向于为当前数据获得更大的网络。然而，在大多数情况下，网络性能与我们的方法相当。这进一步证明了使用WD考虑数据差异来指导调整的必要性。

<div style="text-align: center;"><div style="text-align: center;">表A. Wasserstein距离在AdaXpert中的有效性。'AdaXpert w/o WD'表示在奖励函数中不考虑数据差异（WD）的AdaXpert。</div> </div>




<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>指标</td><td style='text-align: center; word-wrap: break-word;'>方法</td><td style='text-align: center; word-wrap: break-word;'>20%数据</td><td style='text-align: center; word-wrap: break-word;'>40%数据</td><td style='text-align: center; word-wrap: break-word;'>80%数据</td><td style='text-align: center; word-wrap: break-word;'>100%数据</td></tr><tr><td rowspan="2">准确率 (%)</td><td style='text-align: center; word-wrap: break-word;'>AdaXpert w/o WD</td><td style='text-align: center; word-wrap: break-word;'>65.12</td><td style='text-align: center; word-wrap: break-word;'>73.76</td><td style='text-align: center; word-wrap: break-word;'>79.10</td><td style='text-align: center; word-wrap: break-word;'>80.96</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>AdaXpert</td><td style='text-align: center; word-wrap: break-word;'>64.90</td><td style='text-align: center; word-wrap: break-word;'>73.28</td><td style='text-align: center; word-wrap: break-word;'>79.28</td><td style='text-align: center; word-wrap: break-word;'>80.74</td></tr><tr><td rowspan="2">MAdds (M)</td><td style='text-align: center; word-wrap: break-word;'>AdaXpert w/o WD</td><td style='text-align: center; word-wrap: break-word;'>279</td><td style='text-align: center; word-wrap: break-word;'>284</td><td style='text-align: center; word-wrap: break-word;'>298</td><td style='text-align: center; word-wrap: break-word;'>302</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>AdaXpert</td><td style='text-align: center; word-wrap: break-word;'>171</td><td style='text-align: center; word-wrap: break-word;'>199</td><td style='text-align: center; word-wrap: break-word;'>232</td><td style='text-align: center; word-wrap: break-word;'>252</td></tr></table>

高斯假设的讨论。在本文中，我们通过假设当前数据和之前的数据来自两个多元高斯分布来计算它们之间的WD。这里，我们从经验上证明这一假设的合理性。基于在ImageNet-20上训练好的AdaXpert-20，我们计算ImageNet-40的样本矩阵（如第3.2节所述），并随机采样6个维度来可视化其统计直方图。如图III所示，每个维度的样本特征近似满足高斯分布。为了实现更精确的WD计算，也可以使用第3.2节中描述的非参数估计方法（Sriperumbudur et al., 2010）。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//9a3a8274-6944-483e-be8a-40b63ca9fd14/markdown_2/imgs/img_in_chart_box_166_598_460_792.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-11T08%3A45%3A48Z%2F-1%2F%2F9f64a66c4e57ede60281dc25f8a562c9e410e15098c9ab56695d19fef1aaf5f2" alt="Image" width="24%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//9a3a8274-6944-483e-be8a-40b63ca9fd14/markdown_2/imgs/img_in_chart_box_472_599_769_791.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-11T08%3A45%3A48Z%2F-1%2F%2F263f187f7c7e38f2378ff5bd98db10f5224fbcca777298a199133869e8d184c1" alt="Image" width="24%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//9a3a8274-6944-483e-be8a-40b63ca9fd14/markdown_2/imgs/img_in_chart_box_781_598_1075_792.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-11T08%3A45%3A48Z%2F-1%2F%2Fe158fc6d93d93b15e68d36ba8def7fee225af38febb0cf71a62bd3c8a075281f" alt="Image" width="24%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//9a3a8274-6944-483e-be8a-40b63ca9fd14/markdown_2/imgs/img_in_chart_box_165_799_460_992.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-11T08%3A45%3A49Z%2F-1%2F%2Fd66350af93eb062621bd2c7db50f30d9c1cef5cb7c283b24f26bc74e971de914" alt="Image" width="24%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//9a3a8274-6944-483e-be8a-40b63ca9fd14/markdown_2/imgs/img_in_chart_box_471_800_769_992.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-11T08%3A45%3A49Z%2F-1%2F%2Fa55b3b3aa730c4076d3f64c2d9242344d74033ecaa86feffc295a01669c465d9" alt="Image" width="24%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//9a3a8274-6944-483e-be8a-40b63ca9fd14/markdown_2/imgs/img_in_chart_box_783_801_1071_992.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-11T08%3A45%3A49Z%2F-1%2F%2F9158ea416f0354977369b3498fc6ae788f190dc78b1996b9f19f0a5268e39859" alt="Image" width="23%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图III. 在6个随机采样维度上数据矩阵的统计直方图。</div> </div>


### E. 公式(5)中 $\lambda$ 的参数敏感性

在本节中，我们评估了公式(5)中不同权衡参数 $\lambda$ 下我们的方法，$\lambda$ 从 $\{2.0, 2.5, 3.0, 3.5\}e^{-4}$ 中选取。实验在将AdaXpert-20调整为AdaXpert-40上进行。我们在图IV（左图）和（右图）中分别报告了调整后架构的验证准确率和MAdds。

从结果来看，随着 $\lambda$ 的增加，我们的AdaXpert倾向于找到MAdds更少的架构。然而，搜索准确率（即验证准确率）在 $\lambda = 2.5e^{-4}$ 时达到最优。与 $\lambda = 2e^{-4}$ 相比，$\lambda = 2.5e^{-4}$ 在MAdds更少的情况下实现了更好的搜索性能。这个结果进一步证明，在特定数据集上，小模型能够比大模型取得更好的准确率。从这个意义上说，可以设计性能良好的架构，同时尽可能减少模型的MAdds。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//9a3a8274-6944-483e-be8a-40b63ca9fd14/markdown_3/imgs/img_in_chart_box_291_136_586_367.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-11T08%3A45%3A51Z%2F-1%2F%2Fd7b03832b3190491395b2bbef35190dd01f51a409643ba62c688e4050f760e56" alt="Image" width="24%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//9a3a8274-6944-483e-be8a-40b63ca9fd14/markdown_3/imgs/img_in_chart_box_604_134_908_370.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-11T08%3A45%3A51Z%2F-1%2F%2Fe0fc4f016a4dd677bc203411b9ff3259bd005879b061ebaef9c252ba75f52c8f" alt="Image" width="24%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图IV. 不同权衡参数 $\lambda$ 下AdaXpert的训练曲线。</div> </div>


## 参考文献

Cai, H., Gan, C., Wang, T., Zhang, Z., and Han, S. Once for all: Train one network and specialize it for efficient deployment. In International Conference on Learning Representations, 2020.

Cubuk, E. D., Zoph, B., Shlens, J., and Le, Q. V. Randaugment: Practical automated data augmentation with a reduced search space. In IEEE Conference on Computer Vision and Pattern Recognition, pp. 702–703, 2020.

Fuglede, B. and Topsoe, F. Jensen-shannon divergence and hilbert space embedding. In Proceedings of the IEEE International Symposium on Information Theory, 2004.

Guo, Z., Zhang, X., Mu, H., Heng, W., Liu, Z., Wei, Y., and Sun, J. Single path one-shot neural architecture search with uniform sampling. In International Conference on Learning Representations, 2020.

Lu, Z., Sreekumar, G., Goodman, E., Banzhaf, W., Deb, K., and Boddeti, V. N. Neural architecture transfer. arXiv preprint arXiv:2005.05859, 2020.

Pham, H., Guan, M. Y., Zoph, B., Le, Q. V., and Dean, J. Efficient neural architecture search via parameter sharing. In Proceedings of the International Conference on Machine Learning, pp. 4092–4101, 2018.

Sriperumbudur, B. K., Fukumizu, K., Gretton, A., Schölkopf, B., and Lanckriet, G. R. Non-parametric estimation of integral probability metrics. In IEEE International Symposium on Information Theory, pp. 1428–1432. IEEE, 2010.

Takatsu, A. et al. Wasserstein geometry of gaussian measures. Osaka Journal of Mathematics, 48(4):1005–1026, 2011.

Tan, M. and Le, Q. Efficientnet: Rethinking model scaling for convolutional neural networks. In Proceedings of the International Conference on Machine Learning, pp. 6105–6114, 2019.

Zoph, B. and Le, Q. V. Neural architecture search with reinforcement learning. In International Conference on Learning Representations, 2017.

