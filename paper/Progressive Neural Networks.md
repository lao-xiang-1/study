# 渐进式神经网络（Progressive Neural Networks）

Andrei A. Rusu $^{*}$, Neil C. Rabinowitz $^{*}$, Guillaume Desjardins $^{*}$, Hubert Soyer, James Kirkpatrick, Koray Kavukcuoglu, Razvan Pascanu, Raia Hadsell

 $^{*}$ 这些作者对本工作做出了同等贡献

Google DeepMind

London, UK

{andreirusu, ncr, gdesjardins, soyer, kirkpatrick, korayk, razp, raia}@google.com

## 摘要

学习求解复杂的任务序列——既利用迁移（transfer），又避免灾难性遗忘（catastrophic forgetting）——仍然是实现类人智能的一个关键障碍。渐进式网络（progressive networks）方法朝着这一方向迈出了一步：它们对遗忘免疫，并且能够通过横向连接（lateral connections）利用先前学习到的特征中的先验知识。我们在各种各样的强化学习任务（Atari 与 3D 迷宫游戏）上对该架构进行了广泛的评估，并表明其性能优于基于预训练（pretraining）与微调（finetuning）的常见基线。使用一种新颖的敏感度度量，我们证明迁移同时发生在所学策略的低层感知层与高层控制层。

## 1 引言

微调仍然是神经网络迁移学习的首选方法：模型在源域（数据通常充足）上进行预训练，模型的输出层被适配到目标域，然后通过反向传播对网络进行微调。该方法由 [7] 开创，将知识从生成模型迁移到判别模型，此后被推广并取得了巨大成功 [11]。遗憾的是，该方法存在一些缺陷，使其不适合跨多个任务进行迁移：如果我们希望利用在一系列经历中获得的多个知识，应当用哪个模型来初始化后续模型？这似乎不仅要求有一种能够支持迁移学习而不发生灾难性遗忘的学习方法，还要求其低成本、高性能。此外，虽然微调可能使我们能够在目标域中恢复专家级性能，但它是一个破坏性过程，会丢弃先前学习到的函数。人们可以在微调之前复制每个模型，但选择合适的初始化问题依然存在。对于多任务学习 [17] 而言，其潜在解决方案需要为所有任务保留一个持续存在的训练数据储备（reservoir），而这一假设并不总是成立。

本文引入了渐进式网络，一种新颖的模型架构，它显式地支持跨任务序列的迁移。微调只在初始化时纳入先验知识，而渐进式网络在整个训练过程中保留一个预训练模型池，并学习从这些模型出发的横向连接，以提取对新任务有用的特征。通过以这种方式组合先前学习到的特征，渐进式网络实现了更丰富的组合性（compositionality），其中先验知识不再是一次性的，而是可以整合到特征层级的每一层。此外，在预训练网络之外增加新的容量（capacity），使这些模型既能够复用旧的计算，又能够学习新的计算。正如我们将展示的，渐进式网络自然地积累经验，并且从设计上对灾难性遗忘免疫，这使其成为解决长期存在的持续学习（continual learning）或终身学习（lifelong learning）问题的理想跳板。

本文的贡献有三方面。虽然渐进式网络所使用的许多个别要素都能在文献中找到，但它们的组合及其在求解复杂任务序列中的运用是新颖的。第二，我们在复杂的强化学习领域中对模型进行了广泛的评估。在此过程中，我们还在强化学习领域内评估了其他迁移方法（如微调）。特别地，我们表明渐进式网络提供了与传统的微调相当（甚至略好）的迁移性能，但没有破坏性的后果。最后，我们基于费舍尔信息（Fisher Information）与扰动（perturbation）开发了一种新颖的分析方法，使我们能够详细分析迁移如何在任务间发生以及发生在何处。

## 2 渐进式网络

持续学习是机器学习的一个长期目标，其中智能体不仅能学习（并记住）按顺序经历的一系列任务，还能利用先前任务的知识来加速收敛 [20]。渐进式网络将这些不同的数据源直接整合进模型架构：通过为每个正在求解的任务实例化一个新的神经网络（一个列，column）来防止灾难性遗忘，同时通过到先前已学习列的横向连接来实现迁移。本节末尾将讨论该方法的可扩展性。

一个渐进式网络从单个列开始：一个具有 $L$ 层的深度神经网络，其隐藏激活为 $h_i^{(1)} \in \mathbb{R}^{n_i}$，其中 $n_i$ 是第 $i \leq L$ 层的单元数，参数 $\Theta^{(1)}$ 训练至收敛。当切换到第二个任务时，参数 $\Theta^{(1)}$ 被“冻结”，并实例化一个带有参数 $\Theta^{(2)}$ 的新列（随机初始化），其中 $h_i^{(2)}$ 层经由横向连接同时接收来自 $h_{i-1}^{(2)}$ 和 $h_{i-1}^{(1)}$ 的输入。这推广到 $K$ 个任务如下：

$$  h_{i}^{(k)}=f\left(W_{i}^{(k)}h_{i-1}^{(k)}+\sum_{j<k}U_{i}^{(k:j)}h_{i-1}^{(j)}\right),   \tag*{(1)}$$

其中  $W_i^{(k)} \in \mathbb{R}^{n_i \times n_{i-1}}$ 是第  $k$ 列第  $i$ 层的权重矩阵， $U_i^{(k:j)} \in \mathbb{R}^{n_i \times n_j}$ 是从第  $j$ 列第  $i-1$ 层到第  $k$ 列第  $i$ 层的横向连接， $h_0$ 是网络输入。 $f$ 是逐元素非线性函数：对所有中间层我们使用  $f(x) = \max(0, x)$。一个  $K = 3$ 的渐进式网络如图 1 所示。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//60f2b042-f272-4147-a828-91c20df19c42/markdown_1/imgs/img_in_image_box_509_865_715_1064.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A40%3A26Z%2F-1%2F%2Fc0b5eb2f4632684845456c404704c5b3651e2b8ae272489a6ba8319f51229649" alt="Image" width="16%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图 1：一个三列渐进式网络的图示。左侧的前两列（虚线箭头）分别在任务 1 和任务 2 上训练。标有 a 的灰色框表示适配器层（adapter layers，见正文）。为最终任务添加第三列，它可以访问所有先前学习到的特征。</div> </div>


这些建模决策源于我们的以下愿望：(1) 在训练结束时求解  $K$ 个独立任务；(2) 在可能时通过迁移加速学习；(3) 避免灾难性遗忘。

在标准的预训练—微调范式中，通常隐含地假设任务之间存在“重叠”。在这种情形下微调是高效的，因为参数只需稍微调整以适应目标域，并且往往只需重新训练顶层 [23]。相比之下，我们对任务之间的关系不做任何假设，实践中它们可能是正交的、甚至是相互对抗的。虽然微调阶段有可能遗忘这些特征，但这可能很困难。渐进式网络通过为每个新任务分配一个新列（其权重随机初始化）来回避这一问题。与预训练中与任务相关的初始化相比，渐进式网络中的列可以通过横向连接自由地复用、修改或忽略先前学习到的特征。由于横向连接  $U_i^{(k;j)}$ 仅从第  $k$ 列指向第  $j < k$ 列，因此在正向传播中先前的列不会受到新学习特征的影响。又由于在训练  $\Theta^{(k)}$ 时参数  $\{\Theta^{(j)}; j < k\}$ 保持冻结（即对优化器而言是常量），任务之间没有干扰，因此也没有灾难性遗忘。

应用于强化学习（Application to Reinforcement Learning）。虽然渐进式网络具有广泛的适用性，但本文聚焦于其在深度强化学习中的应用。在这种情况下，每一列被训练来求解特定的马尔可夫决策过程（Markov Decision Process, MDP）：第 $k$ 列因此定义了一个策略 $\pi^{(k)}(a\mid s)$，它以环境给出的状态 $s$ 作为输入，并生成动作上的概率分布 $\pi^{(k)}(a\mid s):=h_{L}^{(k)}(s)$。在每个时间步，从该分布中采样一个动作并在环境中执行，产生后续状态。该策略隐含地定义了状态和动作上的平稳分布（stationary distribution）$\rho_{\pi^{(k)}}(s,a)$。

适配器（Adapters）。在实践中，我们用非线性横向连接（我们称之为适配器）来增强式 2 的渐进式网络层。它们既用于改善初始条件，又用于执行降维。定义前向特征向量  $h_{i-1}^{(<k)} = [h_{i-1}^{(1)} \cdots h_{i-1}^{(j)} \cdots h_{i-1}^{(k-1)}]$，其维度为  $n_{i-1}^{(<k)}$，在全连接层的情况下，我们用单隐藏层 MLP 替换线性横向连接。在将横向激活输入 MLP 之前，我们将其乘以一个学习到的标量，该标量由随机小值初始化。其作用是调整不同输入的不同尺度。非线性适配器的隐藏层是一个到  $n_i$ 维子空间的投影。随着索引  $k$ 增大，这确保了来自横向连接的参数数量与  $|\Theta^{(1)}|$ 处于同一数量级。省略偏置项，我们得到：

$$  h_{i}^{(k)}=\sigma\left(W_{i}^{(k)}h_{i-1}^{(k)}+U_{i}^{(k:j)}\sigma(V_{i}^{(k:j)}\alpha_{i-1}^{(<k)}h_{i-1}^{(<k)})\right),   \tag*{(2)}$$

其中  $V_i^{(k:j)} \in \mathbb{R}^{n_{i-1} \times n_{i-1}^{(<k)}}$ 是投影矩阵。对于卷积层，降维通过  $1 \times 1$ 卷积 [10] 实现。

局限性（Limitations）。渐进式网络是迈向完整持续学习智能体的一块垫脚石：它们包含按顺序学习多个任务所必需的要素，同时能够迁移且对灾难性遗忘免疫。该方法的一个缺点是参数数量随任务数量增长。附录 2 的分析表明，只有一小部分新容量被实际利用，且这一趋势随着列数增多而加剧。这表明增长问题是可以解决的，例如通过增加更少的层或更少的容量、通过剪枝（pruning）[9]，或在学习过程中进行在线压缩（online compression）[17]。此外，虽然渐进式网络在测试时保留了求解所有 K 个任务的能力，但选择使用哪一列进行推理需要知道任务标签。这些问题留作未来工作。

## 3 迁移分析

与微调不同，渐进式网络不会破坏在先前任务上学习到的特征。这使我们能够详细研究哪些特征以及在哪个深度上实际发生了迁移。我们探索了两种相关方法：一种基于扰动分析的直观但缓慢的方法，以及一种从费舍尔信息 [2] 导出的更快的分析方法。

平均扰动敏感度（Average Perturbation Sensitivity, APS）。为了评估源列对目标任务的贡献程度，我们可以在架构中的孤立点（例如单个列的某一层）注入高斯噪声，并测量该扰动对性能的影响。性能的显著下降表明最终预测严重依赖于该特征图或层。我们发现该方法得到的结果与下面给出的更快的基于费舍尔的方法相似。因此我们将扰动分析的细节与结果放到附录中。

平均费舍尔敏感度（Average Fisher Sensitivity, AFS）。我们可以使用费舍尔信息矩阵 [2] 获得扰动敏感度的一个局部近似。费舍尔矩阵通常针对模型参数计算，而我们计算网络策略  $\pi$ 关于每一层归一化  $activations^2$ $\hat{h}_i^{(k)}$ 的修正对角费舍尔  $\hat{F}$。对于卷积层，我们定义  $\hat{F}$ 隐式地对像素位置求和。 $\hat{F}$ 可以解释为策略对表示中微小变化的敏感度。我们定义对角矩阵  $\hat{F}$，其元素为  $\hat{F}(m,m)$，并定义第  $k$ 列第  $i$ 层中特征  $m$ 的导出平均费舍尔敏感度（AFS）为：

 $$ \begin{aligned}\hat{F}^{(k)}_{i}&=\mathbb{E}_{\rho(s,a)}\left[\frac{\partial\log\pi}{\partial\hat{h}^{(k)}_{i}}\frac{\partial\log\pi}{\partial\hat{h}^{(k)}_{i}}^{T}\right]&\mathbf{AFS}(i,k,m)&=\frac{\hat{F}^{(k)}_{i}(m,m)}{\sum_{k}\hat{F}^{(k)}_{i}(m,m)}\end{aligned} $$ 

其中期望是关于由在目标任务上训练的渐进式网络所诱导的联合状态—动作分布  $\rho(s,a)$ 取的。在实践中，考虑每层的 AFS 分数 AFS $(i,k)=\sum_{m}\mathrm{AFS}(i,k,m)$ 通常很有用，即对第 i 层的所有特征求和。AFS 与 APS 由此估计网络在计算其输出时对某一层中每个特征或每一列的依赖程度。

## 4 相关工作

迁移与多任务强化学习存在许多不同的范式，因为它们长期以来被视为人工智能研究中的关键挑战 [15, 19, 20]。许多迁移学习方法依赖线性及其他简单模型（例如 [18]），这是其适用性的一个限制因素。近来，人们提出了用于深度强化学习的多任务或迁移学习新方法：[22, 17, 14]。在本工作中，我们提出一种用于深度强化学习的架构，它在顺序任务情形下能够实现无遗忘学习，同时支持从先前已学习任务中进行个体特征迁移。

预训练与微调由 [7] 提出，并被应用于 [4, 11] 中的迁移学习，通常用于无监督到监督或监督到监督的情形。actor-mimic 方法 [14] 将这些原则应用于强化学习，通过在新 Atari 游戏上微调一个 DQN 多任务网络，并表明一些游戏响应为更快的学习，而另一些则不然。渐进式网络与微调方向有本质区别，因为其容量是随着新任务被学习而增加的。

渐进式网络与神经网络文献中提出的增量式（incremental）与构造式（constructive）架构相关。级联相关（cascade-correlation）架构旨在消除遗忘的同时逐步添加和细化特征提取器 [6]。诸如 [24] 的自编码器使用增量特征增强来跟踪概念漂移（concept drift），并且 [16] 等深度架构被设计为专门支持特征迁移。最近，在 [1] 中，各列分别在单个噪声类型上训练，然后线性组合，而 [5] 使用列进行图像分类。[21] 的块模块化（block-modular）架构与我们的方法有许多相似之处，但聚焦于视觉辨别任务。相比之下，渐进式网络方法使用横向连接来访问先前学习到的特征以实现深度组合性。它可用于任何顺序学习场景，但在强化学习中尤其有价值。

## 5 实验

我们在三个不同的强化学习领域上评估渐进式网络。首先，我们考虑 Pong 的合成变体，它们被修改为具有视觉或控制层面的相似性。其次，我们对随机序列的 Atari 游戏进行广泛的实验，并进行特征层面的迁移分析。最后，我们在一组 3D 迷宫游戏上展示性能。图 2 展示了选自各任务的示例。

### 5.1 实验设置

我们依赖 [13] 中引入的异步优势演员—评论家（Async Advantage Actor-Critic, A3C）框架。与 DQN [12] 相比，该模型同时学习一个策略和一个用于预测期望未来回报的价值函数。A3C 在 CPU 上使用多线程训练，并已被证明在 GPU 上比 DQN 收敛更快。这使其更适合本工作所需的大量顺序实验。<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e111d139-083e-4d87-8479-7e4e9f1d02f0/markdown_0/imgs/img_in_image_box_253_164_428_313.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A40%3A24Z%2F-1%2F%2F756a7a27539f989094cb5b76952528e1ee4bf00f8fd1f59d95eb0d5ca5c75369" alt="Image" width="14%" /></div>


<div style="text-align: center;"><div style="text-align: center;">(a) Pong 变体</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e111d139-083e-4d87-8479-7e4e9f1d02f0/markdown_0/imgs/img_in_image_box_455_162_579_314.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A40%3A25Z%2F-1%2F%2F1936630c76c041d538bdbd561d0aa4e356859a0a56d0a53ec3968733a85f28d0" alt="Image" width="10%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e111d139-083e-4d87-8479-7e4e9f1d02f0/markdown_0/imgs/img_in_image_box_598_162_721_313.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A40%3A25Z%2F-1%2F%2Ff08da7c739162f33b0b23b030155ff145095532474c8abba06f3072fb9354bf3" alt="Image" width="10%" /></div>


<div style="text-align: center;"><div style="text-align: center;">(b) 迷宫游戏</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e111d139-083e-4d87-8479-7e4e9f1d02f0/markdown_0/imgs/img_in_image_box_748_161_968_316.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A40%3A25Z%2F-1%2F%2F9011cb7d3da69c34c2a3b6d0e2a7627057d367c7fd187adf1ae564d4fcdc4e18" alt="Image" width="17%" /></div>


<div style="text-align: center;"><div style="text-align: center;">(c) Atari 游戏</div> </div>


<div style="text-align: center;"><div style="text-align: center;">图 2：来自不同任务领域的样本：(a) Pong 变体包括翻转、加噪、缩放和重新着色变换；(b) Labyrinth 是一组 3D 迷宫游戏，具有多样的关卡地图以及多样的正、负奖励物品；(c) Atari 游戏为迁移提供了一个更具挑战性的场景。</div> </div>


我们通过平均 25 项任务中排名前 3 的结果来报告性能，每项任务具有不同的随机种子和超参数随机采样。性能通过测量学习曲线下的面积（训练期间每回合的平均得分）来评估，而非最终得分。迁移分数则定义为某架构相对于仅在目标任务上训练的单一列基线（基线 1）的相对性能。我们给出选定源—目标游戏的迁移分数曲线，并用迁移矩阵总结所有这样的配对。我们考虑的模型与基线如图 3 所示。实验设置的细节见附录第 3 节。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e111d139-083e-4d87-8479-7e4e9f1d02f0/markdown_0/imgs/img_in_image_box_281_652_324_769.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A40%3A25Z%2F-1%2F%2F499ddbd25815fb53a82294f1a3b353b5cd7cbeb80d872b1d6b28c456806e116a" alt="Image" width="3%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e111d139-083e-4d87-8479-7e4e9f1d02f0/markdown_0/imgs/img_in_image_box_372_652_417_765.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A40%3A25Z%2F-1%2F%2F8d3757610006cc175fb55a3ba27b6f1e400cd8134a5803f41f25896c10968cea" alt="Image" width="3%" /></div>


<div style="text-align: center;"><div style="text-align: center;">1) 基线 1 (2) 基线 2 (3) 基线 3</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e111d139-083e-4d87-8479-7e4e9f1d02f0/markdown_0/imgs/img_in_image_box_462_652_510_764.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A40%3A25Z%2F-1%2F%2Fb849d835baab6dee6e773d544449e53b40bc279c84881caae637379c0070940c" alt="Image" width="3%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e111d139-083e-4d87-8479-7e4e9f1d02f0/markdown_0/imgs/img_in_image_box_546_651_615_755.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A40%3A25Z%2F-1%2F%2F9ad9afa0afbd963f40ff17f1d90651f4b4325d35635585cea6a9e34b92b71a0a" alt="Image" width="5%" /></div>


<div style="text-align: center;"><div style="text-align: center;">输入</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e111d139-083e-4d87-8479-7e4e9f1d02f0/markdown_0/imgs/img_in_image_box_657_651_728_767.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A40%3A25Z%2F-1%2F%2Fa194860eca1f2082a5fd9a2e09ad5db2f143a0278a3fca5ed9bdf0caf4ff99e2" alt="Image" width="5%" /></div>


<div style="text-align: center;"><div style="text-align: center;">(4) 基线 4 (5)</div> </div>


<div style="text-align: center;"><div style="text-align: center;">(5) 两列渐进式网络</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e111d139-083e-4d87-8479-7e4e9f1d02f0/markdown_0/imgs/img_in_image_box_761_651_862_770.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A40%3A25Z%2F-1%2F%2F0610b43beb4d2e3047fc693fcab3ae63524479184a8bf916defc158dc0f700cd" alt="Image" width="8%" /></div>


<div style="text-align: center;"><div style="text-align: center;">(6) 三列渐进式网络</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e111d139-083e-4d87-8479-7e4e9f1d02f0/markdown_0/imgs/img_in_image_box_877_654_946_801.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A40%3A25Z%2F-1%2F%2Fdddc07ce18fa8af6b464abe002d2bc014cd0584039d656558c97dc3edd3c215f" alt="Image" width="5%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图 3：不同基线与架构的图示。基线 1 是在目标任务上训练的单一列；基线 2 是单一列，先在源任务上预训练，再在目标任务上微调（仅输出层）；基线 3 与基线 2 相同，但整个模型都被微调；基线 4 是一个两列渐进式架构，先前的一列或多列随机初始化并冻结。</div> </div>


### 5.2 Pong Soup

第一个评估领域是一组 Atari 游戏 Pong 的合成变体（“Pong Soup”），其中视觉效果和玩法都被改动，从而提供了一个我们可以确信任务间存在可迁移要素的场景。这些变体包括 Noisy（向输入添加冻结的高斯噪声）；Black（黑色背景）；White（白色背景）；Zoom（输入缩放 75% 并平移）；V-flip、H-flip 和 VH-flip（输入被水平地和/或垂直地翻转）。示例帧如图 2 所示。在 Pong 变体上训练两列（包括所有相关基线）的结果如图 4 所示。迁移分数在所有目标任务上的汇总见表 1。

<div style="text-align: center;"><div style="text-align: center;">(a)</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e111d139-083e-4d87-8479-7e4e9f1d02f0/markdown_0/imgs/img_in_image_box_225_1189_462_1367.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A40%3A26Z%2F-1%2F%2Fcb42c7926420a4734a30354155c6ac3e74919091e3e497462d27d414d18ac4d7" alt="Image" width="19%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e111d139-083e-4d87-8479-7e4e9f1d02f0/markdown_0/imgs/img_in_chart_box_561_1194_810_1366.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A40%3A26Z%2F-1%2F%2Fda974477f948904fd1f71f0749bbd2eb9789ac932494668d39da93f5fda8666c" alt="Image" width="20%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e111d139-083e-4d87-8479-7e4e9f1d02f0/markdown_0/imgs/img_in_chart_box_821_1198_1008_1366.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A40%3A26Z%2F-1%2F%2F3f854d14afc21ac5f48f026bf4612524457948b7f7be7115ad7d9343c7a617b6" alt="Image" width="15%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图 4：(a) 迁移矩阵。颜色表示迁移分数（截断于 2）。对于渐进式网络，第一列在 Pong、Noisy 或 H-flip 上训练（表格行）；第二列在其他每个 Pong 变体上训练（表格列）。(b) 示例学习曲线。</div> </div>
从这些结果中我们可以得到几点观察。基线 2（单一列，仅微调输出层；见图 3）在大多数实验中未能学习目标任务，因而产生负迁移。该方法在监督学习场景中相当标准，其中 ImageNet 训练网络的特征通常被重新用于新领域。如预期，我们观察到基线 3（单一列，完全微调）——一种成熟的迁移范式——具有很高的正迁移。然而渐进式网络在中位数与均值分数上均优于该基线，且在后者的差异更为明显。由于均值对离群值更敏感，这表明渐进式网络在迁移可能时（即源域与目标域兼容时）能够更好地利用迁移。图 4(b) 支持这一假设，其中渐进式网络在特定游戏对上被证明显著优于基线。渐进式网络与基线 4 相比也表现良好，证实渐进式网络确实利用了先前各列中学习到的特征。

#### 详细分析

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e111d139-083e-4d87-8479-7e4e9f1d02f0/markdown_1/imgs/img_in_image_box_228_453_645_648.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A40%3A27Z%2F-1%2F%2Fba75acda021d9c71f3a53cc3b58bf9c019f54d3f5e63080f2d7d984ae6c64c06" alt="Image" width="34%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e111d139-083e-4d87-8479-7e4e9f1d02f0/markdown_1/imgs/img_in_chart_box_664_456_825_641.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A40%3A27Z%2F-1%2F%2F5fb2bd130246a8c1bededa02e0f141ea1a67e0ee15861c11a3aaa22e5ebec12d" alt="Image" width="13%" /></div>


<div style="text-align: center;"><div style="text-align: center;">(c)</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e111d139-083e-4d87-8479-7e4e9f1d02f0/markdown_1/imgs/img_in_image_box_843_460_993_636.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A40%3A28Z%2F-1%2F%2F8f38f6c5b020a8ae25759ae9f2a651539748d0b7aa61fd6052159795422b1d10" alt="Image" width="12%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图 5：(a) Pong 变体上两列网络的迁移分析。网络输出对每层内各列的相对敏感度（即 AFS）由阴影深浅表示。(b) 单列 Pong 网络 conv. 1 的 8 个特征图的 AFS 值。网络实际上只使用了一个特征图；两列版本也使用同一特征图。下方：空间滤波器分量（红 = 正，蓝 = 负）。(c) 四个游戏的示例状态下 (b) 中滤波器的激活图。</div> </div>


我们使用第 3 节导出的度量来分析 Pong 变体之间迁移了哪些特征。我们看到，当从 Pong 切换到 H-Flip 时，网络复用了低层和中层视觉的相同组成部分（两个卷积层的输出；图 5a）。然而，全连接层必须大部分重新学习，因为该任务中策略相关的特征（球拍和球的相对位置/速度）现在处于新的位置。另一方面，当从 Pong 切换到 Zoom 时，低层视觉为新任务而复用，但学习到新的中层视觉特征。有趣的是，似乎只有一个低层特征被复用：（见图 5b）：这是一个具有相当大时间 DC 分量的时空滤波器。这似乎足以在原始、翻转和缩放的 Pong 中检测球的运动和球拍位置。

最后，当从 Pong 切换到 Noisy 时，一些新的低层视觉被重新学习。这很可能是因为在干净任务上学习到的第一层滤波器对添加的噪声不够容忍。相比之下，当从 Noisy 切换到 Pong 时（图 5a，最右列）这一问题并不存在，此时所有视觉都迁移到新任务。

### 5.3 Atari 游戏

接下来我们研究随机选择的 Atari 游戏 [3] 之间的特征迁移。这是一个有趣的问题，因为 Atari 游戏的视觉彼此差异很大，控制方式和所需策略也各不相同。尽管像 Pong 和 Breakout 这样的游戏在概念上相似（都是用一个球拍击球），但 Pong 是垂直对齐的，而 Breakout 是水平的：这可能是一个难以克服的特征层面差异。其他 Atari 游戏对则即使从概念层面看也没有可辨别的重叠。

为此，我们首先在三个源游戏（Pong、River Raid 和 Seaquest）$^{3}$ 上训练单一列，并评估学习到的特征是否能迁移到一组不同的随机选择目标游戏（Alien、Asterix、Boxing、Centipede、Gopher、Hero、James Bond、Krull、Robotank、Road Runner、Star Gunner 和 Wizard of Wor）。我们评估具有 2、3 和 4 列的渐进式网络，<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e111d139-083e-4d87-8479-7e4e9f1d02f0/markdown_2/imgs/img_in_chart_box_231_144_587_398.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A40%3A29Z%2F-1%2F%2F0e893fb00de003e12c5357b74d743f7273ebd05bd075e4d12d43a30191259947" alt="Image" width="29%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e111d139-083e-4d87-8479-7e4e9f1d02f0/markdown_2/imgs/img_in_chart_box_612_142_991_412.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A40%3A29Z%2F-1%2F%2Fb9cfc063f6da177e725608ee09c8a83987db7b29dde101d093219f0428afa4db" alt="Image" width="30%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图 6：Atari 目标游戏的迁移分数与示例学习曲线，与图 4 类似。</div> </div>




<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'></td><td colspan="2">Pong Soup</td><td colspan="2">Atari</td><td colspan="2">Labyrinth</td></tr><tr><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'>均值 (%)</td><td style='text-align: center; word-wrap: break-word;'>中位数 (%)</td><td style='text-align: center; word-wrap: break-word;'>均值 (%)</td><td style='text-align: center; word-wrap: break-word;'>中位数 (%)</td><td style='text-align: center; word-wrap: break-word;'>均值 (%)</td><td style='text-align: center; word-wrap: break-word;'>中位数 (%)</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>基线 1</td><td style='text-align: center; word-wrap: break-word;'>100</td><td style='text-align: center; word-wrap: break-word;'>100</td><td style='text-align: center; word-wrap: break-word;'>100</td><td style='text-align: center; word-wrap: break-word;'>100</td><td style='text-align: center; word-wrap: break-word;'>100</td><td style='text-align: center; word-wrap: break-word;'>100</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>基线 2</td><td style='text-align: center; word-wrap: break-word;'>35</td><td style='text-align: center; word-wrap: break-word;'>7</td><td style='text-align: center; word-wrap: break-word;'>41</td><td style='text-align: center; word-wrap: break-word;'>21</td><td style='text-align: center; word-wrap: break-word;'>88</td><td style='text-align: center; word-wrap: break-word;'>85</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>基线 3</td><td style='text-align: center; word-wrap: break-word;'>181</td><td style='text-align: center; word-wrap: break-word;'>160</td><td style='text-align: center; word-wrap: break-word;'>133</td><td style='text-align: center; word-wrap: break-word;'>110</td><td style='text-align: center; word-wrap: break-word;'>235</td><td style='text-align: center; word-wrap: break-word;'>112</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>基线 4</td><td style='text-align: center; word-wrap: break-word;'>134</td><td style='text-align: center; word-wrap: break-word;'>131</td><td style='text-align: center; word-wrap: break-word;'>96</td><td style='text-align: center; word-wrap: break-word;'>95</td><td style='text-align: center; word-wrap: break-word;'>185</td><td style='text-align: center; word-wrap: break-word;'>108</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>渐进式 2 列</td><td style='text-align: center; word-wrap: break-word;'>209</td><td style='text-align: center; word-wrap: break-word;'>169</td><td style='text-align: center; word-wrap: break-word;'>132</td><td style='text-align: center; word-wrap: break-word;'>112</td><td style='text-align: center; word-wrap: break-word;'>491</td><td style='text-align: center; word-wrap: break-word;'>115</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>渐进式 3 列</td><td style='text-align: center; word-wrap: break-word;'>222</td><td style='text-align: center; word-wrap: break-word;'>183</td><td style='text-align: center; word-wrap: break-word;'>140</td><td style='text-align: center; word-wrap: break-word;'>111</td><td style='text-align: center; word-wrap: break-word;'>—</td><td style='text-align: center; word-wrap: break-word;'>—</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>渐进式 4 列</td><td style='text-align: center; word-wrap: break-word;'>—</td><td style='text-align: center; word-wrap: break-word;'>—</td><td style='text-align: center; word-wrap: break-word;'>141</td><td style='text-align: center; word-wrap: break-word;'>116</td><td style='text-align: center; word-wrap: break-word;'>—</td><td style='text-align: center; word-wrap: break-word;'>—</td></tr></table>

<div style="text-align: center;"><div style="text-align: center;">表 1：三个领域中的迁移百分比。基线定义见图 3。</div> </div>


并与图 3 的基线进行比较。迁移矩阵与选定的迁移曲线如图 6 所示，结果汇总于表 1。

在所有游戏中，我们从图 6 观察到，渐进式网络在 12 个目标任务中有 8 个产生正迁移，只有两个负迁移情形。这与基线 3 相比更为有利，后者仅在 12 个游戏中的 5 个产生正迁移。这一趋势反映在表 1 中，其中渐进式网络在使用额外列时令人信服地优于基线 3。这尤其令人期待，因为我们在附录中表明，渐进式网络每增加一列所使用的新容量都在递减，这为通过在线压缩或剪枝来缓解模型规模增长指明了一条清晰的道路。

现在考虑特定的序列 Seaquest→Gopher，这是两个不相似游戏的一个例子。在这里，预训练/微调范式（基线 3）表现出负迁移，而渐进式网络则不然（见图 6b，底部），也许是因为它们更能忽略无关特征。对于序列 Seaquest+River Raid++Pong→Boxing，在渐进式网络中使用额外列可以显著增加迁移（见图 6b，顶部）。

##### 详细分析

图 6 表明渐进式网络既可能产生正迁移也可能产生负迁移。为区分这些情况，我们考虑三列情况的平均费舍尔敏感度（例如见图 7a）。在这些例子与其他例子中浮现出一个清晰的模式：最严重的负迁移与对先前各列卷积层的完全依赖相吻合，且新列没有学习到任何新的视觉特征。相比之下，最正面的迁移发生在前两列的特征被新特征增强之时。所有三列网络的统计数据（图 7b）表明，Atari 中的正迁移发生在对源任务特征的严重依赖与对目标任务所有新特征的严重依赖之间的一个“最佳点”。

乍看之下，这一结果似乎不合直觉：如果渐进式网络从源任务中找到了有价值的特征集，我们难道不应预期高度的迁移吗？我们提出两个假设。首先，这可能只是反映了优化上的困难，即源特征提供了快速收敛到一个差的局部极小值。这是迁移学习中一个已知的挑战 [20]：所学源任务赋予一种归纳偏置（inductive bias），在不同情况下既可能有所帮助也可能有所妨碍。其次，这可能反映了<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e111d139-083e-4d87-8479-7e4e9f1d02f0/markdown_3/imgs/img_in_chart_box_233_140_784_343.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A40%3A30Z%2F-1%2F%2F5ca577584d9ec8d49089fc0476521bad0222714d9fd9410f630a6186d407f7b6" alt="Image" width="45%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e111d139-083e-4d87-8479-7e4e9f1d02f0/markdown_3/imgs/img_in_chart_box_829_152_990_337.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A40%3A30Z%2F-1%2F%2F16e76c6c867bca5d6703cbee01cfe09bcb29b7247debe2fe67ab0d7560180a80" alt="Image" width="13%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图 7：(a) 在 12 个目标 Atari 游戏中迁移分数最低（左）与最高（右）的三列网络的 AFS 分数。(b) 跨越 72 个三列网络的迁移统计，作为新列三个卷积层平均 AFS 的函数（即学习到多少新视觉）。</div> </div>


探索的问题，即迁移得来的表示对一个功能可行但次优的策略而言“足够好”。

### 5.4 Labyrinth

渐进式网络的最后一个实验场景是 Labyrinth，一个 3D 迷宫环境，其中输入是渲染图像，仅提供部分可观测性，智能体输出离散动作，包括向上、下、左或右看，以及向前、后、左或右移动。任务和关卡地图都多样，涉及通过“吃”好物品（苹果、草莓）获得正分，以及通过吃坏物品（蘑菇、柠檬）获得负分。细节可在附录中找到。虽然不同任务之间存在概念和视觉上的重叠，但任务呈现了一组具有挑战性的多样游戏元素（图 2）。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e111d139-083e-4d87-8479-7e4e9f1d02f0/markdown_3/imgs/img_in_chart_box_214_738_390_913.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A40%3A30Z%2F-1%2F%2F8e3033a5bcf31114783993141d017e9d2449349d005cce35793980b1d29334ce" alt="Image" width="14%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e111d139-083e-4d87-8479-7e4e9f1d02f0/markdown_3/imgs/img_in_chart_box_495_735_1014_916.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-11-16T05%3A40%3A30Z%2F-1%2F%2F06771bb7cd564a450071811804f2d498a88b08402c7918b9c90f65f0e06032d0" alt="Image" width="42%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图 8：Labyrinth 任务的迁移分数与示例学习曲线。颜色表示迁移（截断于 2）。学习曲线展示了两列渐进式性能对比基线 1 和 3 的两个例子。</div> </div>


与其他领域一样，渐进式方法比任何基线都产生了更多的正迁移（见图 8a 和表 1）。我们在 Seek Track 关卡上观察到较少的迁移，这些关卡在迷宫中遍布稠密的奖励物品，且容易学习。注意，即使对这些简单情形，基线 2 也表现出负迁移，因为它无法学习新的低层视觉特征，而这些特征很重要，因为奖励物品随任务而变。图 8b 中的学习曲线例示了该领域中的典型结果：在更简单的游戏（如 Track 1 和 2）上，所有智能体的学习都快速而稳定。在更困难的游戏（具有更复杂的游戏结构）上，基线表现挣扎，而渐进式网络具有优势。

## 6 结论

持续学习，即积累知识并将其迁移到新领域的能力，是智能生物的核心特征。渐进式神经网络是迈向持续学习的一块垫脚石，本工作通过在三个强化学习领域（包括 Atari，其中包含正交甚至相互对抗的任务）的实验与分析展示了它们的潜力。我们相信，我们是首个在持续学习框架内的深度强化学习智能体中展示正迁移的工作。此外，我们表明渐进式方法能够有效地为兼容的源域与任务域利用迁移；该方法对在不兼容任务中学到的有害特征具有鲁棒性；并且正迁移随列数增加而增加，从而印证了渐进式架构的构造性（而非破坏性）本质。
