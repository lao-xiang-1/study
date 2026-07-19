---
sr-due: 2026-07-18
sr-interval: 8
sr-ease: 250
---
#paper
# Introduction to Predictive Coding Networks for Machine Learning
[原论文地址](https://arxiv.org/pdf/2506.06332)

代码文件夹（本地）：
[pcn_cifar10_notebook](file:///D:\code\pcn-intro/)

May 29, 2025

## References

## 1 Introduction

Already in 1867, Helmholtz proposed that perception is an unconscious inferential process where the brain predicts how planned actions will affect sensory inputs [12]. In the mid-20th century, Barlow's efficient coding hypothesis posited that the brain economizes neural representation by removing predictable redundancies in sensory signals [3], foreshadowing a later focus on unexpected or surprising sensory events. Gregory further argued that perception is a constructive, hypothesis-driven endeavor, with visual illusions highlighting how top-down expectations shape perception [10]. By the 1990s, theoretical models explicitly invoked hierarchical prediction: Mumford proposed a cortical architecture in which higher-level areas send predictions to lower-level areas, with only residual errors fed forward [21]. Such concepts set the stage for the predictive coding model of visual cortex by Rao and Ballard [23], which formalized perception as a hierarchical interplay of top-down predictions and bottom-up error signals.
**早在1867年，亥姆霍兹（Helmholtz）就提出，感知是一个无意识推理过程，大脑在其中预测计划好的动作将如何影响感觉输入[12]。20世纪中期，巴洛（Barlow）的高效编码假说认为，大脑通过消除感觉信号中可预测的冗余来节约神经表征[3]，这一观点预示了后来对意外或令人惊讶的感觉事件的关注。格雷戈里（Gregory）进一步论证，感知是一种建构性的、假设驱动的心智活动，而视觉错觉则凸显了自上而下的期望如何塑造感知[10]。到了20世纪90年代，理论模型明确引入了层级预测：蒙福德（Mumford）提出了一种皮层结构，其中高层级区域向低层级区域发送预测，只有残差误差向前传递[21]。这些概念为拉奥和巴拉德（Rao and Ballard）[23]的视觉皮层预测编码模型奠定了基础，该模型将感知形式化为自上而下的预测与自下而上的误差信号之间的层级交互作用。**

Originally developed to explain extra-classical receptive field effects in visual cortex [23], predictive coding has since been generalized into a unified framework for cortical processing through the free-energy principle [8, 9]. This principle casts perception and action as inference problems, where organisms minimize a variational bound on surprise or prediction error.
预测编码最初用于解释视觉皮层的**非经典感受野效应**，后被推广为解释大脑皮层信息处理的统一理论框架——即通过**自由能原理（free-energy principle）**将"感知"和"行动"都归结为**推理问题**：生物体通过最小化"惊讶度"或"预测误差"来理解世界并作出反应。

Neurophysiological plausibility of predictive coding has been further explored in works such as [28], which unify predictive coding with biased competition models of attention, and [4], which describe potential microcircuit implementations in cortical hierarchies. Keller and Mrsic-Flogel [13] offer evidence that predictive processing is a canonical computation across the cortex, with supportive anatomical and functional data reviewed by Shipp [26]. Further empirical neuroscience evidence is presented in [30, 5].

From a computational modeling perspective, predictive coding networks provide an alternative to traditional feedforward models trained via backpropagation. Notably, Whittington and Bogacz [31] demonstrated that predictive coding networks can approximate backpropagation in multilayer neural networks using only local updates; see also Millidge et al. [19]. Their implementation in spiking neural networks was reviewed in [22].

Predictive coding has also inspired innovations in unsupervised and self-supervised learning. For example, Schmidhuber [25] proposed the predictability minimization principle echoing the redundancy reduction goal of predictive coding. Lotter et al. [15] introduced deep predictive coding networks for video frame prediction, offering state-of-the-art performance using unsupervised objectives. Moreover, [11] developed a “world model”—a network that learns a latent predictive representation of an agent’s environment—showing how hierarchical prediction and the minimization of surprise can facilitate efficient learning and planning.

A broader theoretical and empirical review of predictive coding in both neuroscience and artificial intelligence is provided in [18], which also outlines future research directions.

The references cited here are just a bite-sized sample of the enormous literature on the subject. The interested reader is advised to look into the bibliographies in those for more coverage, and to search the arXiv for the latest developments. There are also many excellent blog posts on the subject, such as [17, 1].

### 2. 神经生理学验证
该理论在大脑中的生物学可行性得到了多方面研究支持：
- 与**注意力偏置竞争模型**相融合；
- 在皮层层级结构中找到了可能的**微回路实现**；
- 有证据表明预测处理是大脑皮层的一种**规范性计算（canonical computation）**，并获得了解剖学和功能数据的支持。

### 3. 计算建模意义：反向传播的生物可信替代
预测编码网络为传统深度学习提供了一种新的计算视角：
- 它可以用**纯局部更新**来近似多层网络中的**反向传播算法**，无需全局误差信号；
- 这一机制还可以映射到**脉冲神经网络（SNN）**中，为神经形态计算提供了理论基础。

### 4. 对机器学习的启发
预测编码思想推动了无监督/自监督学习的发展：
- **可预测性最小化**原则呼应了冗余减少目标；
- **深度预测编码网络**被用于视频帧预测等任务；
- 催生了"**世界模型（World Model）**"概念——让智能体学习环境的潜在预测表征，从而实现更高效的学习与规划。


## 2 Network Architecture

### Model：参数说明
A PCN consists of $L \geq 1$ layers of latent variables $\mathbf{x}^{(l)} \in \mathbb{R}^{d_l}$, $1 \leq l \leq L$, and an input layer of variables $\mathbf{x}^{(0)} \in \mathbb{R}^{d_0}$. Each layer attempts to predict the state of the layer below.
- PCN 由 $L$ 层（至少1层）**隐变量（latent variables）**组成。
- 第 $l$ 层的变量记为 $\mathbf{x}^{(l)}$，是一个 $d_l$ 维的实数向量。
- 除了 $L$ 层隐变量，还有一个**输入层** $\mathbf{x}^{(0)}$，维度为 $d_0$。

- **核心思想**：每一层都试图去"猜测"或**预测**它下面那一层的状态。
- 信息流是**自上而下（top-down）**的。

- 因此，对于每一对相邻层（$l$ 从 0 到 $L-1$），都需要定义相应的**自上而下连接元素**（权重、预测函数等）。

- Weights $\mathbf{W}^{(l)} \in \mathbb{R}^{d_l \times d_{l+1}}$ from layer $l + 1$ to layer $l$

• Preactivations（预激活）
$$ \mathbf{a}^{(l)}=\mathbf{W}^{(l)}\mathbf{x}^{(l+1)}\in\mathbb{R}^{d_{l}} $$
• Predictions
$$ \hat{\mathbf{x}}^{(l)}=f^{(l)}(\mathbf{a}^{(l)})\in\mathbb{R}^{d_{l}} $$ 

*where  $f^{(l)}$ is, often a nonlinear, scalar function applied elementwise（**激活函数**）*

• Prediction errors （隐变量 $-$ 上层的预测值）
$$ \boldsymbol{\varepsilon}^{(l)}=\mathbf{x}^{(l)}-\hat{\mathbf{x}}^{(l)}\in\mathbb{R}^{d_{l}} $$ 

The loss function subject to minimization（**待最小化的损失函数**） is the total square prediction error, or energy:
$$ \mathcal{L}=\frac{1}{2}\sum_{l=0}^{L-1}\left\|\varepsilon^{(l)}\right\|^{2}. $$ 

The network can be viewed as a directed acyclic graph; see Figure 1. The “hanging” root nodes without incoming edges are the input and latent variables; the leaves (end nodes) are the prediction errors. Observe that the generative hierarchy flows from top to bottom, towards the input; more on this will be discussed shortly.
该网络可被视为一个**有向无环图**；参见图 1。没有入边的"悬挂"根节点是**输入变量和隐变量**；叶节点（末端节点）则是**预测误差**。注意，生成式层级结构是**自上而下**流动的，指向输入层；关于这一点将在稍后进一步讨论。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//6999a80c-1f77-4042-951c-2c12ce7ebd2d/markdown_3/imgs/img_in_image_box_456_316_738_579.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-06-15T03%3A39%3A23Z%2F-1%2F%2Fe3df08118ef25b030ebbd2aedd4e699a137dcf7c621f531cc8ef299d5e2a7384" alt="Image" width="23%" /></div>

<div style="text-align: center;"><img src=assets/2506.06332v1.pdf_by_PaddleOCR-VL-1.6.png alt="Image" width="100%" /></div>

<div style="text-align: center;color: #2ecc71;"><div style="text-align: center;">Figure 2: A supervised extension of the PCN with three latent layers in Figure 1 is obtained by stacking a readout layer on top of the highest latent layer, together with a new root node for the target label.</div> </div>

> [!tip] Tip
> pcn的预测不是 从输入一步步传播到输出的
> 而是直接从最后一个隐变量输出：$\hat y = W^{out} * x^{(3)}$
> 其中最后一个隐变量 $x^{(3)}$ 在预测时 通过计算得出，而不是事先设定的

### Alternating minimization procedure (交替最小化步骤)
#### 1. 寻找一组隐变量使能量最小
The generative weights $\mathbf{W} = (\mathbf{W}^{(0)}, \ldots, \mathbf{W}^{(L-1)})$ define the overall shape of the predictive coding **energy landscape**（**能量景观**）. For **fixed** weights and input, the inference process seeks a configuration $\mathbf{x}^*$ of latent values $\mathbf{x} = (\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(L)})$ that minimizes the energy:
- **固定权重和输入**：暂时把权重 $\mathbf{W}$ 和输入 $\mathbf{x}^{(0)}$ 当作常量，不去改变它们。
- **推理过程（inference process）**：寻找一组最优的隐变量 $\mathbf{x}^* = (\mathbf{x}^{*(1)}, \ldots, \mathbf{x}^{*(L)})$，使得总预测误差最小。

$$ \mathbf{x}^{*}=\mathop{\operatorname{a r g}\operatorname*{m i n}}_{\mathbf{x}}\mathcal{L}(\mathbf{x};\mathbf{W},\mathbf{x}^{(0)})\;. $$

  - $\arg\min_{\mathbf{x}}$：找到使后面表达式最小的 $\mathbf{x}$ 值
  - $\mathcal{L}(\mathbf{x}; \mathbf{W}, \mathbf{x}^{(0)})$：以 $\mathbf{x}$ 为自变量，$\mathbf{W}$ 和 $\mathbf{x}^{(0)}$ 为参数的能量函数

#### 2. 固定隐变量，优化权重
Optimization then switches to **learning**, which seeks to perturb the energy landscape by adjusting the weights gently to $\mathbf{W}' = \mathbf{W} + \delta\mathbf{W}$ (e.g., taking one gradient step with a small learning rate) so as to further reduce the energy at the inferred configuration $\mathbf{x}^*$:
- 梯度下降法求权重，使能量函数（损失）变得更小
$$ \mathcal{L}(\mathbf{x}^{*};\mathbf{W}^{\prime},\mathbf{x}^{(0)})<\mathcal{L}(\mathbf{x}^{*};\mathbf{W},\mathbf{x}^{(0)})~. $$ 

In this way, inference performs descent within a fixed landscape, while learning deforms the landscape to better accommodate future inference. This separation of roles gives predictive coding networks their characteristic two-timescale dynamics: fast inference, slow learning.

In practice, the procedure repeats for a new input $\mathbf{x}^{(0)}$ and reinitialized latent configuration $\mathbf{x}$. Such matters are deferred to the last section where an actual application will be discussed together with the training details.

**Convergence.** Several works have established, formally and experimentally, that the alternating optimization procedure in predictive coding networks—consisting of inference steps over latent variables followed by learning updates of the generative weights—converges to a local minimum of the loss under sufficient assumptions [31, 27, 7, 16, 2, 24]. That said, what matters to practitioners is performance in task-specific settings rather than just guarantees of local convergence under idealized conditions. On that note, the application in the last section serves as one example where fast convergence and excellent generalization do occur.

**Generative hierarchy.** The network is built in stacked layers of latent variables. The idea is that each layer represents the data at a different level of abstraction, from raw sensory inputs up through progressively higher-order features. Predictions flow top-down (higher layers predicting lower-layer activity) and errors flow bottom-up. Similar processes are believed to exist in the brain's multi-level sensory hierarchies. Rather than just learning a discriminative mapping from inputs to labels, the PCN explicitly encodes a generative model of how lower-level activity could be produced from higher-level causes, so the network can attempt to reconstruct what the sensory data should look like given its latent beliefs.

**A word about hybrid predictive coding.** In the algorithms and implementation presented in this document, the initial values of the latent variables are random—both literally and figuratively. Recently, an interesting hybrid predictive coding model has been proposed, where the latents $\mathbf{x}^{(l)}$ are initialized to the values $\boldsymbol{\xi}^{(l)}$ predicted by another network [29]. The predictions of this second network flow in the bottom-up direction, opposite to the PCN hierarchy, via “amortized” functions $\mathbf{g}^{(l)} : \mathbb{R}^{d_{l-1}} \to \mathbb{R}^{d_l}$ in a feedforward fashion: $\boldsymbol{\xi}^{(l)} = \mathbf{g}^{(l)}(\boldsymbol{\xi}^{(l-1)})$, $1 \leq l \leq L$, starting from the input $\boldsymbol{\xi}^{(0)} = \mathbf{x}^{(0)}$. Such a construction reflects the adjustment of the network’s posterior beliefs upon receiving sensory input, before the refining inference process of the vanilla PCN begins. The rest of this note does not involve the hybrid model.

##### 1. 双时间尺度动态
推理与学习的分工形成了PCN的典型特征：
- **推理（Inference）**：在**固定的能量景观**中快速下降，寻找最优隐变量
- **学习（Learning）**：**缓慢地重塑能量景观**，使其更利于未来的推理
- 核心特征：**快速推理，缓慢学习**（fast inference, slow learning）

##### 2. 实际迭代过程
- 对每个新输入样本，隐变量都**重新初始化**
- 然后重复"推理→学习"的交替循环
- 具体实现细节（如训练流程）留到后面的应用章节

##### 3. 收敛性保证
- 已有研究从理论和实验上证明：交替优化过程在适当假设下**收敛到局部最小值**
- 但作者强调：实践者更关心的是**具体任务上的实际性能**，而非理想条件下的理论保证
- 文末的CIFAR-10应用将展示**快速收敛和良好泛化**的实例

##### 4. 生成式层级结构
- 每层隐变量代表数据的不同**抽象层次**（从原始感官输入到高级特征）
- **信息流**：预测自上而下，误差自下而上
- 与大脑的多层级感觉层级类似
- PCN不是简单的"输入→标签"判别映射，而是显式编码了**生成模型**——即高层原因如何产生低层活动，从而能够根据内部信念**重构感官数据**

##### 5. 混合预测编码（补充说明）
- **标准PCN**：隐变量初始值完全随机
- **混合模型**：用另一个**自底向上**的前馈网络来预测隐变量的初始值，作为推理的起点
- 这种初始化反映了接收感官输入后**后验信念的快速调整**，在此基础上再进行精细的推理迭代
- 本文后续内容**不涉及**该混合模型

## 3 Inference and Learning Rules

> [!tip] Tip
> PCN也是用梯度下降的方式，但是不从全局损失计算梯度，而只是利用每层计算出的误差

### 3.1 Latent state update rule (inference)

We aim to update each latent variable  $\mathbf{x}^{(l)}$ ( $1 \leq l \leq L$) via gradient descent on  $\mathcal{L}$.

**损失函数**：
$$\mathcal{L} = \frac{1}{2}\sum_{l=0}^{L-1} \|\varepsilon^{(l)}\|^2 = \frac{1}{2}\sum_{l=0}^{L-1}\sum_j (\varepsilon_j^{(l)})^2$$
- 其中 $\varepsilon_j^{(l)} = x_j^{(l)} - \hat{x}_j^{(l)}$。

#### Case $1 \leq l < L$
The variable $\mathbf{x}^{(l)}$ appears in two places in the loss:

| 影响路径 | 公式 | 说明 |
|---------|------|------|
| **直接影响本层误差** | $\varepsilon_i^{(l)} = x_i^{(l)} - \hat{x}_i^{(l)}$ | $x_i^{(l)}$ 是被预测的对象 |
| **间接影响下层误差** | $\varepsilon_j^{(l-1)} = x_j^{(l-1)} - \underbrace{f^{(l-1)}(\mathbf{W}^{(l-1)}\mathbf{x}^{(l)})}_{\hat{x}_j^{(l-1)}}$ | $x_i^{(l)}$ 通过权重参与了对下层的预测 |

分别对两部分求导：
$$
\begin{aligned}
\frac{\partial\mathcal{L}}{\partial x_{i}^{(l)}}
&= \sum_{j}\varepsilon_{j}^{(l)}\frac{\partial\varepsilon_{j}^{(l)}}{\partial x_{i}^{(l)}} + \sum_{j}\varepsilon_{j}^{(l-1)}\frac{\partial\varepsilon_{j}^{(l-1)}}{\partial x_{i}^{(l)}} \\
&= \varepsilon_{i}^{(l)} - \sum_{j}\varepsilon_{j}^{(l-1)}\frac{\partial\hat{x}_{j}^{(l-1)}}{\partial x_{i}^{(l)}}
\end{aligned}
$$
where
$$ \begin{aligned}
\frac{\partial\hat{x}_{j}^{(l-1)}}{\partial x_{i}^{(l)}}
&=\frac{\partial}{\partial x_{i}^{(l)}}(f^{(l-1)}(a_{j}^{(l-1)}))\\
&=f^{(l-1)^{\prime}}(a_{j}^{(l-1)})W_{j i}^{(l-1)}. 
\end{aligned}
$$
总体：
$$\frac{\partial\mathcal{L}}{\partial x_{i}^{(l)}} = \varepsilon_{i}^{(l)} - \sum_{j}\varepsilon_{j}^{(l-1)} \cdot f^{(l-1)^{\prime}}(a_{j}^{(l-1)})W_{j i}^{(l-1)}$$

Hence, the gradient with respect to  $\mathbf{x}^{(l)}$ （关于 $\mathbf{x}^{(l)}$ 的梯度为）is

$$ \begin{array}{r l r}{\nabla_{\mathbf{x}^{(l)}}\mathcal{L}=\boldsymbol{\varepsilon}^{(l)}-\mathbf{W}^{(l-1)\top}\left(f^{(l-1)^{\prime}}(\mathbf{a}^{(l-1)})\odot\boldsymbol{\varepsilon}^{(l-1)}\right)}&{{}}&{(1\leq l<L)}\end{array} $$ 
- $\odot$：逐元素相乘
- $\mathbf{x}^{(l)}$ and $\nabla_{\mathbf{x}^{(l)}}\mathcal{L}$ have the **same shape**.
where $\odot$ denotes the elementwise (Hadamard) product, and the convention used for stacking the entries is that the vectors $\mathbf{x}^{(l)}$ and $\nabla_{\mathbf{x}^{(l)}}\mathcal{L}$ have the same shape.

#### Case $l=L$ 
 Notice that  $\mathbf{x}^{(L)}$ only appears inside  $\varepsilon^{(L-1)}$ in the expression of  $\mathcal{L}$. Hence,
$$ \frac{\partial\mathcal{L}}{\partial x_{i}^{(L)}}=\sum_{j}\varepsilon_{j}^{(L-1)}\frac{\partial\varepsilon_{j}^{(L-1)}}{\partial x_{i}^{(L)}}=-\sum_{j}\varepsilon_{j}^{(L-1)}\frac{\partial\hat{x}_{j}^{(L-1)}}{\partial x_{i}^{(L)}}=-\sum_{j}\varepsilon_{j}^{(L-1)}f^{(L-1)^{\prime}}(a_{j}^{(L-1)})W_{j i}^{(L-1)}\;. $$ 
In vector form,
$$ \nabla_{\mathbf{x}^{(L)}}\mathcal{L}=-\mathbf{W}^{(L-1)\top}\left(f^{(L-1)^{\prime}}(\mathbf{a}^{(L-1)})\odot\boldsymbol{\varepsilon}^{(L-1)}\right) $$ 

which is similar to  $1 \leq l < L$ except for the missing first term: there is no prediction error for the top layer.

#### 总体
Introducing the convenient constant
为简化表达，设顶层误差为0：
$$ \boldsymbol{\varepsilon}^{(L)}=\mathbf{0} $$ 

the inference update rule for the gradient descent algorithm becomes compactly
两种情况即可合并为一种
 $$ \begin{array}{r l r}&{}&{\left|\mathbf{x}^{(l)}\leftarrow\mathbf{x}^{(l)}-\eta_{\mathrm{i n f e r}}\left(\pmb{\varepsilon}^{(l)}-\mathbf{W}^{(l-1)\top}\left(f^{(l-1)^{\prime}}(\mathbf{a}^{(l-1)})\odot\pmb{\varepsilon}^{(l-1)}\right)\right)\right|\qquad(1\leq l\leq L)}\end{array} $$

- where  $\eta_{infer} > 0$ is an inference rate of choice.（学习率）

During inference, all prediction errors and feedback terms are computed first using the current network state, and only then are the latent variables  $\mathbf{x}^{(l)}$ updated. This ensures that each update step is based on a consistent energy landscape and avoids using partially updated states within the same iteration. Conceptually, this corresponds to a synchronous update scheme where all neurons compute their next state based on the same network snapshot.
在推断过程中，首先利用当前网络状态计算所有预测误差和反馈项，然后才对隐变量 $\mathbf{x}^{(l)}$ 进行更新。这确保了每次更新步骤都基于一致的能量景观，并避免在同一次迭代中使用部分更新的状态。从概念上讲，这对应于一种同步更新方案，其中所有神经元都基于同一网络快照计算其下一状态。

### 3.2 Weight update rule (learning)
> 更新权重而不是隐变量

损失函数：
$$\mathcal{L} = \frac{1}{2}\sum_{l}\|\boldsymbol{\varepsilon}^{(l)}\|^2$$
Each weight matrix  $\mathbf{W}^{(l)}$ is responsible for predicting  $\mathbf{x}^{(l)}$ from  $\mathbf{x}^{(l+1)}$, and appears only in  $\boldsymbol{\varepsilon}^{(l)} = \mathbf{x}^{(l)} - f^{(l)}(\mathbf{W}^{(l)}\mathbf{x}^{(l+1)})$. To minimize the loss, we compute
$$ \frac{\partial\mathcal{L}}{\partial W_{ij}^{(l)}}=\sum_{k}\varepsilon_{k}^{(l)}\frac{\partial\varepsilon_{k}^{(l)}}{\partial W_{ij}^{(l)}}=-\sum_{k}\varepsilon_{k}^{(l)}\frac{\partial}{\partial W_{ij}^{(l)}}(f^{(l)}(a_{k}^{(l)})) $$ 
which yields
$$ \frac{\partial\mathcal{L}}{\partial W_{i j}^{(l)}}=-\sum_{k}\varepsilon_{k}^{(l)}f^{(l)^{\prime}}(a_{k}^{(l)})\delta_{i k}x_{j}^{(l+1)}=-\varepsilon_{i}^{(l)}f^{(l)^{\prime}}(a_{i}^{(l)})x_{j}^{(l+1)}. $$ 

As before, using the convention that the matrices  $\mathbf{W}^{(l)}$ and $\nabla_{\mathbf{W}^{(l)}} \mathcal{L}$ have the same shape, we arrive at the gradient
$$ \nabla_{\mathbf{W}^{(l)}}\mathcal{L}=-\left(f^{(l)^{\prime}}(\mathbf{a}^{(l)})\odot\boldsymbol{\varepsilon}^{(l)}\right)\mathbf{x}^{(l+1)\top}\;. $$ 
In particular, the learning update rule via gradient descent is
$$ \begin{array}{r l r l}&{\left|\mathbf{W}^{(l)}\leftarrow\mathbf{W}^{(l)}+\eta_{\mathrm{l e a r n}}\left(f^{(l)^{\prime}}(\mathbf{a}^{(l)})\odot\boldsymbol{\varepsilon}^{(l)}\right)\mathbf{x}^{(l+1)\top}\right|}&&{\quad(0\leq l<L)}\end{array} $$ 
with a learning rate of  $\eta_{learn} > 0$.

Notice that the quantities
$$ \mathbf{h}^{(l)}=f^{(l)^{\prime}}(\mathbf{a}^{(l)})\odot\boldsymbol{\varepsilon}^{(l)}\qquad(0\leq l<L) $$

are central, as they appear in the expressions of the gradients with respect to both the latents and the weights. They could be called gain-modulated errors; see the subsection below.

### 3.3 Locality of the updates（更新的局部性）

One of the original motivations behind predictive coding is its potential biological plausibility: that the brain could implement something akin to deep hierarchical learning using local computations. Locality typically refers to whether a computation depends only on information from a given layer and its immediate neighbors. This concept is important both for computational efficiency and biological plausibility.

The learning update for the weight matrix $\mathbf{W}^{(l)}$ is local in a strong sense: it depends only on the local activity $\mathbf{x}^{(l+1)}$ of layer $l+1$ (presynaptic) and the local prediction error $\boldsymbol{\varepsilon}^{(l)}$ at layer $l$ (postsynaptic), making it compatible with Hebbian-like plasticity mechanisms often summarized as “neurons that fire together, wire together.”

In contrast, while the inference update for a latent variable $\mathbf{x}^{(l)}$ depends only on adjacent layers, it requires access to the error signal $\boldsymbol{\varepsilon}^{(l-1)}$ broadcast from the lower layer, modulated by the top-down weights in $\mathbf{W}^{(l-1)}$. Thus, inference updates are **layer-local** but not neuron-local, since a neuron's update depends on a weighted sum of errors from other neurons in the layer below.
#### 3. 推断更新（隐变量更新）——局部性较弱
隐变量更新公式：
$$\mathbf{x}^{(l)} \leftarrow \mathbf{x}^{(l)} - \eta_{\text{infer}} \left( \boldsymbol{\varepsilon}^{(l)} - \mathbf{W}^{(l-1)\top}(f^{(l-1)'} \odot \boldsymbol{\varepsilon}^{(l-1)}) \right)$$
虽然它只涉及相邻两层，但存在一个问题：

- 它需要**下层 $l-1$ 的所有误差** $\boldsymbol{\varepsilon}^{(l-1)}$ 通过权重矩阵 $\mathbf{W}^{(l-1)}$ 加权求和
- 一个神经元的更新 = 下层**所有神经元误差的线性组合**

> **问题**：单个神经元需要"知道"下层其他神经元的状态，这不是严格意义上的神经元局部。

所以推断更新是**层局部**（layer-local）但不是**神经元局部**（neuron-local）。

Biologically, the learning updates $\delta W_{ij}^{(l)} = -\eta_{\text{learn}}\varepsilon_i^{(l)}f^{(l)^{\prime}}(a_i^{(l)})x_j^{(l+1)}$ are thought to be more plausible than inference updates. Here, $\varepsilon_i^{(l)}$ is the prediction error at the postsynaptic neuron $i$, $a_i^{(l)} = \sum_m W_{im}^{(l)}x_m^{(l+1)}$ is its preactivation, and $x_j^{(l+1)}$ is the activity of the presynaptic neuron $j$. The preactivation represents the total synaptic input to neuron $i$—essentially its membrane potential or driving current. In both biological and artificial neurons, this summation is performed naturally as part of neural activation. The neuron does not need to access other neurons’ states; it simply integrates the inputs it receives via its dendrites. Thus, $f^{(l)^{\prime}}(a_i^{(l)})$ can be interpreted as a local gain or nonlinearity applied to the neuron’s own internal state. This makes the full weight update neuron-local in the strong sense: it depends only on information accessible to the synapse between neurons $j \to i$, including presynaptic activity, postsynaptic error, and internal quantities of the postsynaptic neuron.

### 3.4 Motivations beyond biological plausibility

While predictive coding networks (PCNs) are often motivated by neuroscience and cortical modeling, their architectural design makes them attractive for future machine learning systems on emerging (e.g., neuromorphic) hardware.

#### Locality and parallelism. 
PCNs rely on local computations: each weight update depends only on the activity and prediction error of adjacent neurons. This makes them well suited to distributed and parallel computing, in contrast to global gradient backpropagation.

#### Separation of algorithmic and physical synchrony. 
Inference in PCNs is described here as a synchronous algorithm: each update step operates on a fixed snapshot of the network state. This ensures convergence and simplifies analysis. In principle, the underlying computations could be implemented asynchronously in hardware, without needing a global clock or centralized scheduling.

#### Decentralized control. 
Unlike backpropagation, which requires tightly coordinated forward and backward passes, PCNs do not rely on a global gradient tape or synchronized dataflow. This allows them to operate on hardware with minimal coordination or shared memory.

#### Energy-efficient, adaptive inference. 
PCNs support variable-length inference: predictable inputs can settle in fewer steps, while uncertain or surprising stimuli drive deeper inference. This adaptivity enables anytime computation, useful in energy-constrained settings such as embedded devices or robotics.

#### Architectural versatility. 
PCNs extend naturally to convolutional, recurrent, and graph-based structures by redefining local prediction and feedback pathways.

In short, predictive coding offers a biologically inspired view of computation whose design pattern aligns with the demands of emerging hardware. These properties make PCNs a high-value target for research in scalable, low-power, and distributed learning systems [6, 19].

## 4 Base Algorithms

### 4.1 Unsupervised learning in PCNs
>？无监督的算法根据什么来优化参数

This algorithm implements unsupervised learning in a predictive coding network with $L$ layers of latent variables $\mathbf{x}^{(1)},\ldots,\mathbf{x}^{(L)}$, with the input variables $\mathbf{x}^{(0)}$ clamped to the input data; recall Figure 1 in Section 1. The latent variables are inferred via iterative updates to minimize the global prediction error energy as discussed earlier, starting from a random initial state. Each inference step uses a consistent snapshot of the network: all prediction errors and gradients are computed before any latent state is updated.

Each layer $l$ receives top-down predictions from layer $l+1$ through a learned weight matrix $\mathbf{W}^{(l)}$ and nonlinearity $f^{(l)}{}$. After inference, weights are updated using local Hebbian-like learning rules based on the final prediction errors.
```
Layer L    x^(L) ──预测──→ x^(L-1) 的预测值
   ↑ W^(L-1)
Layer L-1  x^(L-1) ──预测──→ x^(L-2) 的预测值
   ↑ W^(L-2)
   ...
Layer 1    x^(1) ──预测──→ x^(0) 的预测值
   ↑ W^(0)
Layer 0    x^(0) = 输入数据（被固定）
```

**Per-sample training**（单样本训练）. The base algorithm presented here describes a single inference-learning cycle for one input sample. Training proceeds by repeating this cycle over many samples drawn from a dataset. For each sample, latent variables are inferred and weights are updated once. 
For **mini-batch training**（小批量训练） discussed later, the inference loop can be run for the samples in the current batch in parallel, and the subsequent weight update(s) can be carried out using the mean gradient over the batch.

*Algorithm 1 Unsupervised learning in a predictive coding network*

Require: Input $\mathbf{x}^{(0)}$, generative weights $\{\mathbf{W}^{(l)}\}_{l=0}^{L-1}$, activation functions and their derivatives $\{f^{(l)}, f^{(l)^{\prime}}\}$, number of inference steps $T_{infer}$, learning rate $\eta_{learn}$, inference rate $\eta_{infer}$

Clamp $\mathbf{x}^{(0)}$ ← input data
for layer $l = 1$ to $L$ {
	Initialize $\mathbf{x}^{(l)}$ ← small random values *# 初始化隐变量为随机数值*
}
$\varepsilon^{(L)} \leftarrow \mathbf{0}$  *# Top layer has no prediction error*
for step $t = 1$ to $T_{infer}$ {  *# Inference update loop*
	for layer $l = 0$ to $L - 1$ {  *# Store network state snapshot*
		$\mathbf{a}^{(l)} \leftarrow \mathbf{W}^{(l)}\mathbf{x}^{(l+1)}$  ▷ Preactivation
		$\hat{\mathbf{x}}^{(l)} \leftarrow f^{(l)}(\mathbf{a}^{(l)})$  ▷ Prediction
		$\varepsilon^{(l)} \leftarrow \mathbf{x}^{(l)} - \hat{\mathbf{x}}^{(l)}$  ▷ Prediction error
	}
	for layer $l = 1$ to $L$ {  *# Update latents using snapshot*
		$\mathbf{g}_{\mathbf{x}}^{(l)} \leftarrow \varepsilon^{(l)} - \mathbf{W}^{(l-1)\top} \left( f^{(l-1)^{\prime}}(\mathbf{a}^{(l-1)}) \odot \varepsilon^{(l-1)} \right)$  ▷ Gradient wrt $\mathbf{x}^{(l)}$
		$\mathbf{x}^{(l)} \leftarrow \mathbf{x}^{(l)} - \eta_{infer} \mathbf{g}_{\mathbf{x}}^{(l)}$
	}
}
for layer $l = 0$ to $L - 1$ {  *# Weight update*
	$\mathbf{a}^{(l)} \leftarrow \mathbf{W}^{(l)}\mathbf{x}^{(l+1)}$
	$\hat{\mathbf{x}}^{(l)} \leftarrow f^{(l)}(\mathbf{a}^{(l)})$
	$\varepsilon^{(l)} \leftarrow \mathbf{x}^{(l)} - \hat{\mathbf{x}}^{(l)}$
	$\mathbf{g}_{\mathbf{W}}^{(l)} \leftarrow - \left( \varepsilon^{(l)} \odot f^{(l)^{\prime}}(\mathbf{a}^{(l)}) \right) \mathbf{x}^{(l+1)\top}$  ▷ Gradient wrt $\mathbf{W}^{(l)}$
	$\mathbf{W}^{(l)} \leftarrow \mathbf{W}^{(l)} - \eta_{learn} \mathbf{g}_{\mathbf{W}}^{(l)}$
}

> [!tip] Tip
> **训练**：先更新隐状态，后更新权重
> **推理**：不更新权重，只更新隐状态

Remark. The model can also support **anytime inference**（**随时推断**）, where well-predicted inputs converge in fewer steps, and additional inference steps are taken to improve predictions in ambiguous cases. This adaptivity offers potential energy savings in embedded or neuromorphic deployments. The algorithm can be modified in this spirit by choosing a sufficiently large maximum step count $T_{infer}$, and running the inference loop until either $T_{infer}$ steps have been performed or convergence has been detected. Here convergence means, for instance, that the norm of the latest update (or updates over a longer patience window) across all latent variables falls below a preset threshold. In machine learning terminology, this could be phrased as inference with sample-wise early stopping.
**备注。** 该模型还支持**随时推断（anytime inference）**：对于容易预测的输入，只需更少的步骤即可收敛；而对于模糊或不确定的情况，则可以执行额外的推断步骤来改善预测结果。这种自适应特性为嵌入式或神经形态硬件部署提供了潜在的节能空间。

按照这一思路，可以对算法进行修改：设置一个足够大的最大步数 ，然后运行推断循环，直到执行了  步或检测到收敛为止。这里的收敛可以定义为，例如，所有隐变量的最新更新（或更长耐心窗口内的更新）的范数低于某个预设阈值。用机器学习的术语来说，这可以表述为**带样本级早停的推断（sample-wise early stopping）**。

### 4.2 Supervised learning extension

A minimal modification to apply predictive coding in a supervised setting entails simply clamping the top latent representation $\mathbf{x}^{(L)}$ to a predicted label $\hat{\mathbf{y}} \in \mathbb{R}^{d_{\text{out}}}$, treating it as part of the generative hierarchy. 
在监督场景下应用预测编码，一个最小改动就是将最顶层的潜在表示 $\mathbf{x}^{(L)}$ 钳制（固定）到预测标签 $\hat{\mathbf{y}} \in \mathbb{R}^{d_{\text{out}}}$，并将其视为生成层次结构的一部分。

we introduce a separate **readout layer** that maps $\mathbf{x}^{(L)} \mapsto \hat{\mathbf{y}}$ linearly（**读出层**）:
$$\hat{\mathbf{y}}=\mathbf{W}^{\mathrm{out}}\mathbf{x}^{(L)}$$

where $\mathbf{W}^{\text{out}} \in \mathbb{R}^{d_{\text{out}} \times d_L}$. Given a target label $\mathbf{y} \in \mathbb{R}^{d_{\text{out}}}$, we define a **supervised error**：
$$\varepsilon^{\mathrm{sup}}=\hat{\mathbf{y}}-\mathbf{y}.$$

Figure 2 illustrates these changes relative to Figure 1. The loss function (energy) now becomes
$$\mathcal{L}+\mathcal{L}_{\mathrm{sup}}$$

where $\mathcal{L}_{\mathrm{sup}} = \frac{1}{2} \|\boldsymbol{\varepsilon}^{\mathrm{sup}}\|^2$ is the supervised energy.

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//6999a80c-1f77-4042-951c-2c12ce7ebd2d/markdown_3/imgs/img_in_image_box_456_316_738_579.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-06-15T03%3A39%3A23Z%2F-1%2F%2Fe3df08118ef25b030ebbd2aedd4e699a137dcf7c621f531cc8ef299d5e2a7384" alt="Image" width="23%" /></div>

<div style="text-align: center;color: #2ecc71;"><div style="text-align: center;">Figure 2: A supervised extension of the PCN with three latent layers in Figure 1 is obtained by stacking a readout layer on top of the highest latent layer, together with a new root node for the target label.</div> </div>


The supervised error is backpropagated into the top latent representation $\mathbf{x}^{(L)}$ during inference. This does not change the update rules of the lower latents $\mathbf{x}^{(l)}$ or the generative weights $\mathbf{W}^{(l)}$, $0 \leq l < L$, at all. Since $\nabla_{\mathbf{x}^{(L)}} \mathcal{L}_{\mathrm{sup}} = \mathbf{W}^{\mathrm{out}\top} \boldsymbol{\varepsilon}^{\mathrm{sup}}$, the inference update rule for the top latent is modified to
在推断过程中，监督误差会反向传播到顶层潜在表示 $\mathbf{x}^{(L)}$。这**完全不会**改变下层潜在变量 $\mathbf{x}^{(l)}$ 或生成权重 $\mathbf{W}^{(l)}$（其中 $0 \leq l < L$）的更新规则。由于 $\nabla_{\mathbf{x}^{(L)}} \mathcal{L}_{\mathrm{sup}} = \mathbf{W}^{\mathrm{out}\top} \boldsymbol{\varepsilon}^{\mathrm{sup}}$，顶层潜在变量的推断更新规则被修改为

$$\left|\mathbf{x}^{(L)}\gets\mathbf{x}^{(L)}-\eta_{\mathrm{infer}}\left(\mathbf{W}^{\mathrm{out}\top}\boldsymbol{\varepsilon}^{\mathrm{sup}}-\mathbf{W}^{(L-1)\top}\left(f^{(L-1)^{\prime}}(\mathbf{a}^{(L-1)})\odot\boldsymbol{\varepsilon}^{(L-1)}\right)\right)\right|.$$

Note that this becomes formally the same as in the unsupervised case if we introduce the symbol $\boldsymbol{\varepsilon}^{(L)} = \mathbf{W}^{\mathrm{out}\top} \boldsymbol{\varepsilon}^{\mathrm{sup}}$ (instead of $\boldsymbol{\varepsilon}^{(L)} = \mathbf{0}$), which is how we implement the algorithm.

On the other hand, $\nabla_{\mathbf{W}^{\mathrm{out}}}\mathcal{L}_{\mathrm{sup}} = \varepsilon^{\mathrm{sup}}\mathbf{x}^{(L)\top}$. The output weights $\mathbf{W}^{\mathrm{out}}$ are thus updated via

$$\left|\mathbf{W}^{\mathrm{out}}\gets\mathbf{W}^{\mathrm{out}}-\eta_{\mathrm{learn}}\varepsilon^{\mathrm{sup}}\mathbf{x}^{(L)\top}\right|.$$

The core network structure remains unchanged from the unsupervised case; the only modification is the supervised error signal applied to the top layer.

## 5 Application: Supervised Learning on CIFAR-10
代码文件夹（本地）：
[pcn_cifar10_notebook](file:///D:\code\pcn-intro/)

### Define latent layer
```python
class PCNLayer(nn.Module):
    def __init__(self,
                 in_dim,   # d_{l+1}  - dimension of layer above
                 out_dim,  # d_l      - dimension of current layer
                 activation_fn=torch.relu, # nonlinearity f^(l)
                 activation_deriv=lambda a: (a > 0).float() # derivative f^(l)'
                 ):
        super().__init__()
        self.W = nn.Parameter(torch.empty(out_dim, in_dim)) # W^(l)
        nn.init.xavier_uniform_(self.W)
        self.activation_fn     = activation_fn
        self.activation_deriv  = activation_deriv

    def forward(self, x_above):
        with autocast(device_type='cuda'):
            a     = x_above @ self.W.T      # A^(l) = X^(l+1) @ {W^(l)}^T
            x_hat = self.activation_fn(a)   # \hat X^(l) = f^(l)(A^(l))
            return x_hat, a
```

### Define network structure
```python
class PredictiveCodingNetwork(nn.Module):
    def __init__(self,
                 dims,        # [d_0,...,d_L]  - list of layer dimensions
                 output_dim   # d_out          - readout layer dimension
                 ):
        super().__init__()
        self.dims = dims
        self.L = len(dims) - 1            # L  - number of latent layers
        self.layers = nn.ModuleList([     # Build latent layers
            PCNLayer(in_dim=dims[l+1],    # Layer l reads from layer l+1
                     out_dim=dims[l])
            for l in range(self.L)        # l = 0,...,L-1
        ])
        # Build readout layer: maps top latent X^(L) to
        # predicted output \hat Y
        # Note: nn.Linear applies (batch, in_features) @ weight.T under the hood,
        # which corresponds exactly to X^(L) @ (W^out)^T
        self.readout = nn.Linear(dims[-1], output_dim, bias=False)


    def init_latents(self, batch_size, device):
        # returns [X^(1),...,X^(L)] as random normals
        return [
           torch.randn(batch_size, d, device=device, requires_grad=False)
           for d in self.dims[1:]
        ]

    def compute_errors(self, inputs_latents):
        # Compute predictions from input and latent variables
        # Argument: inputs_latents - list of tensors [X^(0), X^(1),...,X^(L)] shaped [(B,d_0),...,(B,d_L)]
        # Returns: two lists of tensors shaped [(B,d_0),...,(B,d_{L-1})]
        errors, gain_modulated_errors = [], []
        for l, layer in enumerate(self.layers):       # l = 0,...,L-1
            # Call to layer returns:
            #   a = X^(l+1) @ W^(l).T  (preactivations A^(l))
            #   x_hat = f^(l)(a)       (predictions \hat X^(l))
            x_hat, a  = layer(inputs_latents[l + 1])
            err       = inputs_latents[l] - x_hat # 隐藏变量 - 上层传下来的预测值
            gm_err    = err * layer.activation_deriv(a)
            errors.append(err)                               # E^(l) - prediction errors
            gain_modulated_errors.append(gm_err)             # H^(l) - gain-modulated errors
        return errors, gain_modulated_errors
```
### 计算误差
```python
# 从每个隐藏层计算
errors, gain_modulated_errors = model.compute_errors(inputs_latents)。
# 输出层单独计算
y_hat           = model.readout(inputs_latents[-1]) # X^(L) @ W^out.T
eps_sup         = y_hat - y_batch
eps_L           = eps_sup @ weights[-1]
errors_extended = errors + [eps_L]
```

### 更新隐变量(inference)：
```python
for l in range(1, model.L + 1):  # l=1,...,L
	# 计算梯度
	grad_Xl = errors_extended[l] - gain_modulated_errors[l-1] @ weights[l-1]
	# 更新参数
	inputs_latents[l] -= eta_infer * grad_Xl
```

### 更新权重(learning)：
```python
for l in range(model.L): # l=0,...,L-1
	# 计算梯度
	grad_Wl = -(gain_modulated_errors[l].T @ inputs_latents[l+1]) / B
	# 更新参数
	weights[l] -= eta_learn * grad_Wl
```


## References

[1] Nick Alonso. Predictive coding: A brief introduction and review for machine learning researchers, 2022. Blog post. URL: https://neuralnetnick.com/2022/12/28/.

[2] Nick Alonso, Beren Millidge, Jeff Krichmar, and Emre Neftci. A theoretical framework for inference learning. In Proceedings of the 36th International Conference on Neural Information Processing Systems, volume 36, pages 37335–37348, 2022. URL: https://dl.acm.org/doi/10.5555/3600270.3602976.

[3] Horace B. Barlow. Possible principles underlying the transformation of sensory messages. In W. A. Rosenblith, editor, Sensory Communication, pages 217–234. MIT Press, Cambridge, MA, 1961.

[4] Andre M. Bastos, W. Martin Usrey, Rick A. Adams, George R. Mangun, Pascal Fries, and Karl J. Friston. Canonical microcircuits for predictive coding. Neuron, 76(4):695–711, 2012. doi:10.1016/j.neuron.2012.10.038.

[5] C. Caucheteux, A. Gramfort, and JR King. Evidence of a predictive coding hierarchy in the human brain listening to speech. Nature Human Behaviour, 7:430–441, 2023. doi:10.1038/s41562-022-01516-2.

[6] Mike Davies, Narayan Srinivasa, Tsung-Han Lin, Gautham Chinya, Yongqiang Cao, Sri Harsha Choday, George Dimou, Harish Joshi, Nabil Imam, Shweta Jain, et al. Loihi: A neuromorphic manycore processor with on-chip learning. IEEE Micro, 38(1):82–99, 2018. doi:10.1109/MM.2018.112130359.

[7] Simon Frieder and Thomas Lukasiewicz. (Non-)Convergence results for predictive coding networks. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 6793–6810, 2022. URL: https://proceedings.mlr.press/v162/frieder22a.html.

[8] Karl Friston. A theory of cortical responses. Philosophical Transactions of the Royal Society B: Biological Sciences, 360(1456):815–836, 2005. doi:10.1098/rstb.2005.1622.

[9] Karl Friston. The free-energy principle: a unified brain theory? Nature Reviews Neuroscience, 11(2):127–138, 2010. doi:10.1038/nrn2787.

[10] Richard L. Gregory. Perceptions as hypotheses. Philosophical Transactions of the Royal Society of London B: Biological Sciences, 290(1038):181–197, 1980. doi:10.1098/rstb.1980.0090.

[11] David Ha and Jürgen Schmidhuber. World models. CoRR, abs/1803.10122, 2018. URL: http://arxiv.org/abs/1803.10122.

[12] Hermann von Helmholtz. Treatise on Physiological Optics, Volume III: Concerning the perceptions in general. Dover, New York, 1867. Translated by J. P. C. Southall, 3rd ed., 1962.

[13] Georg B. Keller and Thomas D. Mrsic-Flogel. Predictive processing: A canonical cortical computation. Neuron, 100(2):424–435, 2018. doi:10.1016/j.neuron.2018.10.003.

[14] Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009. URL: https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf.

[15] William Lotter, Gabriel Kreiman, and David Cox. Deep predictive coding networks for video prediction and unsupervised learning. In International Conference on Learning Representations (ICLR), 2017. URL: https://arxiv.org/abs/1605.08104.

[16] Ankur Mali, Tommaso Salvatori, and Alexander Ororbia. Tight stability, convergence, and robustness bounds for predictive coding networks. 2024. doi:10.48550/arXiv.2410.04708.

[17] Beren Millidge. Predictive coding as backprop and natural gradients, 2020. Blog post. URL: https://www.beren.io/2020-09-12-Predictive-Coding-As-Backprop-And-Natural-Gradients/.

[18] Beren Millidge, Anil K. Seth, and Christopher L. Buckley. Predictive coding: a theoretical and experimental review. 2021. URL: https://arxiv.org/abs/2107.12979.

[19] Beren Millidge, Alexander Tschantz, and Christopher L. Buckley. Predictive coding approximates backprop along arbitrary computation graphs. Neural Computation, 34:1329–1368, 2022. doi:10.1162/neco_a_01497.

[20] Monadillo. An introduction to predictive coding networks for machine learning, 2025. GitHub repository containing the following supplements to this document: Python notebook and model weights. URL: https://github.com/Monadillo/pcn-intro.

[21] David Mumford. On the computational architecture of the neocortex. ii. the role of cortico-cortical loops. Biological Cybernetics, 66(3):241–251, 1992. doi:10.1007/BF00198477.

[22] Antony W. N'dri, William Gebhardt, Céline Teulière, Fleur Zeldenrust, Rajesh P. N. Rao, Jochen Triesch, and Alexander Ororbia. Predictive coding with spiking neural networks: a survey. 2024. URL: https://arxiv.org/abs/2409.05386.

[23] Rajesh P. N. Rao and Dana H. Ballard. Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. Nature neuroscience, 2(1):79–87, 1999. doi:10.1038/4580.

[24] Tommaso Salvatori, Yuhang Song, Yordan Yordanov, Beren Millidge, Zhenghua Xu, Lei Sha, Cornelius Emde, Rafal Bogacz, and Thomas Lukasiewicz. A stable, fast, and fully automatic learning algorithm for predictive coding networks. In International Conference on Learning Representations, 2024. doi:10.48550/arXiv.2212.00720.

[25] Jürgen Schmidhuber. Learning factorial codes by predictability minimization. Neural Computation, 4(6):863–879, 1992. doi:10.1162/neco.1992.4.6.863.

[26] Stewart Shipp. Neural elements for predictive coding. Frontiers in Psychology, 7:1792, 2016. doi:10.3389/fpsyg.2016.01792.

[27] Yuhang Song, Thomas Lukasiewicz, Zhenghua Xu, and Rafal Bogacz. Can the brain do back propagation? — exact implementation of backpropagation in predictive coding networks. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 22566–22579. Curran Associates, Inc., 2020. URL: https://proceedings.neurips.cc/paper_files/paper/2020/file/fec87a37cdecc1c6ecf8181c0aa2d3bf-Paper.pdf.

[28] M.W. Spratling. Reconciling predictive coding and biased competition models of cortical function. Frontiers in Computational Neuroscience, 2:4, 2008. doi:10.3389/neuro.10.004.2008.

[29] Alexander Tschantz, Beren Millidge, Anil K. Seth, and Christopher L. Buckley. Hybrid predictive coding: Inferring, fast and slow. PLOS Computational Biology, 19(8):1–31, 082023. doi:10.1371/journal.pcbi.1011280.

[30] K. S. Walsh, D. P. McGovern, A. Clark, and R. G. O'Connell. Evaluating the neurophysiological evidence for predictive processing as a model of perception. Ann N Y Acad Sci, 464(1):242–268, 3 2020. doi:10.1111/nyas.14321.

[31] James C. R. Whittington and Rafal Bogacz. An approximation of the error backpropagation algorithm in a predictive coding network with local hebbian synaptic plasticity. Neural Computation, 29:1229–1262, 2017. doi:10.1162/NECO_a_00949.

