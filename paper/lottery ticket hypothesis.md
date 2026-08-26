---
sr-due: 2026-08-25
sr-interval: 1
sr-ease: 230
---
#paper 

arXiv:1803.03635v5 [cs.LG] 4 Mar 2019

# THE LOTTERY TICKET HYPOTHESIS: FINDING SPARSE, TRAINABLE NEURAL NETWORKS

Jonathan Frankle

MIT CSAIL

jfrankle@csail.mit.edu

Michael Carbin

MIT CSAIL

mcarbin@csail.mit.edu

## ABSTRACT

Neural network pruning techniques can reduce the parameter counts of trained networks by over 90%, decreasing storage requirements and improving computational performance of inference without compromising accuracy. However, contemporary experience is that the sparse architectures produced by pruning are difficult to train from the start, which would similarly improve training performance.

We find that a standard pruning technique naturally uncovers subnetworks whose initializations made them capable of training effectively. Based on these results, we articulate the lottery ticket hypothesis: dense, randomly-initialized, feed-forward networks contain subnetworks (winning tickets) that—when trained in isolation—reach test accuracy comparable to the original network in a similar number of iterations. The winning tickets we find have won the initialization lottery: their connections have initial weights that make training particularly effective.

We present an algorithm to identify winning tickets and a series of experiments that support the lottery ticket hypothesis and the importance of these fortuitous initializations. We consistently find winning tickets that are less than 10-20% of the size of several fully-connected and convolutional feed-forward architectures for MNIST and CIFAR10. Above this size, the winning tickets that we find learn faster than the original network and reach higher test accuracy.

## 1 引言（Introduction）

消除神经网络中不必要权重的技术（剪枝，pruning）（LeCun et al., 1990; Hassibi & Stork, 1993; Han et al., 2015; Li et al., 2016）可以在不损害精度的情况下把参数量削减 90% 以上。这样做缩小了已训练网络的规模（Han et al., 2015; Hinton et al., 2015）或降低了能耗（Yang et al., 2017; Molchanov et al., 2016; Luo et al., 2017），使推理更加高效。然而，如果网络可以被缩小，我们为什么不直接训练这个更小的结构、从而让训练本身也更高效呢？当代的经验是，剪枝得到的架构难以从头开始训练，其精度低于原网络。 $^{1}$

来看一个例子。在图 1 中，我们从一个用于 MNIST 的全连接网络和用于 CIFAR10 的卷积网络中随机采样并训练子网络。随机采样模拟了 LeCun et al. (1990) 与 Han et al. (2015) 所用的非结构化剪枝的效果。在不同稀疏度下，虚线描绘了最小验证损失所在的迭代 $^{2}$ 以及该迭代处的测试精度。网络越稀疏，学习越慢，最终测试精度也越低。

 $^{1}$”从头训练一个剪枝后的模型，其表现比重训一个剪枝后的模型更差，这可能表明训练小容量网络是困难的。”（Li et al., 2016）”在重训过程中，对于在剪枝中存活下来的连接，保留其初始训练阶段的权重，要优于重新初始化被剪枝的层……梯度下降在网络初次训练时能找到好的解，但在重新初始化某些层并重训之后却做不到。”（Han et al., 2015）

 $^{2}$作为网络学习速度的代理指标，我们使用早停（early-stopping）准则会结束训练的那个迭代。本文通篇采用的特定早停准则是训练过程中验证损失最低的那个迭代。关于这一选择的更多细节见附录 C。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//5c3339b9-fb04-4b1b-849f-3e72c9ab0b04/markdown_1/imgs/img_in_chart_box_219_178_1000_335.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A15Z%2F-1%2F%2F218b7ff84e6c6c6bab484a575ae7cced72c91e7acac9e23da832c4175e57b86b" alt="Image" width="63%" /></div>


<div style="text-align: center;color: lightgreen"><div style="text-align: center;">图 1：用于 MNIST 的 Lenet 架构与用于 CIFAR10 的 Conv-2、Conv-4、Conv-6 架构（见图 2）在不同初始规模下训练时，早停发生的迭代（左）与该迭代处的测试精度（右）。虚线为随机采样的稀疏网络（十次试验的平均）；实线为中奖彩票（五次试验的平均）。</div> </div>


在本文中，我们证明始终存在更小的子网络，它们能从初始状态开始训练，学习速度至少与其较大的对应网络一样快，同时达到相近的测试精度。图 1 中的实线显示了我们找到的网络。基于这些结果，我们提出彩票假说。

彩票假说（The Lottery Ticket Hypothesis）。一个随机初始化的稠密神经网络包含一个子网络，该子网络的初始化方式使得——当它被单独训练时——能在最多相同次数的迭代后达到原网络的测试精度。

更形式化地说，考虑一个稠密前馈神经网络  $f(x; \theta)$ ，其初始参数为  $\theta = \theta_0 \sim \mathcal{D}_\theta$ 。在训练集上用随机梯度下降（SGD）优化时，  $f$  在迭代  $j$  处达到最小验证损失  $l$ ，测试精度为  $a$ 。此外，考虑用掩码  $m \in \{0, 1\}^{|\theta|}$  作用于参数来训练  $f(x; m \odot \theta)$ ，使其初始化为  $m \odot \theta_0$ 。在同一训练集上用 SGD 优化（ $m$ 固定）时，  $f$  在迭代  $j'$  处达到最小验证损失  $l'$ ，测试精度为  $a'$ 。彩票假说预测存在这样的  $m$ ，使得  $j' \leq j$ （训练时间相当）、 $a' \geq a$ （精度相当）且  $\|m\|_0 \ll |\theta|$ （参数更少）。

我们发现，一种标准的剪枝技术能从全连接和卷积前馈网络中自动找出这类可训练子网络。我们将这些可训练子网络  $f(x; m \odot \theta_0)$  称为中奖彩票（winning tickets），因为我们找到的这些子网络凭借一种能够学习的权重与连接的组合赢得了初始化彩票。当它们的参数被随机重新初始化（即  $f(x; m \odot \theta'_0)$ ，其中  $\theta'_0 \sim \mathcal{D}_\theta$ ）时，中奖彩票不再能达到原网络的性能，这为以下观点提供了证据：这些较小的网络只有在恰当初始化时才能有效训练。

识别中奖彩票。我们通过训练一个网络并剪掉其幅度最小的权重来识别中奖彩票。剩余未被剪枝的连接构成中奖彩票的架构。我们工作的独特之处在于，每个未被剪枝的连接的值随后会被重置回其训练之前原网络中的初始值。这构成了我们的核心实验：

1. 随机初始化一个神经网络  $f(x;\theta_0)$ （其中  $\theta_0\sim\mathcal{D}_\theta$ ）。

2. 训练网络  $j$  次迭代，得到参数  $\theta_{j}$ 。

3. 剪掉  $\theta_{j}$  中  $p\%$  的参数，得到掩码  $m$ 。

4. 把剩余参数重置回其在  $\theta_0$  中的值，得到中奖彩票  $f(x; m\odot\theta_0)$ 。

如上所述，这种剪枝方法是一次性（one-shot）的：网络训练一次，剪掉 $p\%$ 的权重，存活权重被重置。然而，在本文中，我们聚焦于迭代剪枝（iterative pruning），它在 $n$ 轮中反复执行训练、剪枝和重置；每一轮剪掉上一轮存活权重中的 $p^{\frac{1}{n}}\%$。我们的结果表明，迭代剪枝找到的中奖彩票在比一次性剪枝更小的规模下就能匹配原网络的精度。

结果。我们在用于 MNIST 的全连接架构和用于 CIFAR10 的卷积架构上，跨多种优化策略（SGD、momentum、Adam）并使用 dropout、权重衰减、批归一化和残差连接等技术识别出中奖彩票。我们使用非结构化剪枝技术，因此这些中奖彩票是稀疏的。在更深层的网络中，我们基于剪枝寻找中奖彩票的策略对学习率敏感：在较高学习率下需要 warmup 才能找到中奖彩票。我们找到的中奖彩票规模为原网络的 10–20%（或更小）。



<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>网络</td><td style='text-align: center; word-wrap: break-word;'>Lenet</td><td style='text-align: center; word-wrap: break-word;'>Conv-2</td><td style='text-align: center; word-wrap: break-word;'>Conv-4</td><td style='text-align: center; word-wrap: break-word;'>Conv-6</td><td style='text-align: center; word-wrap: break-word;'>Resnet-18</td><td style='text-align: center; word-wrap: break-word;'>VGG-19</td></tr><tr><td rowspan="3">卷积层</td><td rowspan="3"></td><td rowspan="3">64, 64, pool</td><td style='text-align: center; word-wrap: break-word;'>64, 64, pool</td><td style='text-align: center; word-wrap: break-word;'>64, 64, pool</td><td style='text-align: center; word-wrap: break-word;'>16, 3x[16, 16]</td><td style='text-align: center; word-wrap: break-word;'>2x64 pool 2x128</td></tr><tr><td rowspan="2">128, 128, pool</td><td style='text-align: center; word-wrap: break-word;'>128, 128, pool</td><td style='text-align: center; word-wrap: break-word;'>3x[32, 32]</td><td style='text-align: center; word-wrap: break-word;'>pool, 4x256, pool</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>256, 256, pool</td><td style='text-align: center; word-wrap: break-word;'>3x[64, 64]</td><td style='text-align: center; word-wrap: break-word;'>4x512, pool, 4x512</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>全连接层</td><td style='text-align: center; word-wrap: break-word;'>300, 100, 10</td><td style='text-align: center; word-wrap: break-word;'>256, 256, 10</td><td style='text-align: center; word-wrap: break-word;'>256, 256, 10</td><td style='text-align: center; word-wrap: break-word;'>256, 256, 10</td><td style='text-align: center; word-wrap: break-word;'>avg-pool, 10</td><td style='text-align: center; word-wrap: break-word;'>avg-pool, 10</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>总/卷积权重</td><td style='text-align: center; word-wrap: break-word;'>266K</td><td style='text-align: center; word-wrap: break-word;'>4.3M / 38K</td><td style='text-align: center; word-wrap: break-word;'>2.4M / 260K</td><td style='text-align: center; word-wrap: break-word;'>1.7M / 1.1M</td><td style='text-align: center; word-wrap: break-word;'>274K / 270K</td><td style='text-align: center; word-wrap: break-word;'>20.0M</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>迭代次数/批</td><td style='text-align: center; word-wrap: break-word;'>50K / 60</td><td style='text-align: center; word-wrap: break-word;'>20K / 60</td><td style='text-align: center; word-wrap: break-word;'>25K / 60</td><td style='text-align: center; word-wrap: break-word;'>30K / 60</td><td style='text-align: center; word-wrap: break-word;'>30K / 128</td><td style='text-align: center; word-wrap: break-word;'>112K / 64</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>优化器</td><td style='text-align: center; word-wrap: break-word;'>Adam 1.2e-3</td><td style='text-align: center; word-wrap: break-word;'>Adam 2e-4</td><td style='text-align: center; word-wrap: break-word;'>Adam 3e-4</td><td style='text-align: center; word-wrap: break-word;'>Adam 3e-4</td><td colspan="2">$\leftarrow$ SGD 0.1-0.01-0.001 Momentum 0.9  $\rightarrow$</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>剪枝率</td><td style='text-align: center; word-wrap: break-word;'>fc20%</td><td style='text-align: center; word-wrap: break-word;'>conv10% fc20%</td><td style='text-align: center; word-wrap: break-word;'>conv10% fc20%</td><td style='text-align: center; word-wrap: break-word;'>conv15% fc20%</td><td style='text-align: center; word-wrap: break-word;'>conv20% fc0%</td><td style='text-align: center; word-wrap: break-word;'>conv20% fc0%</td></tr></table>

<div style="text-align: center;"><div style="text-align: center;">图 2：本文测试的架构。卷积核为 3x3。Lenet 来自 LeCun et al. (1998)。Conv-2/4/6 是 VGG（Simonyan & Zisserman, 2014）的变体。Resnet-18 来自 He et al. (2016)。用于 CIFAR10 的 VGG-19 改编自 Liu et al. (2019)。初始化采用 Gaussian Glorot（Glorot & Bengio, 2010）。括号表示围绕层的残差连接。</div> </div>


（更小的规模）。在这一规模之下，它们在最多相同次数的迭代（训练时间相当）内达到或超过原网络的测试精度（精度相当）。当随机重新初始化时，中奖彩票的表现要差得多，这意味着单凭结构无法解释中奖彩票的成功。

彩票猜想（The Lottery Ticket Conjecture）。回到我们的动机问题，我们把假设扩展为一个未经检验的猜想：SGD 会寻找并训练一个由初始化良好的权重构成的子集。稠密、随机初始化的网络比剪枝得到的稀疏网络更容易训练，因为前者拥有更多可能的子网络，训练可以从中恢复出一个中奖彩票。

贡献（Contributions）。

- 我们证明，剪枝能找出可训练的子网络，它们在相近的迭代次数内达到与原网络（即其来源网络）相当的测试精度。

- 我们表明，剪枝找到的中奖彩票比原网络学得更快，同时达到更高的测试精度并具有更好的泛化能力。

- 我们提出彩票假说，作为关于神经网络构成的新视角来解释这些发现。

意义（Implications）。在本文中，我们对彩票假说进行了实证研究。既然我们已经证明了中奖彩票的存在，我们希望利用这一知识来：

提升训练性能。由于中奖彩票可以从初始状态单独训练，我们希望设计出能够尽早搜索中奖彩票并剪枝的训练方案。

设计更好的网络。中奖彩票揭示了特别擅长学习的稀疏架构与初始化的组合。我们可以从中奖彩票中汲取灵感，设计具有同样利于学习特性的新架构和初始化方案。我们甚至可能把为某个任务发现的中奖彩票迁移到许多其他任务上。

提升对神经网络的理论理解。我们可以研究为什么随机初始化的前馈网络似乎包含中奖彩票，以及这对优化（Du et al., 2019）和泛化（Zhou et al., 2018; Arora et al., 2018）的理论研究可能带来的启示。

## 2 全连接网络中的中奖彩票（Winning Tickets in Fully-Connected Networks）

在本节中，我们评估彩票假说在 MNIST 上训练的全连接网络上的适用性。我们使用图 2 中描述的 Lenet-300-100 架构（LeCun et al., 1998）。我们遵循第 1 节的流程：随机初始化并训练一个网络后，对网络进行剪枝，并把剩余连接重置回其原始初始值。我们采用一种简单的逐层剪枝启发式：在每层内移除一定比例幅度最小的权重（同 Han et al. (2015)）。通向输出的连接按网络其余部分一半的速率剪枝。我们在附录 G 中探索了其他超参数，包括学习率、优化策略（SGD、momentum）、初始化方案和网络规模。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//5c3339b9-fb04-4b1b-849f-3e72c9ab0b04/markdown_3/imgs/img_in_chart_box_226_176_995_416.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A16Z%2F-1%2F%2Fa586d46de7eb732b20b8a9689576c6e1b4eb87b2bf5a41d7f1473df963bc139a" alt="Image" width="62%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图 3：训练过程中 Lenet（迭代剪枝）的测试精度。每条曲线为五次试验的平均。标签  $P_{m}$ 表示剪枝后网络中剩余权重的比例。误差线为任意试验的最小值与最大值。</div> </div>


记号。 $P_m = \frac{\|m\|_0}{|\theta|}$ 是掩码  $m$ 的稀疏度，例如当 75% 的权重被剪掉时  $P_m = 25\%$ 。

迭代剪枝。我们找到的中奖彩票比原网络学得更快。图 3 描绘了将中奖彩票迭代剪枝到不同程度后训练时的平均测试精度。误差线为五次运行的最小值与最大值。在最初的几轮剪枝中，剪枝越多，网络学得越快、测试精度越高（图 3 左图）。一个包含原网络 51.3% 权重的中奖彩票（即  $P_m = 51.3\%$ ）比原网络更快地达到更高的测试精度，但比  $P_m = 21.1\%$ 时慢。当  $P_m < 21.1\%$ 时，学习变慢（中图）。当  $P_m = 3.6\%$ 时，中奖彩票回落到原网络的性能。类似规律在本文中反复出现。

图 4a 总结了每轮迭代剪掉 20% 时所有剪枝程度下的这一行为（蓝色）。左侧是每个网络达到最小验证损失（即早停准则会停止训练的时刻）的迭代与剪枝后剩余权重比例的关系；中间是该迭代处的测试精度。我们用满足早停准则的那个迭代作为网络学习速度的代理指标。

随着  $P_{m}$ 从 100% 降到 21%，中奖彩票学得更快，在 21% 处早停比原网络提前 38%。进一步剪枝导致学习变慢，当  $P_{m} = 3.6\%$ 时回落到原网络的早停性能。测试精度随剪枝而提高，当  $P_{m} = 13.5\%$ 时提升超过 0.3 个百分点；此后精度下降，当  $P_{m} = 3.6\%$ 时回落到原网络水平。

在早停点，训练精度（图 4a 右）随剪枝以与测试精度类似的模式上升，这似乎暗示中奖彩票优化得更有效但泛化并未更好。然而，在第 50,000 次迭代（图 4b），尽管几乎所有网络的训练精度都达到了 100%（附录 D，图 12），迭代剪枝得到的中奖彩票的测试精度仍有最高 0.35 个百分点的提升。这意味着中奖彩票的训练精度与测试精度之间的差距更小，表明其泛化能力得到了改善。

随机重新初始化。为了衡量中奖彩票初始化的重要性，我们保留中奖彩票的结构（即掩码 $m$ ），但随机采样一个新的初始化 $\theta'_0 \sim \mathcal{D}_\theta$ 。我们对每个中奖彩票随机重新初始化三次，图 4 中每个点共 15 次。我们发现初始化对中奖彩票的有效性至关重要。图 3 右图展示了针对迭代剪枝的这一实验。除了原网络以及  $P_m = 51\%$ 和 21% 的中奖彩票之外，还有随机重新初始化实验。中奖彩票在剪枝后学得更快，而被随机重新初始化后则学得越来越慢。

这一实验更广泛的结果是图 4a 中的橙色线。与中奖彩票不同，被重新初始化的网络学得比原网络越来越慢，且仅在少量剪枝后就损失测试精度。重新初始化的迭代中奖彩票的平均测试精度在  $P_m = 21.1\%$ 时从原始精度开始下降，而中奖彩票要到 2.9% 时才下降。当  $P_m = 21\%$ 时，中奖彩票达到最小验证损失的速度比重初始化时快 2.51 倍，且精度高出半个百分点。所有网络在  $P_m \geq 5\%$ 时都达到 100% 训练精度；图

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//93e6c083-175f-4d78-b0df-211eb02fb071/markdown_0/imgs/img_in_chart_box_218_179_474_372.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A14Z%2F-1%2F%2F01aee799e261a6c030ab929fcc3751697546e2adaa189c55850add8144dcb451" alt="Image" width="20%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//93e6c083-175f-4d78-b0df-211eb02fb071/markdown_0/imgs/img_in_chart_box_478_180_738_372.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A14Z%2F-1%2F%2Ff5ce0124c538f8534b199e5c78048dc980fba6627a2ecd90cd53dbbb093b9311" alt="Image" width="21%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//93e6c083-175f-4d78-b0df-211eb02fb071/markdown_0/imgs/img_in_chart_box_744_183_992_372.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A14Z%2F-1%2F%2F45261261df277da85bc578f8e1d346900720b211a858b2ff59a767522514785a" alt="Image" width="20%" /></div>


<div style="text-align: center;"><div style="text-align: center;">(a) 所有剪枝方法下的早停迭代与精度。</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//93e6c083-175f-4d78-b0df-211eb02fb071/markdown_0/imgs/img_in_chart_box_224_407_473_588.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A14Z%2F-1%2F%2F4c6ad9bb505227ded3bb91c40865f5114efdb187e53e68a4314b79f2b5fb3254" alt="Image" width="20%" /></div>


<div style="text-align: center;"><div style="text-align: center;">(b) 训练结束时的精度。</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//93e6c083-175f-4d78-b0df-211eb02fb071/markdown_0/imgs/img_in_chart_box_484_408_734_587.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A14Z%2F-1%2F%2F217bf43c368fc625132cdc413b70d345aab11c0e50f8ed74ff3d6c450eb9b1c6" alt="Image" width="20%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//93e6c083-175f-4d78-b0df-211eb02fb071/markdown_0/imgs/img_in_chart_box_746_412_994_587.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A15Z%2F-1%2F%2Fcebd7ff520ac9fab1796598cc77cee94492cf72f79eb331e2e623312053b7bee" alt="Image" width="20%" /></div>


<div style="text-align: center;"><div style="text-align: center;">(c) 一次性剪枝下的早停迭代与精度。</div> </div>


<div style="text-align: center;"><div style="text-align: center;">图 4：一次性剪枝与迭代剪枝下 Lenet 的早停迭代与精度。五次试验的平均；误差线为最小值与最大值。在第 50,000 次迭代，迭代中奖彩票在  $P_m \geq 2\%$ 时训练精度  $\approx$ 100%（见附录 D，图 12）。</div> </div>


因此图 4b 表明，中奖彩票的泛化能力显著优于随机重新初始化的情况。这一实验支持了彩票假说对初始化的强调：原始初始化能够承受并从剪枝中受益，而随机重新初始化的性能则立即受损并持续下降。

一次性剪枝。虽然迭代剪枝能提取更小的中奖彩票，但反复训练意味着寻找它们的代价高昂。一次性剪枝使我们无需反复训练就能识别中奖彩票。图 4c 展示了一次性剪枝（绿色）与随机重新初始化（红色）的结果；一次性剪枝确实能找到中奖彩票。当  $67.5\% \geq P_m \geq 17.6\%$ 时，平均中奖彩票比原网络更早达到最小验证精度。当  $95.0\% \geq P_m \geq 5.17\%$ 时，测试精度高于原网络。然而，迭代剪枝得到的中奖彩票在更小的网络规模下学得更快、测试精度更高。图 4c 中的绿线和红线被复制到图 4a 的对数坐标轴上，使这一性能差距更加清晰。由于我们的目标是识别尽可能小的中奖彩票，本文其余部分将聚焦于迭代剪枝。

## 3 卷积网络中的中奖彩票（Winning Tickets in Convolutional Networks）

在此，我们把彩票假说应用到 CIFAR10 上的卷积网络，既提高了学习问题的复杂度，也增大了网络规模。我们考虑图 2 中的 Conv-2、Conv-4、Conv-6 架构，它们是 VGG（Simonyan & Zisserman, 2014）家族缩小后的变体。这些网络分别有 2、4、6 个卷积层，后接两个全连接层；每两个卷积层后进行一次最大池化。这些网络覆盖了从接近全连接到传统卷积网络的范围，Conv-2 中卷积层参数占比不到 1%，而 Conv-6 中接近三分之二。 $^{3}$

寻找中奖彩票。图 5（上）的实线展示了以图 2 中的逐层剪枝率对 Conv-2（蓝色）、Conv-4（橙色）、Conv-6（绿色）进行的迭代彩票实验。第 2 节中 Lenet 的规律再次出现：随着网络被剪枝，与原网络相比，它学得更快，测试精度上升。在这种情况下，结果更加显著。中奖

 $^{3}$附录 H 探索了其他超参数，包括学习率、优化策略（SGD、momentum）以及剪枝卷积层与全连接层的相对速率。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//93e6c083-175f-4d78-b0df-211eb02fb071/markdown_1/imgs/img_in_chart_box_222_179_996_560.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A15Z%2F-1%2F%2F6c441d9ae4cfe385f3a20c7bc79cc3e339ddd655eac75571d99e91b482f8b15c" alt="Image" width="63%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图 5：迭代剪枝与随机重新初始化时 Conv-2/4/6 架构的早停迭代以及测试、训练精度。每条实线为五次试验的平均；每条虚线为十五次重新初始化（每次试验三次）的平均。右下方的图描绘了在原网络训练最后一次迭代对应的迭代处（Conv-2 为 20,000、Conv-4 为 25,000、Conv-6 为 30,000）中奖彩票的测试精度；在该迭代处，中奖彩票在  $P_m \geq 2\%$ 时训练精度  $\approx 100\%$（见附录 D）。</div> </div>


彩票达到最小验证损失的速度最好时快 3.5 倍（Conv-2，$P_m = 8.8\%$ ）、3.5 倍（Conv-4，$P_m = 9.2\%$ ）和 2.5 倍（Conv-6，$P_m = 15.1\%$ ）。测试精度最好时提升 3.4 个百分点（Conv-2，$P_m = 4.6\%$ ）、3.5 个百分点（Conv-4，$P_m = 11.1\%$ ）和 3.3 个百分点（Conv-6，$P_m = 26.4\%$ ）。当 $P_m > 2\%$ 时，三个网络都保持在其原始平均测试精度之上。

与第 2 节一样，早停迭代处的训练精度随测试精度一起上升。然而，在 Conv-2 的 20,000、Conv-4 的 25,000 和 Conv-6 的 30,000 次迭代（这些迭代对应原网络的最后一次训练迭代），当  $P_m \geq 2\%$ 时所有网络的训练精度都达到 100%（附录 D，图 13），而中奖彩票仍保持更高的测试精度（图 5 右下）。这意味着中奖彩票的测试与训练精度之间的差距更小，表明其泛化能力更好。

随机重新初始化。我们重复第 2 节的随机重新初始化实验，其结果如图 5 中的虚线。随着持续剪枝，这些网络再次需要越来越长的时间来学习。与 MNIST 上的 Lenet（第 2 节）一样，随机重新初始化实验的测试精度下降得更快。然而，与 Lenet 不同，Conv-2 和 Conv-4 在早停时刻的测试精度最初保持稳定甚至有所提升，这表明——在中等程度的剪枝下——仅凭中奖彩票的结构就可能带来更高的精度。

Dropout。Dropout（Srivastava et al., 2014; Hinton et al., 2012）通过在每次训练迭代时随机禁用一部分单元（即随机采样子网络）来提高精度。Baldi & Sadowski (2013) 将 dropout 刻画为同时训练所有子网络的集成。由于彩票假说暗示这些子网络中的某一个就构成中奖彩票，自然要问 dropout 与我们寻找中奖彩票的策略之间是否存在相互作用。

图 6 展示了以 0.5 的 dropout 率训练 Conv-2、Conv-4、Conv-6 的结果。虚线为无 dropout 时的网络性能（即图 5 中的实线）。 $^{4}$ 使用 dropout 训练时，我们仍能找到中奖彩票。dropout 提高了初始测试精度（Conv-2、Conv-4、Conv-6 平均分别提高 2.1、3.0、2.4 个百分点），迭代剪枝进一步提高（平均分别最多再提高 2.3、4.6、4.7 个百分点）。迭代剪枝使学习如以前一样变快，但在 Conv-2 上不那么显著。

 $^{4}$我们为使用 dropout 训练的网络选择了新的学习率——见附录 H.5。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//93e6c083-175f-4d78-b0df-211eb02fb071/markdown_2/imgs/img_in_chart_box_219_178_998_378.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A15Z%2F-1%2F%2Fc2aef846ef54fd04c17360dde5764613dac85393dfe6bd49086b8203e74bda87" alt="Image" width="63%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图 6：迭代剪枝并用 dropout 训练时 Conv-2/4/6 的早停迭代与早停处的测试精度。虚线为不使用 dropout 训练的相同网络（即图 5 中的实线）。学习率：Conv-2 为 0.0003，Conv-4 和 Conv-6 为 0.0002。</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//93e6c083-175f-4d78-b0df-211eb02fb071/markdown_2/imgs/img_in_chart_box_222_486_478_700.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A16Z%2F-1%2F%2F0157481f9b74c09f97b28b73eb00fd94bee5d9f5e4a2bcf8791b87d7ae3721b9" alt="Image" width="20%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//93e6c083-175f-4d78-b0df-211eb02fb071/markdown_2/imgs/img_in_chart_box_481_488_738_699.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A16Z%2F-1%2F%2F2931feb021caf80e21a50d35c06a7ffde01b41468b1d77ded7304399aa9834b1" alt="Image" width="20%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//93e6c083-175f-4d78-b0df-211eb02fb071/markdown_2/imgs/img_in_chart_box_744_490_996_698.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A16Z%2F-1%2F%2F8150b6bd976a30ec6f7b9c90a4d4e26dab678a1b768d5f2f76b5fc10d15d38d7" alt="Image" width="20%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图 7：迭代剪枝时 VGG-19 的测试精度（在 30K、60K、112K 次迭代处）。</div> </div>


这些改进表明，我们的迭代剪枝策略与 dropout 以互补的方式相互作用。Srivastava et al. (2014) 观察到 dropout 会在最终网络中诱导出稀疏激活；dropout 诱导的稀疏性可能为网络被剪枝做了准备。如果是这样，针对权重的 dropout 技术（Wan et al., 2013）或学习逐权重 dropout 概率的技术（Molchanov et al., 2017; Louizos et al., 2018）可能让中奖彩票更容易被找到。

## 4 CIFAR10 上的 VGG 与 ResNet（VGG and ResNet for CIFAR10）

在此，我们在让人联想到实际使用的架构与技术的网络上研究彩票假说。具体而言，我们考虑 VGG 风格的深层卷积网络（CIFAR10 上的 VGG-19——Simonyan & Zisserman (2014)）和残差网络（CIFAR10 上的 Resnet-18——He et al. (2016)）。 $^{5}$ 这些网络使用批归一化、权重衰减、递减学习率调度和增强的训练数据进行训练。我们在所有这些架构上仍能找到中奖彩票；然而，我们寻找中奖彩票的方法——迭代剪枝——对所用的特定学习率敏感。在这些实验中，我们不测量早停时间（对这些较大的网络而言，早停时间与学习率调度纠缠在一起），而是描绘训练过程中几个时刻的精度，以说明精度提升的相对速率。

全局剪枝。在 Lenet 和 Conv-2/4/6 上，我们以相同的速率分别剪枝每一层。对于 Resnet-18 和 VGG-19，我们略微修改了这一策略：我们全局地剪枝这些更深的网络，在所有卷积层中统一移除幅度最小的权重。在附录 I.1 中，我们发现全局剪枝能为 Resnet-18 和 VGG-19 识别出更小的中奖彩票。我们对这一行为的猜测性解释如下：对这些更深的网络而言，某些层的参数远多于其他层。例如，VGG-19 的前两个卷积层分别有 1728 和 36864 个参数，而最后一层有 235 万个。当所有层以相同速率剪枝时，这些较小的层会成为瓶颈，阻止我们识别出尽可能小的中奖彩票。全局剪枝使我们能够避免这一陷阱。

VGG-19。我们研究由 Liu et al. (2019) 为 CIFAR10 改编的 VGG-19 变体；我们使用相同的训练方案和超参数：160 个 epoch（112,480 次迭代），使用带

 $^{5}$关于网络、超参数和训练方案的细节见图 2 和附录 I。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//93e6c083-175f-4d78-b0df-211eb02fb071/markdown_3/imgs/img_in_chart_box_224_167_996_382.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A16Z%2F-1%2F%2F7e596d2f380d530e30cd44fb1bfb179a6355bf3c0c93884d9f3c0b9a2a4d996b" alt="Image" width="63%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图 8：迭代剪枝时 Resnet-18 的测试精度（在 10K、20K、30K 次迭代处）。</div> </div>


momentum（0.9）的 SGD，并在 80 和 120 个 epoch 时将学习率降低 10 倍。该网络有 2000 万个参数。图 7 展示了在两个初始学习率——0.1（Liu et al. (2019) 所用）和 0.01——下对 VGG-19 进行迭代剪枝与随机重新初始化的结果。在较高学习率下，迭代剪枝找不到中奖彩票，性能并不比随机重新初始化的剪枝网络更好。然而，在较低学习率下，通常的规律再次出现：当  $P_m \geq 3.5\%$ 时，子网络保持在原精度 1 个百分点以内。（它们不是中奖彩票，因为它们没有达到原精度。）当随机重新初始化时，子网络随剪枝以与本文其他实验相同的方式损失精度。虽然这些子网络在训练早期比未剪枝网络学得更快（图 7 左），但由于较低的初始学习率，这一精度优势在训练后期被侵蚀。然而，这些子网络仍比重初始化时学得更快。

为了弥合较低学习率下的彩票行为与较高学习率下的精度优势之间的差距，我们探索了在  $k$ 次迭代内学习率从 0 线性 warmup 到初始学习率的效果。在 0.1 学习率下使用 warmup（ $k = 10000$，绿线）训练 VGG-19，未剪枝网络的测试精度提高了约 1 个百分点。warmup 使找到中奖彩票成为可能，当  $P_m \geq 1.5\%$ 时超过这一初始精度。

Resnet-18。Resnet-18（He et al., 2016）是为 CIFAR10 设计的、带残差连接的 20 层卷积网络。它有 271,000 个参数。我们使用带 momentum（0.9）的 SGD 训练该网络 30,000 次迭代，在 20,000 和 25,000 次迭代时将学习率降低 10 倍。图 8 展示了在 0.1（He et al. (2016) 所用）和 0.01 学习率下迭代剪枝与随机重新初始化的结果。这些结果大体上反映了 VGG 的情况：迭代剪枝在较低学习率下能找到中奖彩票，而在较高学习率下不能。较低学习率下最好的中奖彩票的精度（当  $41.7\% \geq P_m \geq 21.9\%$ 时为 89.5%）低于原网络在较高学习率下的精度（90.5%）。在较低学习率下，中奖彩票同样最初学得更快（图 8 左图），但在训练后期落后于较高学习率下的未剪枝网络（右图）。使用 warmup 训练的中奖彩票缩小了与较高学习率下未剪枝网络的精度差距，在 0.03 学习率（warmup， $k = 20000$ ）和  $P_m = 27.1\%$ 下达到 90.5% 的测试精度。对这些超参数，当  $P_m \geq 11.8\%$ 时我们仍能找到中奖彩票。然而，即使使用 warmup，我们也无法找到在原始学习率 0.1 下能识别出中奖彩票的超参数。

## 5 讨论（Discussion）

关于神经网络剪枝的现有工作（如 Han et al. (2015)）表明，神经网络学到的函数通常可以用更少的参数来表示。剪枝通常先训练原网络，再移除连接并进一步微调。实际上，初始训练初始化了剪枝后网络的权重，使其能在微调期间独立学习。我们试图确定同样稀疏的网络是否能从头开始学习。我们发现本文研究的架构可靠地包含此类可训练子网络，而彩票假说提出这一性质具有普遍性。我们对中奖彩票存在性与本质的实证研究引出了一系列后续问题。

中奖彩票初始化的重要性。当随机重新初始化时，中奖彩票学得更慢、测试精度更低，这表明初始化对其成功很重要。对这一行为的一种可能解释是，这些初始权重接近其训练后的最终

值——在最极端的情况下，它们已经是训练好的。然而，附录 F 的实验表明恰恰相反——中奖彩票的权重比其他权重移动得更多。这表明初始化的好处与优化算法、数据集和模型有关。例如，中奖彩票的初始化可能落在特别利于所选优化算法进行优化的损失景观区域。

Liu et al. (2019) 发现剪枝后的网络在随机重新初始化后确实可以训练，这似乎与传统观点及我们的随机重新初始化实验相矛盾。例如，在 VGG-19 上（我们使用相同的设置），他们发现剪枝最多 80% 并随机重新初始化的网络能达到原网络的精度。我们在图 7 中的实验在这一稀疏度水平上确认了这些发现（Liu et al. 在更低稀疏度下未给出数据）。然而，进一步剪枝后，初始化就很重要了：当 VGG-19 被剪枝最多 98.5% 时我们能找到中奖彩票；而重新初始化后，这些彩票的精度要低得多。我们假设——在一定稀疏度水平内——高度过参数化的网络可以被剪枝、重新初始化并成功重训；然而，超过这一点，被极度剪枝、过参数化程度较轻的网络只有凭借幸运的初始化才能保持精度。

中奖彩票结构的重要性。产生中奖彩票的初始化被安排在一种特定的稀疏架构中。由于我们通过大量使用训练数据来发现中奖彩票，我们假设中奖彩票的结构编码了一种针对当前学习任务定制的归纳偏置。Cohen & Shashua (2016) 表明，嵌入深层网络结构中的归纳偏置决定了它能比浅层网络更参数高效地分离哪些类型的数据；虽然 Cohen & Shashua (2016) 关注卷积网络的池化几何，但中奖彩票的结构可能也有类似的作用，使它们即使在被大量剪枝时仍能学习。

中奖彩票泛化能力的改善。我们可靠地找到泛化能力更好的中奖彩票，它们在达到与原网络相同训练精度的同时超过其测试精度。随剪枝的进行，测试精度先升后降，形成一座"Occam 山丘"（Rasmussen & Ghahramani, 2001）：原始过参数化模型的复杂度过高（可能过拟合），而被极度剪枝的模型复杂度又过低。关于压缩与泛化之间关系的传统观点是，紧凑的假设能更好地泛化（Rissanen, 1986）。近期的理论工作表明神经网络也有类似的联系，为可被进一步压缩的网络证明了更紧的泛化界（Zhou et al. (2018) 针对剪枝/量化，Arora et al. (2018) 针对噪声鲁棒性）。彩票假说为这一关系提供了互补的视角——更大的网络可能明确地包含更简单的表示。

对神经网络优化的意义。中奖彩票能以显著更少的参数达到与原未剪枝网络相当的精度。这一观察与近期关于过参数化在神经网络训练中作用的工作相联系。例如，Du et al. (2019) 证明，足够过参数化的两层 relu 网络（第二层规模固定）用 SGD 训练能收敛到全局最优。那么，一个关键问题是：中奖彩票的存在对于 SGD 将神经网络优化到特定测试精度是必要条件还是充分条件。我们猜想（但未经验证）SGD 会寻找并训练一个初始化良好的子网络。按此逻辑，过参数化网络之所以更容易训练，是因为它们拥有更多可能成为中奖彩票的子网络组合。

## 6 局限与未来工作（Limitations and Future Work）

我们只考虑了较小数据集（MNIST、CIFAR10）上的视觉中心分类任务。我们没有研究更大的数据集（即 Imagenet（Russakovsky et al., 2015））：迭代剪枝计算量大，需要对网络连续训练 15 次或更多次以进行多次试验。在未来工作中，我们打算探索更高效的中奖彩票寻找方法，以便在更耗费资源的环境下研究彩票假说。

稀疏剪枝是我们寻找中奖彩票的唯一方法。虽然我们减少了参数量，但得到的架构并未针对现代库或硬件进行优化。在未来工作中，我们打算研究大量当代文献中的其他剪枝方法，例如结构化剪枝（将产生针对当代硬件优化的网络）和非幅度剪枝方法（可能产生更小的中奖彩票或更早找到它们）。

我们找到的中奖彩票具有这样的初始化：在随机初始化网络无法达到相同性能的规模下，它们仍能匹配未剪枝网络的性能。在未来工作中，我们打算研究这些初始化的性质——它们与被剪枝网络架构的归纳偏置共同作用，使这些网络特别擅长学习。

在更深的网络（Resnet-18 和 VGG-19）上，除非我们使用学习率 warmup 训练网络，否则迭代剪枝无法找到中奖彩票。在未来工作中，我们计划探索为什么需要 warmup，以及对我们识别中奖彩票方案的其他改进能否免除这些超参数修改的需要。

## 7 相关工作（Related Work）

在实践中，神经网络往往严重过参数化。蒸馏（Ba & Caruana, 2014; Hinton et al., 2015）和剪枝（LeCun et al., 1990; Han et al., 2015）依赖这样一个事实：参数可以在保持精度的同时被减少。即使有足够的能力记忆训练数据，网络也会自然地学习更简单的函数（Zhang et al., 2016; Neyshabur et al., 2014; Arpit et al., 2017）。当代经验（Bengio et al., 2006; Hinton et al., 2015; Zhang et al., 2016）和图 1 表明，过参数化的网络更容易训练。我们表明，稠密网络包含能够从原始初始化开始独立学习的稀疏子网络。还有其他几个研究方向旨在训练小型或稀疏网络。

训练之前。Squeezenet（Iandola et al., 2016）和 MobileNets（Howard et al., 2017）是专门设计的图像识别网络，比标准架构小一个数量级。Denil et al. (2013) 将权重矩阵表示为低秩因子的乘积。Li et al. (2018) 将优化限制在参数空间中一个小的随机采样子空间内（意味着所有参数仍可更新）；他们在此限制下成功训练了网络。我们表明，优化一个网络甚至无需更新所有参数，我们通过一个包含剪枝的原则性搜索过程找到中奖彩票。我们对这类方法的贡献在于证明：稀疏、可训练的网络存在于更大的网络内部。

训练之后。蒸馏（Ba & Caruana, 2014; Hinton et al., 2015）训练小网络来模仿大网络的行为；在这一范式中，小网络更容易训练。近期的剪枝工作压缩大模型以便在有限资源下运行（例如在移动设备上）。虽然剪枝是我们实验的核心，但我们研究的是：为什么训练需要这些使剪枝成为可能的过参数化网络。LeCun et al. (1990) 和 Hassibi & Stork (1993) 最早探索了基于二阶导数的剪枝。最近，Han et al. (2015) 表明基于逐权重幅度的剪枝能大幅缩小图像识别网络的规模。Guo et al. (2016) 在被剪掉的连接重新变得相关时恢复它们。Han et al. (2017) 和 Jin et al. (2016) 在小权重被剪掉、存活权重被微调之后恢复被剪掉的连接以增加网络容量。其他被提出的剪枝启发式包括基于激活的剪枝（Hu et al., 2016）、基于冗余的剪枝（Mariet & Sra, 2016; Srinivas & Babu, 2015a）、逐层二阶导数剪枝（Dong et al., 2017）以及基于能量/计算效率的剪枝（Yang et al., 2017）（例如剪枝卷积滤波器（Li et al., 2016; Molchanov et al., 2016; Luo et al., 2017）或通道（He et al., 2017））。Cohen et al. (2016) 观察到卷积滤波器对初始化敏感（"滤波器彩票"，The Filter Lottery）；在整个训练过程中，他们随机重新初始化不重要的滤波器。

训练过程中。Bellec et al. (2018) 用稀疏网络训练，并把达到零的权重重置为新的随机连接。Srinivas et al. (2017) 和 Louizos et al. (2018) 学习最小化非零参数数量的门控变量。Narang et al. (2017) 将基于幅度的剪枝整合进训练。Gal & Ghahramani (2016) 表明 dropout 近似于高斯过程中的贝叶斯推断。关于 dropout 的贝叶斯视角在训练过程中学习 dropout 概率（Gal et al., 2017; Kingma et al., 2015; Srinivas & Babu, 2016）。学习逐权重、逐单元（Srinivas & Babu, 2016）或结构化 dropout 概率的技术，随着某些权重的 dropout 概率达到 1，会在训练过程中自然地（Molchanov et al., 2017; Neklyudov et al., 2017）或显式地（Louizos et al., 2017; Srinivas & Babu, 2015b）剪枝并稀疏化网络。相比之下，我们至少训练网络一次以寻找中奖彩票。这些技术也可能找到中奖彩票，或者通过诱导稀疏性，可能对我们的方法产生有益的作用。

## REFERENCES

Sanjeev Arora, Rong Ge, Behnam Neyshabur, and Yi Zhang. Stronger generalization bounds for deep nets via a compression approach. ICML, 2018.

Devansh Arpit, Stanisław Jastrzębski, Nicolas Ballas, David Krueger, Emmanuel Bengio, Maxinder S Kanwal, Tegan Maharaj, Asja Fischer, Aaron Courville, Yoshua Bengio, et al. A closer look at memorization in deep networks. In International Conference on Machine Learning, pp. 233–242, 2017.

Jimmy Ba and Rich Caruana. Do deep nets really need to be deep? In Advances in neural information processing systems, pp. 2654–2662, 2014.

Pierre Baldi and Peter J Sadowski. Understanding dropout. In Advances in neural information processing systems, pp. 2814–2822, 2013.

Guillaume Bellec, David Kappel, Wolfgang Maass, and Robert Legenstein. Deep rewiring: Training very sparse deep networks. Proceedings of ICLR, 2018.

Yoshua Bengio, Nicolas L Roux, Pascal Vincent, Olivier Delalleau, and Patrice Marcotte. Convex neural networks. In Advances in neural information processing systems, pp. 123–130, 2006.

Joseph Paul Cohen, Henry Z Lo, and Wei Ding. Randomout: Using a convolutional gradient norm to win the filter lottery. ICLR Workshop, 2016.

Nadav Cohen and Amnon Shashua. Inductive bias of deep convolutional networks through pooling geometry. arXiv preprint arXiv:1605.06743, 2016.

Misha Denil, Babak Shakibi, Laurent Dinh, Nando De Freitas, et al. Predicting parameters in deep learning. In Advances in neural information processing systems, pp. 2148–2156, 2013.

Xin Dong, Shangyu Chen, and Sinno Pan. Learning to prune deep neural networks via layer-wise optimal brain surgeon. In Advances in Neural Information Processing Systems, pp. 4860–4874, 2017.

Simon S. Du, Xiyu Zhai, Barnabas Poczos, and Aarti Singh. Gradient descent provably optimizes over-parameterized neural networks. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=S1eK3i09YQ.

Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning, pp. 1050–1059, 2016.

Yarin Gal, Jiri Hron, and Alex Kendall. Concrete dropout. In Advances in Neural Information Processing Systems, pp. 3584–3593, 2017.

Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the thirteenth international conference on artificial intelligence and statistics, pp. 249–256, 2010.

Yiwen Guo, Anbang Yao, and Yurong Chen. Dynamic network surgery for efficient dnns. In Advances In Neural Information Processing Systems, pp. 1379–1387, 2016.

Song Han, Jeff Pool, John Tran, and William Dally. Learning both weights and connections for efficient neural network. In Advances in neural information processing systems, pp. 1135–1143, 2015.

Song Han, Jeff Pool, Sharan Narang, Huizi Mao, Shijian Tang, Erich Elsen, Bryan Catanzaro, John Tran, and William J Dally. Dsd: Regularizing deep neural networks with dense-sparse-dense training flow. Proceedings of ICLR, 2017.

Babak Hassibi and David G Stork. Second order derivatives for network pruning: Optimal brain surgeon. In Advances in neural information processing systems, pp. 164–171, 1993.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016.

Yihui He, Xiangyu Zhang, and Jian Sun. Channel pruning for accelerating very deep neural networks. In International Conference on Computer Vision (ICCV), volume 2, pp. 6, 2017.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

Geoffrey E Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever, and Ruslan R Salakhutdinov. Improving neural networks by preventing co-adaptation of feature detectors. arXiv preprint arXiv:1207.0580, 2012.

Andrew G Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861, 2017.

Hengyuan Hu, Rui Peng, Yu-Wing Tai, and Chi-Keung Tang. Network trimming: A data-driven neuron pruning approach towards efficient deep architectures. arXiv preprint arXiv:1607.03250, 2016.

Forrest N Iandola, Song Han, Matthew W Moskewicz, Khalid Ashraf, William J Dally, and Kurt Keutzer. Squeezenet: Alexnet-level accuracy with 50x fewer parameters and &lt; 0.5 mb model size. arXiv preprint arXiv:1602.07360, 2016.

Xiaojie Jin, Xiaotong Yuan, Jiashi Feng, and Shuicheng Yan. Training skinny deep neural networks with iterative hard thresholding methods. arXiv preprint arXiv:1607.05423, 2016.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Diederik P Kingma, Tim Salimans, and Max Welling. Variational dropout and the local reparameterization trick. In Advances in Neural Information Processing Systems, pp. 2575–2583, 2015.

Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. 2009

Yann LeCun, John S Denker, and Sara A Solla. Optimal brain damage. In Advances in neural information processing systems, pp. 598–605, 1990.

Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.

Chunyuan Li, Heerad Farkhoor, Rosanne Liu, and Jason Yosinski. Measuring the intrinsic dimension of objective landscapes. Proceedings of ICLR, 2018.

Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, and Hans Peter Graf. Pruning filters for efficient convnets. arXiv preprint arXiv:1608.08710, 2016.

Zhuang Liu, Mingjie Sun, Tinghui Zhou, Gao Huang, and Trevor Darrell. Rethinking the value of network pruning. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=rJlnB3C5Ym.

Christos Louizos, Karen Ullrich, and Max Welling. Bayesian compression for deep learning. In Advances in Neural Information Processing Systems, pp. 3290–3300, 2017.

Christos Louizos, Max Welling, and Diederik P Kingma. Learning sparse neural networks through  $l_{0}$ regularization. Proceedings of ICLR, 2018.

Jian-Hao Luo, Jianxin Wu, and Weiyao Lin. Thinet: A filter level pruning method for deep neural network compression. arXiv preprint arXiv:1707.06342, 2017.

Zelda Mariet and Suvrit Sra. Diversity networks. Proceedings of ICLR, 2016.

Dmitry Molchanov, Arsenii Ashukha, and Dmitry Vetrov. Variational dropout sparsifies deep neural networks. arXiv preprint arXiv:1701.05369, 2017.

Pavlo Molchanov, Stephen Tyree, Tero Karras, Timo Aila, and Jan Kautz. Pruning convolutional neural networks for resource efficient transfer learning. arXiv preprint arXiv:1611.06440, 2016.

Sharan Narang, Erich Elsen, Gregory Diamos, and Shubho Sengupta. Exploring sparsity in recurrent neural networks. Proceedings of ICLR, 2017.

Kirill Neklyudov, Dmitry Molchanov, Arsenii Ashukha, and Dmitry P Vetrov. Structured bayesian pruning via log-normal multiplicative noise. In Advances in Neural Information Processing Systems, pp. 6778–6787, 2017.

Behnam Neyshabur, Ryota Tomioka, and Nathan Srebro. In search of the real inductive bias: On the role of implicit regularization in deep learning. arXiv preprint arXiv:1412.6614, 2014.

Carl Edward Rasmussen and Zoubin Ghahramani. Occam's razor. In T. K. Leen, T. G. Dietterich, and V. Tresp (eds.), Advances in Neural Information Processing Systems 13, pp. 294–300. MIT Press, 2001. URL http://papers.nips.cc/paper/1925-occams-razor.pdf.

Jorma Rissanen. Stochastic complexity and modeling. The annals of statistics, pp. 1080–1100, 1986.

Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International Journal of Computer Vision, 115(3):211–252, 2015.

Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.

Suraj Srinivas and R Venkatesh Babu. Data-free parameter pruning for deep neural networks. arXiv preprint arXiv:1507.06149, 2015a.

Suraj Srinivas and R Venkatesh Babu. Learning neural network architectures using backpropagation. arXiv preprint arXiv:1511.05497, 2015b.

Suraj Srinivas and R Venkatesh Babu. Generalized dropout. arXiv preprint arXiv:1611.06791, 2016.

Suraj Srinivas, Akshayvarun Subramanya, and R Venkatesh Babu. Training sparse neural networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pp. 138–145, 2017.

Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: A simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research, 15(1):1929–1958, 2014.

Li Wan, Matthew Zeiler, Sixin Zhang, Yann Le Cun, and Rob Fergus. Regularization of neural networks using dropconnect. In International Conference on Machine Learning, pp. 1058–1066, 2013.

Tien-Ju Yang, Yu-Hsin Chen, and Vivienne Sze. Designing energy-efficient convolutional neural networks using energy-aware pruning. arXiv preprint, 2017.

Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization. arXiv preprint arXiv:1611.03530, 2016.

Wenda Zhou, Victor Veitch, Morgane Austern, Ryan P Adams, and Peter Orbanz. Compressibility and generalization in large-scale deep learning. arXiv preprint arXiv:1804.05862, 2018.

### A ACKNOWLEDGMENTS

We gratefully acknowledge IBM, which—through the MIT-IBM Watson AI Lab—contributed the computational resources necessary to conduct the experiments in this paper. We particularly thank IBM researchers German Goldszmidt, David Cox, Ian Molloy, and Benjamin Edwards for their generous contributions of infrastructure, technical support, and feedback. We also wish to thank Aleksander Madry, Shafi Goldwasser, Ed Felten, David Bieber, Karolina Dziugaite, Daniel Weitzner, and R. David Edelman for support, feedback, and helpful discussions over the course of this project. This work was supported in part by the Office of Naval Research (ONR N00014-17-1-2699).

### B ITERATIVE PRUNING STRATEGIES

In this Appendix, we examine two different ways of structuring the iterative pruning strategy that we use throughout the main body of the paper to find winning tickets.

#### Strategy 1: Iterative pruning with resetting.

1. Randomly initialize a neural network  $f(x; m \odot \theta)$ where  $\theta = \theta_0$ and  $m = 1^{|\theta|}$ is a mask.

2. Train the network for  $j$ iterations, reaching parameters  $m \odot \theta_j$.

3. Prune s% of the parameters, creating an updated mask  $m'$ where  $P_{m'} = (P_m - s)\%$.

4. Reset the weights of the remaining portion of the network to their values in  $\theta_{0}$. That is, let  $\theta = \theta_{0}$.

5. Let  $m = m'$ and repeat steps 2 through 4 until a sufficiently pruned network has been obtained.

### Strategy 2: Iterative pruning with continued training.

1. Randomly initialize a neural network  $f(x; m \odot \theta)$ where  $\theta = \theta_0$ and  $m = 1^{|\theta|}$

2. Train the network for j iterations.

3. Prune s% of the parameters, creating an updated mask  $m'$ where  $P_{m'} = (P_m - s)\%$.

4. Let  $m = m'$ and repeat steps 2 and 3 until a sufficiently pruned network has been obtained.

5. Reset the weights of the remaining portion of the network to their values in  $\theta_{0}$. That is, let  $\theta = \theta_{0}$.

The difference between these two strategies is that, after each round of pruning, Strategy 2 retrains using the already-trained weights, whereas Strategy 1 resets the network weights back to their initial values before retraining. In both cases, after the network has been sufficiently pruned, its weights are reset back to the original initialization.

Figures 9 and 10 compare the two strategies on the Lenet and Conv-2/4/6 architectures on the hyperparameters we select in Appendices G and H. In all cases, the Strategy 1 maintains higher validation accuracy and faster early-stopping times to smaller network sizes.

### C EARLY STOPPING CRITERION

Throughout this paper, we are interested in measuring the speed at which networks learn. As a proxy for this quantity, we measure the iteration at which an early-stopping criterion would end training. The specific criterion we employ is the iteration of minimum validation loss. In this Subsection, we further explain that criterion.

Validation and test loss follow a pattern where they decrease early in the training process, reach a minimum, and then begin to increase as the model overfits to the training data. Figure 11 shows an example of the validation loss as training progresses; these graphs use Lenet, iterative pruning, and Adam with a learning rate of 0.0012 (the learning rate we will select in the following subsection). This Figure shows the validation loss corresponding to the test accuracies in Figure 3.

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//609b08c5-1b30-4116-a454-f704005ecebd/markdown_2/imgs/img_in_chart_box_219_201_1000_445.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A18Z%2F-1%2F%2F11014a18bd814ace2856c68cd68b590757e7db91a0e78ec364a9f0b0a327fa53" alt="Image" width="63%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 9: The early-stopping iteration and accuracy at early-stopping of the iterative lottery ticket experiment on the Lenet architecture when iteratively pruned using the resetting and continued training strategies.</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//609b08c5-1b30-4116-a454-f704005ecebd/markdown_2/imgs/img_in_chart_box_217_598_1000_840.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A19Z%2F-1%2F%2Fbc1e0b597db81aa1990e03d5cef6313c103267fc9f8cf12d71389a99756ca2ac" alt="Image" width="63%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 10: The early-stopping iteration and accuracy at early-stopping of the iterative lottery ticket experiment on the Conv-2, Conv-4, and Conv-6 architectures when iteratively pruned using the resetting and continued training strategies.</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//609b08c5-1b30-4116-a454-f704005ecebd/markdown_2/imgs/img_in_chart_box_233_996_986_1239.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A19Z%2F-1%2F%2F3a0f4540581b9726c69235509b09cadd3a6b1d03cacf0cbf49738069c57850c1" alt="Image" width="61%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 11: The validation loss data corresponding to Figure 3, i.e., the validation loss as training progresses for several different levels of pruning in the iterative pruning experiment. Each line is the average of five training runs at the same level of iterative pruning; the labels are the percentage of weights from the original network that remain after pruning. Each network was trained with Adam at a learning rate of 0.0012. The left graph shows winning tickets that learn increasingly faster than the original network and reach lower loss. The middle graph shows winning tickets that learn increasingly slower after the fastest early-stopping time has been reached. The right graph contrasts the loss of winning tickets to the loss of randomly reinitialized networks.</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//609b08c5-1b30-4116-a454-f704005ecebd/markdown_3/imgs/img_in_chart_box_219_174_998_579.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A21Z%2F-1%2F%2F0b321f0a2c407f0ee3f81f78918ee6e7be23994bdebaf8a3da08b437f908dac8" alt="Image" width="63%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 12: Figure 4 augmented with a graph of the training accuracy at the end of 50,000 iterations.</div> </div>


In all cases, validation loss initially drops, after which it forms a clear bottom and then begins increasing again. Our early-stopping criterion identifies this bottom. We consider networks that reach this moment sooner to have learned “faster.” In support of this notion, the ordering in which each experiment meets our early-stopping criterion in Figure 3 is the same order in which each experiment reaches a particular test accuracy threshold in Figure 3.

Throughout this paper, in order to contextualize this learning speed, we also present the test accuracy of the network at the iteration of minimum validation loss. In the main body of the paper, we find that winning tickets both arrive at early-stopping sooner and reach higher test accuracy at this point.

### D TRAINING ACCURACY FOR LOTTERY TICKET EXPERIMENTS

This Appendix accompanies Figure 4 (the accuracy and early-stopping iterations of Lenet on MNIST from Section 2) and Figure 5 (the accuracy and early-stopping iterations of Conv-2, Conv-4, and Conv-6 in Section Section 3) in the main body of the paper. Those figures show the iteration of early-stopping, the test accuracy at early-stopping, the training accuracy at early-stopping, and the test accuracy at the end of the training process. However, we did not have space to include a graph of the training accuracy at the end of the training process, which we assert in the main body of the paper to be 100% for all but the most heavily pruned networks. In this Appendix, we include those additional graphs in Figure 12 (corresponding to Figure 4) and Figure 13 (corresponding to Figure 5). As we describe in the main body of the paper, training accuracy reaches 100% in all cases for all but the most heavily pruned networks. However, training accuracy remains at 100% longer for winning tickets than for randomly reinitialized networks.

### E COMPARING RANDOM REINITIALIZATION AND RANDOM SPARSITY

In this Appendix, we aim to understand the relative performance of randomly reinitialized winning tickets and randomly sparse networks.

1. Networks found via iterative pruning with the original initializations (blue in Figure 14).

2. Networks found via iterative pruning that are randomly reinitialized (orange in Figure 14).

3. Random sparse subnetworks with the same number of parameters as those found via iterative pruning (green in Figure 14).

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e927b777-e3eb-4bad-9bc6-1bf5cf41cfd6/markdown_0/imgs/img_in_chart_box_216_486_1002_1106.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A14Z%2F-1%2F%2F8b68751807803c5653de256dd77081c9c64138bcf0143d15aab5c8067747a9af" alt="Image" width="64%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 13: Figure 5 augmented with a graph of the training accuracy at the end of the training process.</div> </div>


Figure 14 shows this comparison for all of the major experiments in this paper. For the fully-connected Lenet architecture for MNIST, we find that the randomly reinitialized networks outperform random sparsity. However, for all of the other, convolutional networks studied in this paper, there is no significant difference in performance between the two. We hypothesize that the fully-connected network for MNIST sees these benefits because only certain parts of the MNIST images contain useful information for classification, meaning connections in some parts of the network will be more valuable than others. This is less true with convolutions, which are not constrained to any one part of the input image.

### F EXAMINING WINNING TICKETS

In this Appendix, we examine the structure of winning tickets to gain insight into why winning tickets are able to learn effectively even when so heavily pruned. Throughout this Appendix, we study the winning tickets from the Lenet architecture trained on MNIST. Unless otherwise stated, we use the same hyperparameters as in Section 2: glorot initialization and adam optimization.

### F.1 WINNING TICKET INITIALIZATION (ADAM)

Figure 15 shows the distributions of winning ticket initializations for four different levels of  $P_{m}$. To clarify, these are the distributions of the initial weights of the connections that have survived the pruning process. The blue, orange, and green lines show the distribution of weights for the first hidden layer, second hidden layer, and output layer, respectively. The weights are collected from five different trials of the lottery ticket experiment, but the distributions for each individual trial closely mirror those aggregated from across all of the trials. The histograms have been normalized so that the area under each curve is 1.

The left-most graph in Figure 15 shows the initialization distributions for the unpruned networks. We use glorot initialization, so each of the layers has a different standard deviation. As the network is pruned, the first hidden layer maintains its distribution. However, the second hidden layer and the output layer become increasingly bimodal, with peaks on either side of 0. Interestingly, the peaks are asymmetric: the second hidden layer has more positive initializations remaining than negative initializations, and the reverse is true for the output layer.

The connections in the second hidden layer and output layer that survive the pruning process tend to have higher magnitude-initializations. Since we find winning tickets by pruning the connections with the lowest magnitudes in each layer at the end, the connections with the lowest-magnitude initializations must still have the lowest-magnitude weights at the end of training. A different trend holds for the input layer: it maintains its distribution, meaning a connection's initialization has less relation to its final weight.

### F.2 WINNING TICKET INITIALIZATIONS (SGD)

We also consider the winning tickets obtained when training the network with SGD learning rate 0.8 (selected as described in Appendix G). The bimodal distributions from Figure 15 are present across all layers (see Figure 16). The connections with the highest-magnitude initializations are more likely to survive the pruning process, meaning winning ticket initializations have a bimodal distribution with peaks on opposite sides of 0. Just as with the adam-optimized winning tickets, these peaks are of different sizes, with the first hidden layer favoring negative initializations and the second hidden layer and output layer favoring positive initializations. Just as with the adam results, we confirm that each individual trial evidences the same asymmetry as the aggregate graphs in Figure 16.

### F.3 REINITIALIZING FROM WINNING TICKET INITIALIZATIONS

Considering that the initialization distributions of winning tickets  $\mathcal{D}_m$ are so different from the Gaussian distribution  $\mathcal{D}$ used to initialize the unpruned network, it is natural to ask whether randomly reinitializing winning tickets from  $\mathcal{D}_m$ rather than  $\mathcal{D}$ will improve winning ticket performance. We do not find this to be the case. Figure 17 shows the performance of winning tickets whose initializations are randomly sampled from the distribution of initializations contained in the winning tickets for

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e927b777-e3eb-4bad-9bc6-1bf5cf41cfd6/markdown_2/imgs/img_in_image_box_219_282_1001_1308.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A14Z%2F-1%2F%2F8b95501c58f0fa6a87fd33bec1157016983fdb565a71f957a3ec02c46b392f6a" alt="Image" width="63%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 14: The test accuracy at the final iteration for each of the networks studied in this paper.</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e927b777-e3eb-4bad-9bc6-1bf5cf41cfd6/markdown_3/imgs/img_in_chart_box_218_163_1000_371.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A14Z%2F-1%2F%2F47344e8d67ff7aec35c76d5dd20a52c06a5668509759a0cfff9e00c5fb456559" alt="Image" width="63%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 15: The distribution of initializations in winning tickets pruned to the levels specified in the titles of each plot. The blue, orange, and green lines show the distributions for the first hidden layer, second hidden layer, and output layer of the Lenet architecture for MNIST when trained with the adam optimizer and the hyperparameters used in 2. The distributions have been normalized so that the area under each curve is 1.</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e927b777-e3eb-4bad-9bc6-1bf5cf41cfd6/markdown_3/imgs/img_in_chart_box_218_529_1001_732.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A15Z%2F-1%2F%2F0bbf69a3eb36363dc7ab33df09c05a70e6f587375aff0a4ff2e60e5c05f69baf" alt="Image" width="63%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 16: Same as Figure 15 where the network is trained with SGD at rate 0.8.</div> </div>


adam. More concretely, let  $\mathcal{D}_m = \{\theta_0^{(i)} | m^{(i)} = 1\}$ be the set of initializations found in the winning ticket with mask  $m$. We sample a new set of parameters  $\theta'_0 \sim \mathcal{D}_m$ and train the network  $f(x; m \odot \theta'_0)$. We perform this sampling on a per-layer basis. The results of this experiment are in Figure 17. Winning tickets reinitialized from  $\mathcal{D}_m$ perform little better than when randomly reinitialized from  $\mathcal{D}$. We attempted the same experiment with the SGD-trained winning tickets and found similar results.

### F.4 PRUNING AT ITERATION 0

One other way of interpreting the graphs of winning ticket initialization distributions is as follows: weights that begin small stay small, get pruned, and never become part of the winning ticket. (The only exception to this characterization is the first hidden layer for the adam-trained winning tickets.) If this is the case, then perhaps low-magnitude weights were never important to the network and can be pruned from the very beginning. Figure 18 shows the result of attempting this pruning strategy. Winning tickets selected in this fashion perform even worse than when they are found by iterative

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e927b777-e3eb-4bad-9bc6-1bf5cf41cfd6/markdown_3/imgs/img_in_chart_box_217_1183_1001_1372.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A15Z%2F-1%2F%2F3e3997ed12e07842d208f5cc3f8683a1a1a2a741cee0a4071d3b9cda758fc068" alt="Image" width="64%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 17: The performance of the winning tickets of the Lenet architecture for MNIST when the layers are randomly reinitialized from the distribution of initializations contained in the winning ticket of the corresponding size.</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//333fd7bc-56e9-4b93-9368-117b95b4ebf1/markdown_0/imgs/img_in_chart_box_218_180_1000_363.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A14Z%2F-1%2F%2F468cf0331bca06d64e9996f1003c3770c909ffdbd04b4b3bb45122fc7291c58d" alt="Image" width="63%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 18: The performance of the winning tickets of the Lenet architecture for MNIST when magnitude pruning is performed before the network is ever trained. The network is subsequently trained with adam.</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//333fd7bc-56e9-4b93-9368-117b95b4ebf1/markdown_0/imgs/img_in_chart_box_316_491_902_722.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A14Z%2F-1%2F%2Fb4209f860bdcd3b04ce31cf5a6a6c2196170cffc7343e1c77f8e71def06e4627" alt="Image" width="47%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 19: Between the first and last training iteration of the unpruned network, the magnitude by which weights in the network change. The blue line shows the distribution of magnitudes for weights that are not in the eventual winning ticket; the orange line shows the distribution of magnitudes for weights that are in the eventual winning ticket.</div> </div>


pruning and randomly reinitialized. We attempted the same experiment with the SGD-trained winning tickets and found similar results.

### F.5 COMPARING INITIAL AND FINAL WEIGHTS IN WINNING TICKETS

In this subsection, we consider winning tickets in the context of the larger optimization process. To do so, we examine the initial and final weights of the unpruned network from which a winning ticket derives to determine whether weights that will eventually comprise a winning ticket exhibit properties that distinguish them from the rest of the network.

We consider the magnitude of the difference between initial and final weights. One possible rationale for the success of winning tickets is that they already happen to be close to the optimum that gradient descent eventually finds, meaning that winning ticket weights should change by a smaller amount than the rest of the network. Another possible rationale is that winning tickets are well placed in the optimization landscape for gradient descent to optimize productively, meaning that winning ticket weights should change by a larger amount than the rest of the network. Figure 19 shows that winning ticket weights tend to change by a larger amount than weights in the rest of the network, evidence that does not support the rationale that winning tickets are already close to the optimum.

It is notable that such a distinction exists between the two distributions. One possible explanation for this distinction is that the notion of a winning ticket may indeed be a natural part of neural network optimization. Another is that magnitude-pruning biases the winning tickets we find toward those containing weights that change in the direction of higher magnitude. Regardless, it offers hope that winning tickets may be discernible earlier in the training process (or after a single training run), meaning that there may be more efficient methods for finding winning tickets than iterative pruning.

Figure 20 shows the directions of these changes. It plots the difference between the magnitude of the final weight and the magnitude of the initial weight, i.e., whether the weight moved toward or away.

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//333fd7bc-56e9-4b93-9368-117b95b4ebf1/markdown_1/imgs/img_in_chart_box_315_181_904_407.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A14Z%2F-1%2F%2F2836aede5d63026d7939e6a527f3a66d476b4e7672b3a8c3a3517f77c7f23f24" alt="Image" width="48%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 20: Between the first and last training iteration of the unpruned network, the magnitude by which weights move away from 0. The blue line shows the distribution of magnitudes for weights that are not in the eventual winning ticket; the orange line shows the distribution of magnitudes for weights that are in the eventual winning ticket.</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//333fd7bc-56e9-4b93-9368-117b95b4ebf1/markdown_1/imgs/img_in_chart_box_316_551_903_750.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A14Z%2F-1%2F%2F8c60f67afa45bcebd637d9e6e54420b3882f293f32aab72830ca4c742f50fbe6" alt="Image" width="47%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 21: The fraction of incoming connections that survive the pruning process for each node in each layer of the Lenet architecture for MNIST as trained with adam.</div> </div>


from 0. In general, winning ticket weights are more likely to increase in magnitude (that is, move away from 0) than are weights that do not participate in the eventual winning ticket.

### F.6 WINNING TICKET CONNECTIVITY

In this Subsection, we study the connectivity of winning tickets. Do some hidden units retain a large number of incoming connections while others fade away, or does the network retain relatively even sparsity among all units as it is pruned? We find the latter to be the case when examining the incoming connectivity of network units: for both adam and SGD, each unit retains a number of incoming connections approximately in proportion to the amount by which the overall layer has been pruned. Figures 21 and 22 show the fraction of incoming connections that survive the pruning process for each node in each layer. Recall that we prune the output layer at half the rate as the rest of the network, which explains why it has more connectivity than the other layers of the network.

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//333fd7bc-56e9-4b93-9368-117b95b4ebf1/markdown_1/imgs/img_in_chart_box_317_1213_903_1414.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A14Z%2F-1%2F%2Fb7cc4e8d0445932a0c57b1daedf5c1e8864d58c466fee5801fda3858ada5aa8f" alt="Image" width="47%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 22: Same as Figure 21 where the network is trained with SGD at rate 0.8.</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//333fd7bc-56e9-4b93-9368-117b95b4ebf1/markdown_2/imgs/img_in_chart_box_318_166_507_368.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A15Z%2F-1%2F%2Fdfdaebf5d552356234d257984259229e11fe14c2cca8cd94035bac71375e602f" alt="Image" width="15%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//333fd7bc-56e9-4b93-9368-117b95b4ebf1/markdown_2/imgs/img_in_chart_box_515_170_706_368.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A15Z%2F-1%2F%2F0254579b332fb6937962fbb451d06f3a6cd12f60b00f0b2e283c923fb46cc5cf" alt="Image" width="15%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//333fd7bc-56e9-4b93-9368-117b95b4ebf1/markdown_2/imgs/img_in_chart_box_714_170_901_368.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A15Z%2F-1%2F%2F05f967cdc4286c711559d8e6f4f6c2b409ce39c9ae512b05e33ae435155c8614" alt="Image" width="15%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 23: The fraction of outgoing connections that survive the pruning process for each node in each layer of the Lenet architecture for MNIST as trained with adam. The blue, orange, and green lines are the outgoing connections from the input layer, first hidden layer, and second hidden layer, respectively.</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//333fd7bc-56e9-4b93-9368-117b95b4ebf1/markdown_2/imgs/img_in_chart_box_318_538_506_739.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A15Z%2F-1%2F%2Fb957dfd9e4b0708e06ca11ee2a0216830293a7a0af6fbd3b96c5231499d4f03b" alt="Image" width="15%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//333fd7bc-56e9-4b93-9368-117b95b4ebf1/markdown_2/imgs/img_in_chart_box_515_545_705_740.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A15Z%2F-1%2F%2Ff73887f71f8c32478c480441f7fdbb21dd0a41450ebf659cb12950c9fe89ebc6" alt="Image" width="15%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//333fd7bc-56e9-4b93-9368-117b95b4ebf1/markdown_2/imgs/img_in_chart_box_714_543_902_739.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A15Z%2F-1%2F%2F5f470e3908221ef87b1e4575dc1f0fc9345a3b067167307d639332caca560e7a" alt="Image" width="15%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 24: Same as Figure 23 where the network is trained with SGD at rate 0.8.</div> </div>


However, this is not the case for the outgoing connections. To the contrary, for the adam-trained networks, certain units retain far more outgoing connections than others (Figure 23). The distributions are far less smooth than those for the incoming connections, suggesting that certain features are far more useful to the network than others. This is not unexpected for a fully-connected network on a task like MNIST, particularly for the input layer: MNIST images contain centered digits, so the pixels around the edges are not likely to be informative for the network. Indeed, the input layer has two peaks, one larger peak for input units with a high number of outgoing connections and one smaller peak for input units with a low number of outgoing connections. Interestingly, the adam-trained winning tickets develop a much more uneven distribution of outgoing connectivity for the input layer than does the SGD-trained network (Figure 24).

### F.7 ADDING NOISE TO WINNING TICKETS

In this Subsection, we explore the extent to which winning tickets are robust to Gaussian noise added to their initializations. In the main body of the paper, we find that randomly reinitializing a winning ticket substantially slows its learning and reduces its eventual test accuracy. In this Subsection, we study a less extreme way of perturbing a winning ticket. Figure 25 shows the effect of adding Gaussian noise to the winning ticket initializations. The standard deviation of the noise distribution of each layer is a multiple of the standard deviation of the layer's initialization Figure 25 shows noise distributions with standard deviation  $0.5\sigma$,  $\sigma$,  $2\sigma$, and  $3\sigma$. Adding Gaussian noise reduces the test accuracy of a winning ticket and slows its ability to learn, again demonstrating the importance of the original initialization. As more noise is added, accuracy decreases. However, winning tickets are surprisingly robust to noise. Adding noise of  $0.5\sigma$ barely changes winning ticket accuracy. Even after adding noise of  $3\sigma$, the winning tickets continue to outperform the random reinitialization experiment.

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//333fd7bc-56e9-4b93-9368-117b95b4ebf1/markdown_3/imgs/img_in_chart_box_221_179_999_361.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A15Z%2F-1%2F%2Fd34f96f29caf207f666dd58193dfd6c45475db1ad9da16aec9206dc29347b50d" alt="Image" width="63%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 25: The performance of the winning tickets of the Lenet architecture for MNIST when Gaussian noise is added to the initializations. The standard deviations of the noise distributions for each layer are a multiple of the standard deviations of the initialization distributions; in this Figure, we consider multiples 0.5, 1, 2, and 3.</div> </div>


### G HYPERPARAMETER EXPLORATION FOR FULLY-CONNECTED NETWORKS

This Appendix accompanies Section 2 of the main paper. It explores the space of hyperparameters for the Lenet architecture evaluated in Section 2 with two purposes in mind:

1. To explain the hyperparameters selected in the main body of the paper.

2. To evaluate the extent to which the lottery ticket experiment patterns extend to other choices of hyperparameters.

### G.1 EXPERIMENTAL METHODOLOGY

This Section considers the fully-connected Lenet architecture (LeCun et al., 1998), which comprises two fully-connected hidden layers and a ten unit output layer, on the MNIST dataset. Unless otherwise stated, the hidden layers have 300 and 100 units each.

The MNIST dataset consists of 60,000 training examples and 10,000 test examples. We randomly sampled a 5,000-example validation set from the training set and used the remaining 55,000 training examples as our training set for the rest of the paper (including Section 2). The hyperparameter selection experiments throughout this Appendix are evaluated using the validation set for determining both the iteration of early-stopping and the accuracy at early-stopping; the networks in the main body of this paper (which make use of these hyperparameters) have their accuracy evaluated on the test set. The training set is presented to the network in mini-batches of 60 examples; at each epoch, the entire training set is shuffled.

Unless otherwise noted, each line in each graph comprises data from three separate experiments. The line itself traces the average performance of the experiments and the error bars indicate the minimum and maximum performance of any one experiment.

Throughout this Appendix, we perform the lottery ticket experiment iteratively with a pruning rate of 20% per iteration (10% for the output layer); we justify the choice of this pruning rate later in this Appendix. Each layer of the network is pruned independently. On each iteration of the lottery ticket experiment, the network is trained for 50,000 training iterations regardless of when early-stopping occurs; in other words, no validation or test data is taken into account during the training process, and early-stopping times are determined retroactively by examining validation performance. We evaluate validation and test performance every 100 iterations.

For the main body of the paper, we opt to use the Adam optimizer (Kingma & Ba, 2014) and Gaussian Glorot initialization (Glorot & Bengio, 2010). Although we can achieve more impressive results on the lottery ticket experiment with other hyperparameters, we intend these choices to be as generic as possible in an effort to minimize the extent to which our main results depend on hand-chosen hyperparameters. In this Appendix, we select the learning rate for Adam that we use in the main body of the paper.

In addition, we consider a wide range of other hyperparameters, including other optimization algorithms (SGD with and without momentum), initialization strategies (Gaussian distributions

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//f9dcec75-58e0-42b6-a863-afa4b399abbf/markdown_0/imgs/img_in_chart_box_219_175_1000_417.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A15Z%2F-1%2F%2F3b6701dcabd6ff426815a0420424e5d917e9939f45973875f2f025987e57ee02" alt="Image" width="63%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 26: The early-stopping iteration and validation accuracy at that iteration of the iterative lottery ticket experiment on the Lenet architecture trained with MNIST using the Adam optimizer at various learning rates. Each line represents a different learning rate.</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//f9dcec75-58e0-42b6-a863-afa4b399abbf/markdown_0/imgs/img_in_chart_box_220_545_1000_789.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A15Z%2F-1%2F%2F4c50af436288a1bbe558ea4ae0f3f27ae1810fbdb0999c008af475f2242a87cd" alt="Image" width="63%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 27: The early-stopping iteration and validation accuracy at that iteration of the iterative lottery ticket experiment on the Lenet architecture trained with MNIST using stochastic gradient descent at various learning rates.</div> </div>


with various standard deviations), network sizes (larger and smaller hidden layers), and pruning strategies (faster and slower pruning rates). In each experiment, we vary the chosen hyperparameter while keeping all others at their default values (Adam with the chosen learning rate, Gaussian Glorot initialization, hidden layers with 300 and 100 units). The data presented in this appendix was collected by training variations of the Lenet architecture more than 3,000 times.

### G.2 LEARNING RATE

In this Subsection, we perform the lottery ticket experiment on the Lenet architecture as optimized with Adam, SGD, and SGD with momentum at various learning rates.

Here, we select the learning rate that we use for Adam in the main body of the paper. Our criteria for selecting the learning rate are as follows:

1. On the unpruned network, it should minimize training iterations necessary to reach early-stopping and maximize validation accuracy at that iteration. That is, it should be a reasonable hyperparameter for optimizing the unpruned network even if we are not running the lottery ticket experiment.

2. When running the iterative lottery ticket experiment, it should make it possible to match the early-stopping iteration and accuracy of the original network with as few parameters as possible.

3. Of those options that meet (1) and (2), it should be on the conservative (slow) side so that it is more likely to productively optimize heavily pruned networks under a variety of conditions with a variety of hyperparameters.

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//f9dcec75-58e0-42b6-a863-afa4b399abbf/markdown_1/imgs/img_in_chart_box_219_177_999_426.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A18Z%2F-1%2F%2Fbc10cf913697c46517b2a0350b3f56c851788e24a3cbc7076b8bce97ca470375" alt="Image" width="63%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 28: The early-stopping iteration and validation accuracy at that iteration of the iterative lottery ticket experiment on the Lenet architecture trained with MNIST using stochastic gradient descent with momentum  $(0.9)$ at various learning rates.</div> </div>


Figure 26 shows the early-stopping iteration and validation accuracy at that iteration of performing the iterative lottery ticket experiment with the Lenet architecture optimized with Adam at various learning rates. According to the graph on the right of Figure 26, several learning rates between 0.0002 and 0.002 achieve similar levels of validation accuracy on the original network and maintain that performance to similar levels as the network is pruned. Of those learning rates, 0.0012 and 0.002 produce the fastest early-stopping times and maintain them to the smallest network sizes. We choose 0.0012 due to its higher validation accuracy on the unpruned network and in consideration of criterion (3) above.

We note that, across all of these learning rates, the lottery ticket pattern (in which learning becomes faster and validation accuracy increases with iterative pruning) remains present. Even for those learning rates that did not satisfy the early-stopping criterion within 50,000 iterations (2.5e-05 and 0.0064) still showed accuracy improvements with pruning.

### G.3 OTHER OPTIMIZATION ALGORITHMS

### G.3.1 SGD

Here, we explore the behavior of the lottery ticket experiment when the network is optimized with stochastic gradient descent (SGD) at various learning rates. The results of doing so appear in Figure 27. The lottery ticket pattern appears across all learning rates, including those that fail to satisfy the early-stopping criterion within 50,000 iterations. SGD learning rates 0.4 and 0.8 reach early-stopping in a similar number of iterations as the best Adam learning rates (0.0012 and 0.002) but maintain this performance when the network has been pruned further (to less than 1% of its original size for SGD vs. about 3.6% of the original size for Adam). Likewise, on pruned networks, these SGD learning rates achieve equivalent accuracy to the best Adam learning rates, and they maintain that high accuracy when the network is pruned as much as the Adam learning rates.

### G.3.2 MOMENTUM

Here, we explore the behavior of the lottery ticket experiment when the network is optimized with SGD with momentum (0.9) at various learning rates. The results of doing so appear in Figure 28. Once again, the lottery ticket pattern appears across all learning rates, with learning rates between 0.025 and 0.1 maintaining high validation accuracy and faster learning for the longest number of pruning iterations. Learning rate 0.025 achieves the highest validation accuracy on the unpruned network; however, its validation accuracy never increases as it is pruned, instead decreasing gradually, and higher learning rates reach early-stopping faster.

### G.4 ITERATIVE PRUNING RATE

When running the iterative lottery ticket experiment on Lenet, we prune each layer of the network separately at a particular rate. That is, after training the network, we prune  $k\%$ of the weights in

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//f9dcec75-58e0-42b6-a863-afa4b399abbf/markdown_2/imgs/img_in_chart_box_221_181_999_433.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A19Z%2F-1%2F%2Ffc4926145d297a6712d6b8d00d3c033fb10238653d8c402a132a7c9791ac42ec" alt="Image" width="63%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 29: The early-stopping iteration and validation accuracy at that iteration of the iterative lottery ticket experiment when pruned at different rates. Each line represents a different pruning rate—the percentage of lowest-magnitude weights that are pruned from each layer after each training iteration.</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//f9dcec75-58e0-42b6-a863-afa4b399abbf/markdown_2/imgs/img_in_chart_box_219_553_1002_795.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A20Z%2F-1%2F%2Ffc91584de7bb0ef9684cc5f26d31c9aaf99e188a6bbbc9d278b00c8d2296177f" alt="Image" width="63%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 30: The early-stopping iteration and validation accuracy at that iteration of the iterative lottery ticket experiment initialized with Gaussian distributions with various standard deviations. Each line is a different standard deviation for a Gaussian distribution centered at 0.</div> </div>


each layer ( $\frac{k}{2}\%$ of the weights in the output layer) before resetting the weights to their original initializations and training again. In the main body of the paper, we find that iterative pruning finds smaller winning tickets than one-shot pruning, indicating that pruning too much of the network at once diminishes performance. Here, we explore different values of k.

Figure 29 shows the effect of the amount of the network pruned on each pruning iteration on early-stopping time and validation accuracy. There is a tangible difference in learning speed and validation accuracy at early-stopping between the lowest pruning rates (0.1 and 0.2) and higher pruning rates (0.4 and above). The lowest pruning rates reach higher validation accuracy and maintain that validation accuracy to smaller network sizes; they also maintain fast early-stopping times to smaller network sizes. For the experiments throughout the main body of the paper and this Appendix, we use a pruning rate of 0.2, which maintains much of the accuracy and learning speed of 0.1 while reducing the number of training iterations necessary to get to smaller network sizes.

In all of the Lenet experiments, we prune the output layer at half the rate of the rest of the network. Since the output layer is so small (1,000 weights out of 266,000 for the overall Lenet architecture), we found that pruning it reaches a point of diminishing returns much earlier the other layers.

### G.5 INITIALIZATION DISTRIBUTION

To this point, we have considered only a Gaussian Glorot (Glorot & Bengio, 2010) initialization scheme for the network. Figure 30 performs the lottery ticket experiment while initializing the Lenet architecture from Gaussian distributions with a variety of standard deviations. The networks were optimized with Adam at the learning rate chosen earlier. The lottery ticket pattern continues to appear across all standard deviations. When initialized from a Gaussian distribution with standard deviation

0.1, the Lenet architecture maintained high validation accuracy and low early-stopping times for the longest, approximately matching the performance of the Glorot-initialized network.

### G.6 NETWORK SIZE

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//f9dcec75-58e0-42b6-a863-afa4b399abbf/markdown_3/imgs/img_in_chart_box_221_299_997_583.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A22Z%2F-1%2F%2F3d21a0b53a9522c0575fe286ef92d91327ed9cfaa9b4b4154a32aa8c0204b952" alt="Image" width="63%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//f9dcec75-58e0-42b6-a863-afa4b399abbf/markdown_3/imgs/img_in_chart_box_220_606_997_856.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A23Z%2F-1%2F%2F4ee646c9a06d165ab0f7ff13666eb6e8c3f704f92270208f799fe4e5671419ff" alt="Image" width="63%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 31: The early-stopping iteration and validation accuracy at that iteration of the iterative lottery ticket experiment on the Lenet architecture with various layer sizes. The label for each line is the size of the first and second hidden layers of the network. All networks had Gaussian Glorot initialization and were optimized with Adam (learning rate 0.0012). Note that the x-axis of this plot charts the number of weights remaining, while all other graphs in this section have charted the percent of weights remaining.</div> </div>


Throughout this section, we have considered the Lenet architecture with 300 units in the first hidden layer and 100 units in the second hidden layer. Figure 31 shows the early-stopping iterations and validation accuracy at that iteration of the Lenet architecture with several other layer sizes. All networks we tested maintain the 3:1 ratio between units in the first hidden layer and units in the second hidden layer.

The lottery ticket hypothesis naturally invites a collection of questions related to network size. Generalizing, those questions tend to take the following form: according to the lottery ticket hypothesis, do larger networks, which contain more subnetworks, find “better” winning tickets? In line with the generality of this question, there are several different answers.

If we evaluate a winning ticket by the accuracy it achieves, then larger networks do find better winning tickets. The right graph in Figure 31 shows that, for any particular number of weights (that is, any particular point on the x-axis), winning tickets derived from initially larger networks reach higher accuracy. Put another way, in terms of accuracy, the lines are approximately arranged from bottom to top in increasing order of network size. It is possible that, since larger networks have more subnetworks, gradient descent found a better winning ticket. Alternatively, the initially larger networks have more units even when pruned to the same number of weights as smaller networks, meaning they are able to contain sparse subnetwork configurations that cannot be expressed by initially smaller networks.

If we evaluate a winning ticket by the time necessary for it to reach early-stopping, then larger networks have less of an advantage. The left graph in Figure 31 shows that, in general, early-stopping iterations do not vary greatly between networks of different initial sizes that have been pruned to the same number of weights. Upon exceedingly close inspection, winning tickets derived from initially larger networks tend to learn marginally faster than winning tickets derived from initially smaller networks, but these differences are slight.

If we evaluate a winning ticket by the size at which it returns to the same accuracy as the original network, the large networks do not have an advantage. Regardless of the initial network size, the right graph in Figure 31 shows that winning tickets return to the accuracy of the original network when they are pruned to between about 9,000 and 15,000 weights.

### H HYPERPARAMETER EXPLORATION FOR CONVOLUTIONAL NETWORKS

This Appendix accompanies Sections 3 of the main paper. It explores the space of optimization algorithms and hyperparameters for the Conv-2, Conv-4, and Conv-6 architectures evaluated in Section 3 with the same two purposes as Appendix G: explaining the hyperparameters used in the main body of the paper and evaluating the lottery ticket experiment on other choices of hyperparameters.

### H.1 EXPERIMENTAL METHODOLOGY

The Conv-2, Conv-4, and Conv-6 architectures are variants of the VGG (Simonyan & Zisserman, 2014) network architecture scaled down for the CIFAR10 (Krizhevsky & Hinton, 2009) dataset. Like VGG, the networks consist of a series of modules. Each module has two layers of 3x3 convolutional filters followed by a maxpool layer with stride 2. After all of the modules are two fully-connected layers of size 256 followed by an output layer of size 10; in VGG, the fully-connected layers are of size 4096 and the output layer is of size 1000. Like VGG, the first module has 64 convolutions in each layer, the second has 128, the third has 256, etc. The Conv-2, Conv-4, and Conv-6 architectures have 1, 2, and 3 modules, respectively.

The CIFAR10 dataset consists of 50,000 32x32 color (three-channel) training examples and 10,000 test examples. We randomly sampled a 5,000-example validation set from the training set and used the remaining 45,000 training examples as our training set for the rest of the paper. The hyperparameter selection experiments throughout this Appendix are evaluated on the validation set, and the examples in the main body of this paper (which make use of these hyperparameters) are evaluated on test set. The training set is presented to the network in mini-batches of 60 examples; at each epoch, the entire training set is shuffled.

The Conv-2, Conv-4, and Conv-6 networks are initialized with Gaussian Glorot initialization (Glorot & Bengio, 2010) and are trained for the number of iterations specified in Figure 2. The number of training iterations was selected such that heavily-pruned networks could still train in the time provided. On dropout experiments, the number of training iterations is tripled to provide enough time for the dropout-regularized networks to train. We optimize these networks with Adam, and select the learning rate for each network in this Appendix.

As with the MNIST experiments, validation and test performance is only considered retroactively and has no effect on the progression of the lottery ticket experiments. We measure validation and test loss and accuracy every 100 training iterations.

Each line in each graph of this section represents the average of three separate experiments, with error bars indicating the minimum and maximum value that any experiment took on at that point. (Experiments in the main body of the paper are conducted five times.)

We allow convolutional layers and fully-connected layers to be pruned at different rates; we select those rates for each network in this Appendix. The output layer is pruned at half of the rate of the fully-connected layers for the reasons described in Appendix G.

### H.2 LEARNING RATE

In this Subsection, we perform the lottery ticket experiment on the the Conv-2, Conv-4, and Conv-6 architectures as optimized with Adam at various learning rates.

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//3c6dedba-85bc-4813-b50a-d32af2e2a8fc/markdown_1/imgs/img_in_chart_box_220_457_605_673.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A15Z%2F-1%2F%2F47302e1796d8bc13e382dead9d1399277a25ca303b1f4217781a054483739c35" alt="Image" width="31%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//3c6dedba-85bc-4813-b50a-d32af2e2a8fc/markdown_1/imgs/img_in_chart_box_616_469_998_673.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A16Z%2F-1%2F%2F0d0920efa57f256c05e0368baaeb23ac4b66a512607acec0b5a22b373eac5e10" alt="Image" width="31%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//3c6dedba-85bc-4813-b50a-d32af2e2a8fc/markdown_1/imgs/img_in_chart_box_220_689_605_894.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A16Z%2F-1%2F%2F259f6f906f70172d60cd8ea908b800647eb006ed8e4ae8ba87ff4d276da300fc" alt="Image" width="31%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//3c6dedba-85bc-4813-b50a-d32af2e2a8fc/markdown_1/imgs/img_in_chart_box_616_692_998_895.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A16Z%2F-1%2F%2F220227792011b42887a690ad1200526eb8681d3d5038e64dee8e610029417da2" alt="Image" width="31%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//3c6dedba-85bc-4813-b50a-d32af2e2a8fc/markdown_1/imgs/img_in_chart_box_219_910_605_1116.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A16Z%2F-1%2F%2F083c5b13a09d5b7e9093f630f6e4ec0b724ab4ebd509f135d7f9d4a6a9912ebd" alt="Image" width="31%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//3c6dedba-85bc-4813-b50a-d32af2e2a8fc/markdown_1/imgs/img_in_chart_box_615_912_998_1114.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A16Z%2F-1%2F%2F7199c91814812711b3e5e346df27a8e64a21039077128b1afa40d4de914642d0" alt="Image" width="31%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 32: The early-stopping iteration and validation accuracy at that iteration of the iterative lottery ticket experiment on the Conv-2 (top), Conv-4 (middle), and Conv-6 (bottom) architectures trained using the Adam optimizer at various learning rates. Each line represents a different learning rate.</div> </div>


Here, we select the learning rate that we use for Adam in the main body of the paper. Our criteria for selecting the learning rate are the same as in Appendix G: minimizing training iterations and maximizing accuracy at early-stopping, finding winning tickets containing as few parameters as possible, and remaining conservative enough to apply to a range of other experiments.

Figure 32 shows the results of performing the iterative lottery ticket experiment on the Conv-2 (top), Conv-4 (middle), and Conv-6 (bottom) architectures. Since we have not yet selected the pruning rates for each network, we temporarily pruned fully-connected layers at 20% per iteration, convolutional layers at 10% per iteration, and the output layer at 10% per iteration; we explore this part of the hyperparameter space in a later subsection.

For Conv-2, we select a learning rate of 0.0002, which has the highest initial validation accuracy, maintains both high validation accuracy and low early-stopping times for the among the longest, and reaches the fastest early-stopping times. This learning rate also leads to a 3.3 percentage point improvement in validation accuracy when the network is pruned to 3% of its original size. Other learning rates, such as 0.0004, have lower initial validation accuracy (65.2% vs 67.6%) but eventually reach higher absolute levels of validation accuracy (71.7%, a 6.5 percentage point increase, vs. 70.9%, a 3.3 percentage point increase). However, learning rate 0.0002 shows the highest proportional decrease in early-stopping times: 4.8x (when pruned to 8.8% of the original network size).

For Conv-4, we select learning rate 0.0003, which has among the highest initial validation accuracy, maintains high validation accuracy and fast early-stopping times when pruned by among the most, and balances improvements in validation accuracy (3.7 percentage point improvement to 78.6% when 5.4% of weights remain) and improvements in early-stopping time (4.27x when 11.1% of weights remain). Other learning rates reach higher validation accuracy (0.0004—3.6 percentage point improvement to 79.1% accuracy when 5.4% of weights remain) or show better improvements in early-stopping times (0.0002—5.1x faster when 9.2% of weights remain) but not both.

For Conv-6, we also select learning rate 0.0003 for similar reasons to those provided for Conv-4. Validation accuracy improves by 2.4 percentage points to 81.5% when 9.31% of weights remain and early-stopping times improve by 2.61x when pruned to 11.9%. Learning rate 0.0004 reaches high final validation accuracy (81.9%, an increase of 2.7 percentage points, when 15.2% of weights remain) but with smaller improvements in early-stopping times, and learning rate 0.0002 shows greater improvements in early-stopping times (6.26x when 19.7% of weights remain) but reaches lower overall validation accuracy.

We note that, across nearly all combinations of learning rates, the lottery ticket pattern—where early-stopping times were maintain or decreased and validation accuracy was maintained or increased during the course of the lottery ticket experiment—continues to hold. This pattern fails to hold at the very highest learning rates: early-stopping times decreased only briefly (in the case of Conv-2 or Conv-4) or not at all (in the case of Conv-6), and accuracy increased only briefly (in the case of all three networks). This pattern is similar to that which we observe in Section 4: at the highest learning rates, our iterative pruning algorithm fails to find winning tickets.

### H.3 OTHER OPTIMIZATION ALGORITHMS

### H.3.1 SGD

Here, we explore the behavior of the lottery ticket experiment when the Conv-2, Conv-4, and Conv-6 networks are optimized with stochastic gradient descent (SGD) at various learning rates. The results of doing so appear in Figure 33. In general, these networks—particularly Conv-2 and Conv-4—proved challenging to train with SGD and Glorot initialization. As Figure 33 reflects, we could not find SGD learning rates for which the unpruned networks matched the validation accuracy of the same networks when trained with Adam; at best, the SGD-trained unpruned networks were typically 2-3 percentage points less accurate. At higher learning rates than those in Figure 32, gradients tended to explode when training the unpruned network; at lower learning rates, the networks often failed to learn at all.

At all of the learning rates depicted, we found winning tickets. In all cases, early-stopping times initially decreased with pruning before eventually increasing again, just as in other lottery ticket experiments. The Conv-6 network also exhibited the same accuracy patterns as other experiments, with validation accuracy initially increasing with pruning before eventually decreasing again.

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//3c6dedba-85bc-4813-b50a-d32af2e2a8fc/markdown_3/imgs/img_in_chart_box_218_366_607_622.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A19Z%2F-1%2F%2Fe6d19ab91d1dd01922c2aa0b649f6b59db9724b1462968bdf67de71d3eb40ecc" alt="Image" width="31%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//3c6dedba-85bc-4813-b50a-d32af2e2a8fc/markdown_3/imgs/img_in_chart_box_615_403_999_617.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A19Z%2F-1%2F%2Fb47818eef4029c7a93ea6ca54e4bf62d11ea56d89f47afdb31ddd3ff925f54fb" alt="Image" width="31%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//3c6dedba-85bc-4813-b50a-d32af2e2a8fc/markdown_3/imgs/img_in_chart_box_219_687_605_892.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A19Z%2F-1%2F%2F504cee7920c2c6a07ae211d41ce3ba133f4883b8c44d9ddb964fc00742bb69ca" alt="Image" width="31%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//3c6dedba-85bc-4813-b50a-d32af2e2a8fc/markdown_3/imgs/img_in_chart_box_615_688_998_891.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A19Z%2F-1%2F%2Ff592f0040c81c695f8ba3eb4a9ab0026e2bf7b16d58cf923e7d63f4168ab813c" alt="Image" width="31%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//3c6dedba-85bc-4813-b50a-d32af2e2a8fc/markdown_3/imgs/img_in_chart_box_219_960_605_1166.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A20Z%2F-1%2F%2Fc1544bf8f124a0e1555b5f9608bd5d4b204dc0d7f098f955549c625bd6a83b2f" alt="Image" width="31%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//3c6dedba-85bc-4813-b50a-d32af2e2a8fc/markdown_3/imgs/img_in_chart_box_615_960_998_1165.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A20Z%2F-1%2F%2F83d4fc22483235b707864586e5cd6eb401aec8d90e8b8ee264498ac974a42bd9" alt="Image" width="31%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 33: The early-stopping iteration and validation accuracy at that iteration of the iterative lottery ticket experiment on the Conv-2 (top), Conv-4 (middle), and Conv-6 (bottom) architectures trained using SGD at various learning rates. Each line represents a different learning rate. The legend for each pair of graphs is above the graphs.</div> </div>


However, the Conv-2 and Conv-4 architectures exhibited a different validation accuracy pattern from other experiments in this paper. Accuracy initially declined with pruning before rising as the network was further pruned; it eventually matched or surpassed the accuracy of the unpruned network. When they eventually did surpass the accuracy of the original network, the pruned networks reached early-stopping in about the same or fewer iterations than the original network, constituting a winning ticket by our definition. Interestingly, this pattern also appeared for Conv-6 networks at slower SGD learning rates, suggesting that faster learning rates for Conv-2 and Conv-4 than those in Figure 32 might cause the usual lottery ticket accuracy pattern to reemerge. Unfortunately, at these higher learning rates, gradients exploded on the unpruned networks, preventing us from running these experiments.

### H.3.2 MOMENTUM

Here, we explore the behavior of the lottery ticket experiment when the network is optimized with SGD with momentum  $(0.9)$ at various learning rates. The results of doing so appear in Figure 34. In general, the lottery ticket pattern continues to apply, with early-stopping times decreasing and accuracy increasing as the networks are pruned. However, there were two exceptions to this pattern:

1. At the very lowest learning rates (e.g., learning rate 0.001 for Conv-4 and all but the highest learning rate for Conv-2), accuracy initially decreased before increasing to higher levels than reached by the unpruned network; this is the same pattern we observed when training these networks with SGD.

2. At the very highest learning rates (e.g., learning rates 0.005 and 0.008 for Conv-2 and Conv-4), early-stopping times never decreased and instead remained stable before increasing; this is the same pattern we observed for the highest learning rates when training with Adam.

### H.4 ITERATIVE PRUNING RATE

For the convolutional network architectures, we select different pruning rates for convolutional and fully-connected layers. In the Conv-2 and Conv-4 architectures, convolutional parameters make up a relatively small portion of the overall number of parameters in the models. By pruning convolutions more slowly, we are likely to be able to prune the model further while maintaining performance. In other words, we hypothesize that, if all layers were pruned evenly, convolutional layers would become a bottleneck that would make it more difficult to find lower parameter-count models that are still able to learn. For Conv-6, the opposite may be true: since nearly two thirds of its parameters are in convolutional layers, pruning fully-connected layers could become the bottleneck.

Our criterion for selecting hyperparameters in this section is to find a combination of pruning rates that allows networks to reach the lowest possible parameter-counts while maintaining validation accuracy at or above the original accuracy and early-stopping times at or below that for the original network.

Figure 35 shows the results of performing the iterative lottery ticket experiment on Conv-2 (top), Conv-4 (middle), and Conv-6 (bottom) with different combinations of pruning rates.

According to our criteria, we select an iterative convolutional pruning rate of 10% for Conv-2, 10% for Conv-4, and 15% for Conv-6. For each network, any rate between 10% and 20% seemed reasonable. Across all convolutional pruning rates, the lottery ticket pattern continued to appear.

### H.5 LEARNING RATES (DROPOUT)

In order to train the Conv-2, Conv-4, and Conv-6 architectures with dropout, we repeated the exercise from Section H.2 to select appropriate learning rates. Figure 32 shows the results of performing the iterative lottery ticket experiment on Conv-2 (top), Conv-4 (middle), and Conv-6 (bottom) with dropout and Adam at various learning rates. A network trained with dropout takes longer to learn, so we trained each architecture for three times as many iterations as in the experiments without dropout: 60,000 iterations for Conv-2, 75,000 iterations for Conv-4, and 90,000 iterations for Conv-6. We iteratively pruned these networks at the rates determined in Section H.4.

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//a0df7659-202f-4860-b802-1e43d09ad304/markdown_1/imgs/img_in_chart_box_219_335_606_585.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A16Z%2F-1%2F%2F016654a632deae29eb35297d7b57f05a47f35365e05f2a2a4b3182171ac93333" alt="Image" width="31%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//a0df7659-202f-4860-b802-1e43d09ad304/markdown_1/imgs/img_in_chart_box_613_342_1000_581.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A17Z%2F-1%2F%2F69866faad53cb7d2f2c00580a200f15c224af10584b7ac208566896826e62359" alt="Image" width="31%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//a0df7659-202f-4860-b802-1e43d09ad304/markdown_1/imgs/img_in_chart_box_219_649_605_856.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A17Z%2F-1%2F%2F3e9d6de1bac22ff23afe71d781d72f4aa0c9eb5275b98776fa6c16929b753169" alt="Image" width="31%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//a0df7659-202f-4860-b802-1e43d09ad304/markdown_1/imgs/img_in_chart_box_615_649_998_855.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A18Z%2F-1%2F%2F9875dcd5fceda6ed0a3b88ca460ac2822b99362938c056df8a0ad12c782c622f" alt="Image" width="31%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//a0df7659-202f-4860-b802-1e43d09ad304/markdown_1/imgs/img_in_chart_box_219_925_604_1128.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A18Z%2F-1%2F%2Ff44c7b8234f6a0d5cd1a16608b1acf8994d58054dbd96c3f93d9f18b18bc9c6d" alt="Image" width="31%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//a0df7659-202f-4860-b802-1e43d09ad304/markdown_1/imgs/img_in_chart_box_615_923_998_1127.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A18Z%2F-1%2F%2F56cec2b55ba7a78a98e730af89b1bff05706aa36b097410182b0b53958b58689" alt="Image" width="31%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 34: The early-stopping iteration and validation accuracy at that iteration of the iterative lottery ticket experiment on the Conv-2 (top), Conv-4 (middle), and Conv-6 (bottom) architectures trained using SGD with momentum (0.9) at various learning rates. Each line represents a different learning rate. The legend for each pair of graphs is above the graphs. Lines that are unstable and contain large error bars (large vertical lines) indicate that some experiments failed to learn effectively, leading to very low accuracy and very high early-stopping times; these experiments reduce the averages that the lines trace and lead to much wider error bars.</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//a0df7659-202f-4860-b802-1e43d09ad304/markdown_2/imgs/img_in_chart_box_220_456_604_662.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A20Z%2F-1%2F%2F206163f6467f314b9a5c61bc50b2145b6eed20254d625fe01c72c32590a73a77" alt="Image" width="31%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//a0df7659-202f-4860-b802-1e43d09ad304/markdown_2/imgs/img_in_chart_box_615_460_998_661.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A21Z%2F-1%2F%2F490589d5cf4e9891415c1c6d780b7248c8e6175fd957f006c0c571cb5f847b20" alt="Image" width="31%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//a0df7659-202f-4860-b802-1e43d09ad304/markdown_2/imgs/img_in_chart_box_220_680_605_882.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A21Z%2F-1%2F%2Fd59d7c85ed4ba13ba9a146e0c161c780506377a1872a69ce337d03396b7ba9f9" alt="Image" width="31%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//a0df7659-202f-4860-b802-1e43d09ad304/markdown_2/imgs/img_in_chart_box_615_680_998_882.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A22Z%2F-1%2F%2F5002efc9b1d94e2a4b9a12a00a6b91185b6ef5ad1ebd7926222abbcd1c113ff7" alt="Image" width="31%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//a0df7659-202f-4860-b802-1e43d09ad304/markdown_2/imgs/img_in_chart_box_219_900_605_1104.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A22Z%2F-1%2F%2F35ae2af3142aebba0c2468fa8589eec422ff78063896be7de866f80fd7f225e0" alt="Image" width="31%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//a0df7659-202f-4860-b802-1e43d09ad304/markdown_2/imgs/img_in_chart_box_615_898_998_1103.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A23Z%2F-1%2F%2F61ce9cd98981c6fca3409051a3cd868895a961f0d22d028c8148b1b560c434e8" alt="Image" width="31%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 35: The early-stopping iteration and validation accuracy at that iteration of the iterative lottery ticket experiment on the Conv-2 (top), Conv-4 (middle), and Conv-6 (bottom) architectures with an iterative pruning rate of 20% for fully-connected layers. Each line represents a different iterative pruning rate for convolutional layers.</div> </div>


The Conv-2 network proved to be difficult to consistently train with dropout. The top right graph in Figure 36 contains wide error bars and low average accuracy for many learning rates, especially early in the lottery ticket experiments. This indicates that some or all of the training runs failed to learn; when they were averaged into the other results, they produced the aforementioned pattern in the graphs. At learning rate 0.0001, none of the three trials learned productively until pruned to more than 26.5%, at which point all three trials started learning. At learning rate 0.0002, some of the trials failed to learn productively until several rounds of iterative pruning had passed. At learning rate 0.0003, all three networks learned productively at every pruning level. At learning rate 0.0004, one network occasionally failed to learn. We selected learning rate 0.0003, which seemed to allow networks to learn productively most often while achieving the highest initial accuracy.

It is interesting to note that networks that were unable to learn at a particular learning rate (for example, 0.0001) eventually began learning after several rounds of the lottery ticket experiment (that is, training, pruning, and resetting repeatedly). It is worth investigating whether this phenomenon was entirely due to pruning (that is, removing any random collection of weights would put the network in a configuration more amenable to learning) or whether training the network provided useful information for pruning, even if the network did not show improved accuracy.

For both the Conv-4 and Conv-6 architectures, a slightly slower learning rate (0.0002 as opposed to 0.0003) leads to the highest accuracy on the unpruned networks in addition to the highest sustained accuracy and fastest sustained learning as the networks are pruned during the lottery ticket experiment.

With dropout, the unpruned Conv-4 architecture reaches an average validation accuracy of 77.6%, a 2.7 percentage point improvement over the unpruned Conv-4 network trained without dropout and one percentage point lower than the highest average validation accuracy attained by a winning ticket. The dropout-trained winning tickets reach 82.6% average validation accuracy when pruned to 7.6%. Early-stopping times improve by up to 1.58x (when pruned to 7.6%), a smaller improvement than then 4.27x achieved by a winning ticket obtained without dropout.

With dropout, the unpruned Conv-6 architecture reaches an average validation accuracy of 81.3%, an improvement of 2.2 percentage points over the accuracy without dropout; this nearly matches the 81.5% average accuracy obtained by Conv-6 trained without dropout and pruned to 9.31%. The dropout-trained winning tickets further improve upon these numbers, reaching 84.8% average validation accuracy when pruned to 10.5%. Improvements in early-stopping times are less dramatic than without dropout: a 1.5x average improvement when the network is pruned to 15.1%.

At all learning rates we tested, the lottery ticket pattern generally holds for accuracy, with improvements as the networks are pruned. However, not all learning rates show the decreases in early-stopping times. To the contrary, none of the learning rates for Conv-2 show clear improvements in early-stopping times as seen in the other lottery ticket experiments. Likewise, the faster learning rates for Conv-4 and Conv-6 maintain the original early-stopping times until pruned to about 40%, at which point early-stopping times steadily increase.

### H.6 PRUNING CONVOLUTIONS VS. PRUNING FULLY-CONNECTED LAYERS

Figure 37 shows the effect of pruning convolutions alone (green), fully-connected layers alone (orange) and pruning both (blue). The x-axis measures the number of parameters remaining to emphasize the relative contributions made by pruning convolutions and fully-connected layers to the overall network. In all three cases, pruning convolutions alone leads to higher test accuracy and faster learning; pruning fully-connected layers alone generally causes test accuracy to worsen and learning to slow. However, pruning convolutions alone has limited ability to reduce the overall parameter-count of the network, since fully-connected layers comprise 99%, 89%, and 35% of the parameters in Conv-2, Conv-4, and Conv-6.

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//b6f1709d-a62e-4bca-8383-e4d460de0832/markdown_0/imgs/img_in_chart_box_220_460_604_663.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A14Z%2F-1%2F%2F3de6030239987fde39b5cc15cdbba0153d80ce90dcdbf091e4d1be310d97864e" alt="Image" width="31%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//b6f1709d-a62e-4bca-8383-e4d460de0832/markdown_0/imgs/img_in_chart_box_615_459_998_662.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A14Z%2F-1%2F%2F4c760add12c3960a33504046c4ae2b7bc6940eae42eb60e70524cefa880e993e" alt="Image" width="31%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//b6f1709d-a62e-4bca-8383-e4d460de0832/markdown_0/imgs/img_in_chart_box_219_680_604_883.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A14Z%2F-1%2F%2F9e5f9c077238bf43fd73bf779e33e224932c5b9286b68a4cb2243c8d5af2b237" alt="Image" width="31%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//b6f1709d-a62e-4bca-8383-e4d460de0832/markdown_0/imgs/img_in_chart_box_615_678_997_883.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A14Z%2F-1%2F%2Fc592c97475e403d32ac9730ef6efa59f9377fb005a4ffc12ba44596c31b31d54" alt="Image" width="31%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//b6f1709d-a62e-4bca-8383-e4d460de0832/markdown_0/imgs/img_in_chart_box_219_900_604_1105.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A14Z%2F-1%2F%2F053d749cdd0caf3bdb6720ec495753a260de385d4d563f02eda9a5ae1d0b69c5" alt="Image" width="31%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//b6f1709d-a62e-4bca-8383-e4d460de0832/markdown_0/imgs/img_in_chart_box_615_899_998_1104.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A14Z%2F-1%2F%2Fa26736edabcd0c80df52bec189e034b6cba809c52580823f98a5e11a595cda0e" alt="Image" width="31%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 36: The early-stopping iteration and validation accuracy at that iteration of the iterative lottery ticket experiment on the Conv-2 (top), Conv-4 (middle), and Conv-6 (bottom) architectures trained using dropout and the Adam optimizer at various learning rates. Each line represents a different learning rate.</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//b6f1709d-a62e-4bca-8383-e4d460de0832/markdown_1/imgs/img_in_chart_box_223_362_995_1128.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A15Z%2F-1%2F%2Facff20869463085f1f8c5472ca79127bf53382119ec0fe297ac39b5255b3cc84" alt="Image" width="63%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 37: Early-stopping iteration and accuracy of the Conv-2 (top), Conv-4 (middle), and Conv-6 (bottom) networks when only convolutions are pruned, only fully-connected layers are pruned, and both are pruned. The x-axis measures the number of parameters remaining, making it possible to see the relative contributions to the overall network made by pruning FC layers and convolutions individually.</div> </div>


## I HYPERPARAMETER EXPLORATION FOR VGG-19 AND RESNET-18 ON CIFAR10

This Appendix accompanies the VGG-19 and Resnet-18 experiments in Section 4. It details the pruning scheme, training regimes, and hyperparameters that we use for these networks.

#### I.1 GLOBAL PRUNING

In our experiments with the Lenet and Conv-2/4/6 architectures, we separately prune a fraction of the parameters in each layer (layer-wise pruning). In our experiments with VGG-19 and Resnet-18, we instead prune globally; that is, we prune all of the weights in convolutional layers collectively without regard for the specific layer from which any weight originated.

Figures 38 (VGG-19) and 39 (Resnet-18) compare the winning tickets found by global pruning (solid lines) and layer-wise pruning (dashed lines) for the hyperparameters from Section 4. When training VGG-19 with learning rate 0.1 and warmup to iteration 10,000, we find winning tickets when  $P_m \geq 6.9\%$ for layer-wise pruning vs.  $P_m \geq 1.5\%$ for global pruning. For other hyperparameters, accuracy similarly drops off when sooner for layer-wise pruning than for global pruning. Global pruning also finds smaller winning tickets than layer-wise pruning for Resnet-18, but the difference is less extreme than for VGG-19.

In Section 4, we discuss the rationale for the efficacy of global pruning on deeper networks. In summary, the layers in these deep networks have vastly different numbers of parameters (particularly severely so for VGG-19); if we prune layer-wise, we conjecture that layers with fewer parameters become bottlenecks on our ability to find smaller winning tickets.

Regardless of whether we use layer-wise or global pruning, the patterns from Section 4 hold: at learning rate 0.1, iterative pruning finds winning tickets for neither network; at learning rate 0.01, the lottery ticket pattern reemerges; and when training with warmup to a higher learning rate, iterative pruning finds winning tickets. Figures 40 (VGG-19) and 41 (Resnet-18) present the same data as Figures 7 (VGG-19) and 8 (Resnet-18) from Section 4 with layer-wise pruning rather than global pruning. The graphs follow the same trends as in Section 4, but the smallest winning tickets are larger than those found by global pruning.

#### I.2 VGG-19 DETAILS

The VGG19 architecture was first designed by Simonyan & Zisserman (2014) for Imagenet. The version that we use here was adapted by Liu et al. (2019) for CIFAR10. The network is structured as described in Figure 2: it has five groups of 3x3 convolutional layers, the first four of which are followed by max-pooling (stride 2) and the last of which is followed by average pooling. The network has one final dense layer connecting the result of the average-pooling to the output.

We largely follow the training procedure for resnet18 described in Appendix I:

- We use the same train/test/validation split.

We use the same data augmentation procedure.

We use a batch size of 64.

We use batch normalization.

We use a weight decay of 0.0001.

- We use three stages of training at decreasing learning rates. We train for 160 epochs (112,480 iterations), decreasing the learning rate by a factor of ten after 80 and 120 epochs.

We use Gaussian Glorot initialization.

We globally prune the convolutional layers of the network at a rate of 20% per iteration, and we do not prune the 5120 parameters in the output layer.

Liu et al. (2019) uses an initial pruning rate of 0.1. We train VGG19 with both this learning rate and a learning rate of 0.01.

#### I.3 RESNET-18 DETAILS

The Resnet-18 architecture was first introduced by He et al. (2016). The architecture comprises 20 total layers as described in Figure 2: a convolutional layer followed by nine pairs of convolutional layers (with residual connections around the pairs), average pooling, and a fully-connected output layer.

We follow the experimental design of He et al. (2016):

- We divide the training set into 45,000 training examples and 5,000 validation examples. We use the validation set to select hyperparameters in this appendix and the test set to evaluate in Section 4.

- We augment training data using random flips and random four pixel pads and crops.

We use a batch size of 128.

We use batch normalization.

We use weight decay of 0.0001.

• We train using SGD with momentum (0.9).

- We use three stages of training at decreasing learning rates. Our stages last for 20,000, 5,000, and 5,000 iterations each, shorter than the 32,000, 16,000, and 16,000 used in He et al. (2016). Since each of our iterative pruning experiments requires training the network 15-30 times consecutively, we select this abbreviated training schedule to make it possible to explore a wider range of hyperparameters.

We use Gaussian Glorot initialization.

We globally prune convolutions at a rate of 20% per iteration. We do not prune the 2560 parameters used to downsample residual connections or the 640 parameters in the fully-connected output layer, as they comprise such a small portion of the overall network.

#### I.4 LEARNING RATE

In Section 4, we observe that iterative pruning is unable to find winning tickets for VGG-19 and Resnet-18 at the typical, high learning rate used to train the network (0.1) but it is able to do so at a lower learning rate (0.01). Figures 42 and 43 explore several other learning rates. In general, iterative pruning cannot find winning tickets at any rate above 0.01 for either network; for higher learning rates, the pruned networks with the original initialization perform no better than when randomly reinitialized.

#### I.5 WARMUP ITERATION

In Section 4, we describe how adding linear warmup to the initial learning rate makes it possible to find winning tickets for VGG-19 and Resnet-18 at higher learning rates (and, thereby, winning tickets that reach higher accuracy). In Figures 44 and 45, we explore the number of iterations k over which warmup should occur.

For VGG-19, we were able to find values of k for which iterative pruning could identify winning tickets when the network was trained at the original learning rate (0.1). For Resnet-18, warmup made it possible to increase the learning rate from 0.01 to 0.03, but no further. When exploring values of k, we therefore us learning rate 0.1 for VGG-19 and 0.03 for Resnet-18.

In general, the greater the value of k, the higher the accuracy of the eventual winning tickets.

Resnet-18. For values of k below 5000, accuracy improves rapidly as k increases. This relationship reaches a point of diminishing returns above k = 5000. For the experiments in Section 4, we select k = 20000, which achieves the highest validation accuracy.

VGG-19. For values of k below 5000, accuracy improves rapidly as k increases. This relationship reaches a point of diminishing returns above k = 5000. For the experiments in Section 4, we select k = 10000, as there is little benefit to larger values of k.

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//bf25a612-00e1-4708-af18-a6ac435674a7/markdown_0/imgs/img_in_chart_box_219_190_998_404.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A13Z%2F-1%2F%2Ff78257cc5ea8264c34509c22804d94c04fcd1ed7bd403854174924bec5d034df" alt="Image" width="63%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 38: Validation accuracy (at 30K, 60K, and 112K iterations) of VGG-19 when iteratively pruned with global (solid) and layer-wise (dashed) pruning.</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//bf25a612-00e1-4708-af18-a6ac435674a7/markdown_0/imgs/img_in_chart_box_220_503_998_726.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A13Z%2F-1%2F%2F810b69237e495e829c30d18f30f1a5cea07bcd9dfbe2e278eac751248f9f2370" alt="Image" width="63%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 39: Validation accuracy (at 10K, 20K, and 30K iterations) of Resnet-18 when iteratively pruned with global (solid) and layer-wise (dashed) pruning.</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//bf25a612-00e1-4708-af18-a6ac435674a7/markdown_0/imgs/img_in_chart_box_221_822_1002_1036.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A14Z%2F-1%2F%2Fa255ac1c75f4de9ec1076e86c2d9c758ce76c09b7c96d0f06c0ad42bef689700" alt="Image" width="63%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 40: Test accuracy (at 30K, 60K, and 112K iterations) of VGG-19 when iteratively pruned with layer-wise pruning. This is the same as Figure 7, except with layer-wise pruning rather than global pruning.</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//bf25a612-00e1-4708-af18-a6ac435674a7/markdown_0/imgs/img_in_chart_box_222_1154_996_1376.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A14Z%2F-1%2F%2F7b4353d8ab9874d9f3ff3a3bf121aacbddc2def32c23bb3d6af5ee8ca7c165eb" alt="Image" width="63%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 41: Test accuracy (at 10K, 20K, and 30K iterations) of Resnet-18 when iteratively pruned with layer-wise pruning. This is the same as Figure 8 except with layer-wise pruning rather than global pruning.</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//bf25a612-00e1-4708-af18-a6ac435674a7/markdown_1/imgs/img_in_chart_box_220_170_1000_431.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A14Z%2F-1%2F%2F8df6489db45cd9b2337676d35b776d876a0466773e71604200d68a487135c38a" alt="Image" width="63%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 42: Validation accuracy (at 10K, 20K, and 30K iterations) of Resnet-18 when iteratively pruned and trained with various learning rates.</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//bf25a612-00e1-4708-af18-a6ac435674a7/markdown_1/imgs/img_in_chart_box_220_512_1000_767.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A14Z%2F-1%2F%2F5a11b9e569ad598340840d51cc859e153a7d5a466d99b42cc005e33dae47fc0b" alt="Image" width="63%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 43: Validation accuracy (at 30K, 60K, and 112K iterations) of VGG-19 when iteratively pruned and trained with various learning rates.</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//bf25a612-00e1-4708-af18-a6ac435674a7/markdown_1/imgs/img_in_chart_box_221_852_998_1086.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A14Z%2F-1%2F%2F86d6c7b20b97c9414e228604cce1971be62da8ef060d746f121325f039ee20fb" alt="Image" width="63%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 44: Validation accuracy (at 10K, 20K, and 30K iterations) of Resnet-18 when iteratively pruned and trained with varying amounts of warmup at learning rate 0.03.</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//bf25a612-00e1-4708-af18-a6ac435674a7/markdown_1/imgs/img_in_chart_box_220_1171_997_1406.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-23T23%3A18%3A14Z%2F-1%2F%2Fb3d5557ad81d2c0c811c02f0a1b3d79a84c64927fc660737f10cdf21d0f275db" alt="Image" width="63%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 45: Validation accuracy (at 30K, 60K, and 112K iterations) of VGG-19 when iteratively pruned and trained with varying amounts of warmup at learning rate 0.1.</div> </div>



