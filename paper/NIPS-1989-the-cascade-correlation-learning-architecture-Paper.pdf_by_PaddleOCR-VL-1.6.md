---
sr-due: 2026-08-16
sr-interval: 11
sr-ease: 270
---
#paper 

# 级联相关学习架构（The Cascade-Correlation Learning Architecture）

Scott E. Fahlman 与 Christian Lebiere

计算机科学学院

卡内基梅隆大学（Carnegie-Mellon University）

宾夕法尼亚州匹兹堡，邮编 15213

## 摘要（ABSTRACT）

级联相关（Cascade-Correlation）是一种用于人工神经网络的新型架构与监督学习算法。它并非仅仅在一个固定拓扑结构的网络中调整权重，而是从一个最小化的网络开始，然后自动地逐个训练并添加新的隐层单元，从而构建出多层结构。一旦新的隐层单元被加入网络，其输入端的权重即被冻结。该单元随后成为网络中一个永久性的特征检测器（feature-detector），可用于产生输出，或用于构建其他更复杂的特征检测器。级联相关架构相比现有算法具有若干优势：它学习速度非常快；网络能够自主确定其规模与拓扑结构；即使训练集发生变化，它也能保留已构建的结构；并且它不需要通过网络连接反向传播误差信号。

## 级联相关的描述（DESCRIPTION OF CASCADE-CORRELATION）

阻碍人工神经网络在真实世界问题中广泛应用的最重要问题，是现有学习算法（如反向传播，back-propagation 或简称 "backprop"）的缓慢速度。
造成这种缓慢的一个因素是我们所谓的**移动目标问题（moving target problem）**：由于网络中所有权值同时都在变化，每个隐层单元都处于一个不断变化的环境中。它们无法迅速地在整体问题求解中承担有用的角色，而是进行着一种复杂的“舞蹈”，其中包含了大量无效的动作。

级联相关学习算法正是为了尝试解决这一问题而开发的。在我们所考察的问题中，它的学习速度远快于反向传播，同时还解决了其他一些问题。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//902c6735-8d56-4f1d-9753-9890197b2bf8/markdown_1/imgs/img_in_image_box_167_126_997_540.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-01T10%3A56%3A49Z%2F-1%2F%2F68e6de92b54c9089bb5fb720735fe2c7d98bc2f62f645d1071675bbeccfef80e" alt="Image" width="67%" /></div>

<div style="text-align: center;"><div style="text-align: center;">图 1：级联架构（Cascade architecture），在已添加两个隐层单元之后的状态。垂直线表示对所有输入激活值求和。方框标记的连接为冻结（frozen）连接，X 标记的连接为反复训练的连接。</div> </div>

级联相关结合了两大核心思想：第一是**级联架构（cascade architecture）**，即隐层单元被逐个添加到网络中，且在添加后不再改变。第二是**学习算法**，它负责创建并安装新的隐层单元。对于每一个新的隐层单元，我们尝试最大化该单元输出与待消除的残余误差信号（residual error signal）之间相关性的幅值。

### 整体网络架构
级联架构如图 1 所示。它开始时只有**若干输入**和**一个或多个输出单元**，没有隐层单元。输入和输出的数量由问题本身以及实验者选择的输入/输出表示方式决定。每个输入都通过一个可调权重的连接与每个输出单元相连。此外还有一个偏置输入（bias input），永久设置为 +1。

输出单元可以仅对其加权输入进行线性求和，也可以使用某种非线性激活函数。在我们迄今为止运行的实验中，我们使用了一种对称的 S 型激活函数（sigmoidal activation function），即**双曲正切**（hyperbolic tangent），其输出范围为 -1.0 到 +1.0。对于需要精确模拟量输出而非二分类的问题，线性输出单元可能是最佳选择，但我们尚未研究过此类问题。

### 隐层单元介绍
隐层单元 可以**逐个**添加到网络中
- 每个新的隐层单元都接收 来自网络**所有原始输入**的连接，以及来自**每一个先前已存在的隐层单元**的连接。
- 该隐层单元的**输入权重**在其被加入网络时即被**冻结**；只有**输出权重**会被反复训练。

每个新单元都会为网络增加一个新的one-unit layer（除非它的一些传入权重恰好为零）。这导致了非常强大的高阶特征检测器的产生；但同时也可能导致网络非常深，且隐层单元的扇入（fan-in）很高。在添加新单元时，有多种可能的策略可以最小化网络深度和扇入，但我们尚未探索这些策略。

### 优化算法（不同于反向传播）
学习算法从不包含隐层单元开始。直接输入-输出连接在整个训练集上进行尽可能好的训练。由于不需要通过隐层单元反向传播，我们可以使用 Widrow-Hoff 规则（或称 "delta" 规则）、感知机学习算法（Perceptron learning algorithm），或任何其他用于单层网络的著名学习算法。在我们的模拟中，我们使用 Fahlman 的 "quickprop" 算法 [Fahlman, 1988] 来训练输出权重。在没有隐层单元的情况下，它的作用本质上类似于 delta 规则，只是收敛速度更快。

### 何时添加隐层单元
在某一点上，这种训练将接近渐近线。当在**一定数量**（由超参数`patience`指定）的训练周期（training cycles）后**没有发生显著的误差减小**时，我们让网络最后一次遍历整个训练集以测量误差。
如果我们对网络的性能感到满意，就停止；如果不满意，我们尝试通过向网络添加一个新的隐层单元来进一步减小残余误差。新单元被加入网络后，其输入权重被冻结，然后所有输出权重再次使用 quickprop 进行训练。
这个循环不断重复，直到误差小到可接受为止（或者直到我们放弃）。

### 如何构建隐层单元
为了创建一个新的隐层单元，我们从一个**候选单元**（candidate unit）开始，该单元接收来自网络所有外部输入 以及 所有先前已存在隐层单元的可训练输入连接。**该候选单元的输出尚未连接到活跃网络中**。我们对训练集中的样本进行若干轮遍历，在每轮之后调整候选单元的**输入权重**

#### 优化目标
这种调整的目标是最大化 $S$（**候选单元的输出值 与 最终输出误差的 相关性**），
即对所有输出单元 $o$ 的候选单元值 $V$ 与在单元 $o$ 处观测到的残余输出误差 $E_o$ 之间相关性（协方差）幅值的总和。我们将 $S$ 定义为
$$ S=\sum_{o}\left|\sum_{p}(V_{p}-\overline{V})\left(E_{p,o}-\overline{E_{o}}\right)\right| $$ 
其中 $o$ 是测量误差的网络输出，$p$ 是训练模式（？单个训练样本）。
- $V_p$：这个候选单元在第p个训练样本上的输出值
- $E_{p,o}$：输出单元 $o$ 在 第p个训练样本上的 输出误差

>最终要优化的参数还是 候选单元的输入权重$w$

#### 优化算法
为了最大化 $S$，我们必须计算 $\partial S/\partial w_i$，即 $S$ 关于候选单元每个输入权重 $w_i$ 的偏导数。按照与 反向传播规则的推导非常类似的方式，我们可以展开并对 $S$ 的公式求导，得到
$$ \partial S/\partial w_{i}=\sum_{p,o}\sigma_{o}(E_{p,o}-\overline{{E_{o}}})f_{p}^{\prime}I_{i,p} $$ 其中 $\sigma_0$ 是候选单元值与输出 $o$ 之间相关性的符号，$f_p^{\prime}$ 是候选单元激活函数对其输入总和在模式 $p$ 下的导数，$I_{i,p}$ 是候选单元在模式 $p$ 下从单元 $i$ 接收到的输入。

在计算完每个输入连接的 $\partial S/\partial w_i$ 后，我们可以执行**梯度上升**（gradient ascent）以最大化 $S$。这里我们同样只训练单层权重。我们再次使用 quickprop 更新规则以实现更快的收敛。当 $S$ 停止改善时，我们将该候选单元安装为活跃网络中的一个单元，冻结其输入权重，并继续上述循环。

由于 $S$ 的公式中含有绝对值，候选单元只关心其与给定输出处误差相关性的幅值，而不关心相关性的符号。
一般而言，如果一个隐层单元与某个给定单元处的误差呈正相关，它将发展出一个指向该单元的负连接权重，试图抵消部分误差；如果相关性为负，则输出权重将为正。
由于一个单元指向不同输出的权重可能符号混杂，一个单元有时可以通过与某一输出的误差发展正相关、而与另一输出的误差发展负相关，从而同时服务于两个目的。

### 其他构建方法
可以使用**候选单元池**（pool of candidate units）以并行构建，每个单元具有不同的随机初始权重。它们都接收相同的输入信号，并看到每个模式、每个输出的相同残余误差。由于它们在训练期间互不干扰，也不影响活跃网络，所有这些候选单元都可以并行训练；每当我们判断不再有进展时，就安装相关性得分最高的候选单元。使用这种候选池有两方面的好处：它极大地降低了因单个候选在训练过程中陷入停滞而永久安装一个无用单元的风险，并且（在并行机器上）它可以加速训练，因为权重空间的许多部分可以同时被探索。

**候选单元可以是不同类型的**，隐层单元和候选单元可以都是同一类型，例如使用 S 型激活函数。或者，我们可以创建一个包含多种非线性激活函数混合的候选单元池——有些是 S 型的，有些是高斯（Gaussian）的，有些具有径向激活函数（radial activation functions）等等——并让它们竞争被选入活跃网络。迄今为止，我们已经探索了全 S 型和全高斯型的情况，但对于混合单元类型的网络，我们尚未获得大量的模拟数据。

关于该算法实现的最后一点说明：当输出层的权重被训练时，活跃网络中的其他权重是被冻结的。当候选权重被训练时，活跃网络中的任何权重都不会改变。在内存充足的机器上，可以记录整个 epoch 的单元值和输出误差，然后在训练期间反复使用这些缓存值，而不是为每个训练样本反复重新计算。当活跃网络变得很大时，这可以带来巨大的速度提升。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//b717ce63-2445-4ef0-961c-b0c1bb662d2d/markdown_0/imgs/img_in_image_box_243_117_1058_479.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-01T10%3A56%3A48Z%2F-1%2F%2F5ac1821cec2d7f35081526149877092da9e05a8710f95c2787229e11a9fb5c29" alt="Image" width="66%" /></div>

<div style="text-align: center;"><div style="text-align: center;">图 2：双螺旋问题（two-spirals problem）的训练点，以及使用级联相关训练的一个网络的输出模式。</div> </div>

## 2 基准测试结果（BENCHMARK RESULTS）

### 2.1 双螺旋问题（THE TWO-SPIRALS PROBLEM）

选择“双螺旋”基准作为本研究的主要基准，是因为对于反向传播家族算法来说，它是一个极其困难的问题。该问题最初由 MITRE 公司的 Alexis Wieland 提出。网络有两个连续值输入和一个单一输出。训练集由 194 个 X-Y 值组成，其中一半应产生 +1 输出，另一半产生 -1 输出。这些训练点排列成两个相互交织的螺旋，围绕原点旋转三圈，如图 2a 所示。目标是开发一个具有 S 型单元的前馈网络，能够正确分类全部 194 个训练案例。显然需要一些隐层单元，因为单个线性分隔器无法将这样交织在一起的两组数据分开。

Wieland（未发表）报告称，MITRE 使用的一种改进版反向传播需要 150,000 到 200,000 个 epoch 才能解决这个问题，而且他们从未使用标准反向传播获得过解决方案。Lang 和 Witbrock [Lang, 1988] 尝试使用 2-5-5-5-1 网络（三个隐藏层，每层五个单元）来解决这个问题。他们的网络不同寻常之处在于提供了“捷径”连接：每个单元接收来自前面每一层中每个单元的传入连接，而不仅仅来自紧邻的前一层。使用这种架构，标准反向传播能够在 20,000 个 epoch 内解决这个问题，使用改进误差函数的反向传播需要 12,000 个 epoch，而 quickprop 需要 8,000 个 epoch。这是迄今为止报告的最佳双螺旋性能。Lang 和 Witbrock 还报告称，他们使用 2-5-5-1 网络（总共只有十个隐层单元）获得了一个解决方案，但该解决方案需要 60,000 个 quickprop epoch。

我们使用级联相关算法对该问题运行了 100 次，输出单元和隐层单元均使用 S 型激活函数，候选单元池大小为 8。所有试验都成功了，平均需要 1700 个 epoch。（这个数字同时包括了用于训练输出权重和训练候选单元的 epoch。）构建到网络中的隐层单元数量从 12 到 19 不等，平均为 15.2，中位数为 15。以下是所创建隐层单元数量的直方图：

<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>隐层单元数（Hidden Units）</td><td style='text-align: center; word-wrap: break-word;'>试验次数（Number of Trials）</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>12</td><td style='text-align: center; word-wrap: break-word;'>4 ####</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>13</td><td style='text-align: center; word-wrap: break-word;'>9 ########</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>14</td><td style='text-align: center; word-wrap: break-word;'>24 ########################</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>15</td><td style='text-align: center; word-wrap: break-word;'>19 ###################</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>16</td><td style='text-align: center; word-wrap: break-word;'>24 ########################</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>17</td><td style='text-align: center; word-wrap: break-word;'>13 #############</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>18</td><td style='text-align: center; word-wrap: break-word;'>5 #####</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>19</td><td style='text-align: center; word-wrap: break-word;'>2 ##</td></tr></table>

就训练 epoch 而言，级联相关的速度是 quickprop 的 5 倍，是标准反向传播的 10 倍，同时构建的网络复杂度大致相同（15 个隐层单元）。然而，就串行机器上的实际计算量而言，速度提升远大于这些数字所暗示的。在反向传播和 quickprop 中，每个训练样本都需要通过网络中所有连接进行一次前向传播和一次反向传播；而级联相关只需要一次前向传播。此外，级联相关的许多 epoch 是在网络远小于其最终规模时运行的。最后，上述缓存策略使得可以避免为网络中不变化的部分重新计算单元值。

假设我们不以 epoch 为单位，而是以**连接交叉**（connection crossings）来衡量学习时间，其定义为通过网络前向传播激活值和反向传播误差值所需的乘加步骤数量。这种度量遗漏了一些计算步骤，但它是比比较不同大小的 epoch 或比较不同机器上的运行时间更准确的计算复杂度度量。Lang 和 Witbrock 的 20,000 个 backprop epoch 结果需要约 11 亿次连接交叉。他们在同一网络上使用 8000 个 quickprop epoch 的解决方案需要约 4.38 亿次交叉。使用 8 个候选单元池的级联相关平均运行需要约 1900 万次交叉——比 quickprop 快 23 倍，比标准反向传播快 50 倍。如果使用更小的候选单元池，（在串行机器上的）速度提升会更大，但产生的网络可能会稍大一些。

图 2b 展示了当输入在 X-Y 平面上扫描时，由级联相关构建的 12 隐层单元网络的输出。该网络正确分类了全部 194 个训练点。我们可以看到，它在螺旋大约前 1.5 圈的范围内平滑地进行插值，但在更外层，训练点间距较大时，变得有些凹凸不平。这种“感受野（receptive field）”图与 Lang 和 Witbrock 使用反向传播获得的结果类似，但要稍微平滑一些。

### 2.2 N 输入奇偶校验问题（N-INPUT PARITY）

由于奇偶校验（parity）一直是其他研究者中流行的基准测试，我们在 N 从 2 到 8 的 N 输入奇偶校验问题上运行了级联相关。最佳结果使用 S 型输出单元和输出为加权输入总和的高斯函数（Gaussian function）的隐层单元获得。基于每个 N 值的五次试验，我们的结果如下：

<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>N</td><td style='text-align: center; word-wrap: break-word;'>案例数（Cases）</td><td style='text-align: center; word-wrap: break-word;'>隐层单元数（Hidden Units）</td><td style='text-align: center; word-wrap: break-word;'>平均 Epoch 数（Average Epochs）</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>2</td><td style='text-align: center; word-wrap: break-word;'>4</td><td style='text-align: center; word-wrap: break-word;'>1</td><td style='text-align: center; word-wrap: break-word;'>24</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>3</td><td style='text-align: center; word-wrap: break-word;'>8</td><td style='text-align: center; word-wrap: break-word;'>1</td><td style='text-align: center; word-wrap: break-word;'>32</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>4</td><td style='text-align: center; word-wrap: break-word;'>16</td><td style='text-align: center; word-wrap: break-word;'>2</td><td style='text-align: center; word-wrap: break-word;'>66</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>5</td><td style='text-align: center; word-wrap: break-word;'>32</td><td style='text-align: center; word-wrap: break-word;'>2-3</td><td style='text-align: center; word-wrap: break-word;'>142</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>6</td><td style='text-align: center; word-wrap: break-word;'>64</td><td style='text-align: center; word-wrap: break-word;'>3</td><td style='text-align: center; word-wrap: break-word;'>161</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>7</td><td style='text-align: center; word-wrap: break-word;'>128</td><td style='text-align: center; word-wrap: break-word;'>4-5</td><td style='text-align: center; word-wrap: break-word;'>292</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>8</td><td style='text-align: center; word-wrap: break-word;'>256</td><td style='text-align: center; word-wrap: break-word;'>4-5</td><td style='text-align: center; word-wrap: break-word;'>357</td></tr></table>

作为一个粗略的比较，Tesauro 和 Janssens [Tesauro, 1988] 报告称标准反向传播解决 8 输入奇偶校验问题大约需要 2000 个 epoch。在他们的研究中，使用了 2N 个隐层单元。级联相关可以用少于 N 个隐层单元解决该问题，因为它使用了捷径连接（short-cut connections）。

作为泛化能力（generalization）的测试，我们在 10 输入奇偶校验问题上运行了少量级联相关试验，对 1024 个模式中的 50% 或 25% 进行训练，并在其余模式上测试。构建的隐层单元数量从 4 到 7 不等，训练时间从 276 个 epoch 到 551 个不等。当在一半模式上训练时，测试集上的平均正确率为 96%；当在四分之一模式上训练时，测试集平均正确率为 90%。注意，最近邻算法（nearest neighbor algorithm）几乎会把所有测试集案例都搞错。

### 基准测试总结

#### 1. 双螺旋问题（Two-Spirals Problem）

这是论文的主要基准，对反向传播家族算法而言极其困难。

| 对比项 | 级联相关 | 其他方法（对比基准） |
|--------|----------|----------------------|
| **平均 Epoch** | **1,700** | Quickprop: 8,000；标准反向传播: 20,000 |
| **隐层单元数** | 平均 **15.2** 个（12–19 个） | Lang 和 Witbrock 的 2-5-5-5-1 网络共 15 个隐层单元 |
| **连接交叉次数** | 约 **1,900 万** | Quickprop: 4.38 亿；标准反向传播: 11 亿 |

- **成功率**：100 次试验全部成功。
- **速度提升**：以 epoch 计，级联相关约为 Quickprop 的 **5 倍**、标准反向传播的 **10 倍**；以实际计算量（连接交叉）计，分别提升至约 **23 倍** 和 **50 倍**。
- **网络特性**：级联相关仅通过前向传播完成训练，且许多 epoch 在网络规模尚小时运行，并可缓存不变化部分的单元值，因此实际加速比更显著。

#### 2. N 输入奇偶校验问题（N-Input Parity）

使用 S 型输出单元 + 高斯隐层单元取得最佳效果。

| N | 训练样本数 | 隐层单元数 | 平均 Epoch |
|---|-----------|-----------|-----------|
| 2 | 4 | 1 | 24 |
| 3 | 8 | 1 | 32 |
| 4 | 16 | 2 | 66 |
| 5 | 32 | 2–3 | 142 |
| 6 | 64 | 3 | 161 |
| 7 | 128 | 4–5 | 292 |
| 8 | 256 | 4–5 | 357 |

- **对比**：Tesauro 和 Janssens 报告称标准反向传播解决 8 输入奇偶校验约需 **2,000 epoch**，且使用 **2N 个隐层单元**；级联相关仅需 **4–5 个隐层单元**（少于 N），得益于其 shortcut 连接。
- **泛化能力**（10 输入奇偶校验）：
  - 用 **50%** 样本训练，测试集平均正确率 **96%**；
  - 用 **25%** 样本训练，测试集平均正确率 **90%**。


## 3 讨论（DISCUSSION）

我们认为，级联相关算法相比目前使用的网络学习算法具有以下优势：

- 无需事先猜测网络的大小、深度和连接模式。一个相当小（尽管不一定最优）的网络可以被自动构建出来，甚至可能混合使用多种单元类型。

- 级联相关学习速度快。在反向传播中，隐层单元在 settled 到不同有用角色之前会进行复杂的“舞蹈”；而在级联相关中，每个单元面对的是一个固定的问题，可以果断地解决该问题。对于我们迄今为止研究的问题，以 epoch 衡量的学习时间大致随 $N \log N$ 增长，其中 $N$ 是最终解决问题所需的隐层单元数量。

- 级联相关可以构建深度网络（高阶特征检测器），而不会像深度反向传播网络那样出现急剧的减速。

- 级联相关适用于增量学习（incremental learning），即在已训练好的网络上添加新信息。一旦构建完成，特征检测器永远不会被“蚕食（cannibalized）”。从那一刻起，它就可用于产生输出或更复杂的特征。

- 在任何给定时间，我们只训练网络中的一层权重。网络的其余部分是恒定的，因此结果可以被缓存。

- 永远不需要通过网络连接反向传播误差信号。单一的残余误差信号可以广播给所有候选单元。加权连接仅以单一方向传输信号，消除了这些网络与生物突触之间的一个差异。

- 候选单元之间除了选出优胜者外互不干扰。每个候选单元看到相同的输入和误差信号。这种有限的通信使得该架构对并行实现具有吸引力。

## 4 与其他工作的关系（RELATION TO OTHER WORK）

级联相关与旧有学习架构的主要区别在于：隐层单元的动态创建、我们将新单元以多层方式堆叠（具有固定的输出层）、单元被加入网络时即被冻结，以及我们通过爬山法（hill-climbing）训练新单元以最大化其与残余误差的相关性。最有趣的发现是，通过一次只训练一个单元而不是同时训练整个网络，我们可以显著加快学习过程，同时仍然创建一个规模相当小且泛化良好的网络。

许多研究者 [Ash, 1989, Moody, 1989] 研究了在单层网络中学习过程中添加新单元或感受野（receptive fields）的网络。虽然单层系统非常适合某些问题，但这些系统无法创建高阶特征检测器来组合现有单元的输出。构建特征检测器然后将其冻结的想法，部分受到 Waibel 关于模块化网络（modular networks）工作 [Waibel, 1989] 的启发，但在他的模型中，子网络的结构必须在开始学习之前固定。

据我们所知，只有少数尝试在训练过程中构建多层网络。我们决定研究每个单元都能看到所有先前存在单元的模型，在某种程度上受到卡内基梅隆大学 Merrick Furst 和 Jeff Jackson 关于逐步加深阈值逻辑模型（progressively deepening threshold-logic models）工作的启发。（他们目前并未积极开展这方面研究。）Gallant [Gallant, 1986] 简要提及了一种逐步加深的感知机模型（他的“倒金字塔”模型），其中单元在安装后被冻结。然而，他的大部分研究工作集中在随机生成新隐层单元而非通过有意的训练过程生成新隐层单元的模型上。Tenorio 和 Lee [Tenorio, 1989] 的 SONN 模型构建了一个多层拓扑结构以适应手头的问题。他们的算法将新的双输入单元放置在随机选择的位置，使用模拟退火搜索来只保留最有用的单元——一种与我们非常不同的方法。

## 致谢（Acknowledgments）

我们要感谢 Merrick Furst、Paul Gleichauf 和 David Touretzky 提出的宝贵问题，这些问题帮助塑造了这项工作。本研究部分由国家科学基金会（合同号 EET-8716324）资助，部分由国防高级研究计划局（合同号 F33615-87-C-1499）资助。

## 参考文献（References）

[Ash, 1989] Ash, T. (1989) “Dynamic Node Creation in Back-Propagation Networks”, Technical Report 8901, Institute for Cognitive Science, University of California, San Diego.

[Fahlman, 1988] Fahlman, S. E. (1988) “Faster-Learning Variations on Back-Propagation: An Empirical Study” in Proceedings of the 1988 Connectionist Models Summer School, Morgan Kaufmann.

[Gallant, 1986] Gallant, S. I. (1986) "Three Constructive Algorithms for Network Learning" in Proceedings, 8th Annual Conference of the Cognitive Science Society.

[Lang, 1988] Lang, K. J. and Witbrock, M. J. (1988) “Learning to Tell Two Spirals Apart” in Proceedings of the 1988 Connectionist Models Summer School, Morgan Kaufmann.

[Moody, 1989] Moody, J. (1989) “Fast Learning in Multi-Resolution Hierarchies” in D. S. Touretzky (ed.), Advances in Neural Information Processing Systems 1, Morgan Kaufmann.

[Rumelhart, 1986] Rumelhart, D. E., Hinton, G. E., and Williams, R. J. (1986) “Learning Internal Representations by Error Propagation” in Rumelhart, D. E. and McClelland, J. L., Parallel Distributed Processing: Explorations in the Microstructure of Cognition, MIT Press.

[Tenorio, 1989] Tenorio, M. F., and Lee, W. T. (1989) “Self-Organizing Neural Nets for the Identification Problem” in D. S. Touretzky (ed.), Advances in Neural Information Processing Systems 1, Morgan Kaufmann.

[Tesauro, 1988] Tesauro, G. and Janssens, B. (1988) “Scaling Relations in Back-Propagation Learning” in Complex Systems 2 39-44.

[Waibel, 1989] Waibel, A. (1989) “Consonant Recognition by Modular Construction of Large Phonemic Time-Delay Neural Networks” in D. S. Touretzky (ed.), Advances in Neural Information Processing Systems 1, Morgan Kaufmann.
