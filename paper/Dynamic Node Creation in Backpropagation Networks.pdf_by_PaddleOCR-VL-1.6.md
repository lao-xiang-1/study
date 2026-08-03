# 反向传播网络中的动态节点创建

TIMUR ASH

本文介绍了一种名为动态节点创建（Dynamic Node Creation, DNC）的新方法，该方法能够自动增长BP网络，直到目标问题被解决。DNC依次向网络的隐藏层逐个添加节点，直到达到所需的近似精度。本文展示了奇偶校验、对称性、二进制加法以及编码器问题的仿真结果。该过程在许多情况下都能找到已知的最小拓扑结构，并且与最小值相差不超过三个节点。寻找解的计算开销与训练具有相同最终拓扑结构的普通BP网络相当。以少于解决问题所需节点数开始实际上似乎有助于找到解。该方法对每一个尝试的问题都能得出解。

## 引言

必须考虑网络架构在训练过程中的演化。这在实际层面上很重要，因为它赋予网络更大的灵活性来适应其功能并找到解。从生物学角度来看，这也很有意思。迄今为止，固定的神经网络架构一直不适合模拟大脑随时间变化的拓扑结构（Aoki et al., 1988）。但是，反向传播算法（Rumelhart et al., 1985; Werbos, 1974）的发展及其隐藏层使得不仅能够改变权重和偏置向量的值，还能改变它们的维度。

## 过去的架构选择方式
在很大程度上，架构的选择仍然很困难。

在大多数情况下，网络被丢弃的原因有一两个。如果选择的拓扑结构太小（从信息容量的意义上说），期望的输入到输出的映射就无法以令人满意的精度学习。过大的网络常常会过拟合数据，以追求更低的误差（专注于有限训练集的统计特性）。在极端情况下，网络只是记住了它接触到的所有模式，对新输入的反应很差。

1. **从大网络开始，不断减小到合适大小**
理想情况下，我们需要的是一个足够大以学习映射、同时又尽可能小以获得良好泛化能力的网络（Huyser & Horowitz, 1988）。找到这种网络有两种通用方法。一种是使用比所需更大的拓扑结构并训练它直到找到映射。之后，如果网络中的某些元素没有被积极使用，就将它们剪枝掉（例如，如果某个特定权重的值始终非常接近零，就可以在不影哬网络的情况下将其消除）。Rumelhart在这方面的工作（Rumelhart, 1988）试图最小化输出节点上的误差以及节点和权重数量的函数。Hanson & Pratt（1989）表明，可以用这种方法获得更小的网络规模，但代价是收敛速度。剪枝也可以通过分析待移除元素的"相关性"来实现（Mozer & Smolensky, 1989）。

2. **从一个小网络开始，增长额外的节点和权重直到找到解**
另一种方法是从一个小网络开始，增长额外的节点和权重直到找到解。在其他神经网络模型中，训练期间添加节点的例子可以在用于模式分类任务的受限库仑能量（Restricted Coulomb Energy, RCE）网络中看到（Reilly et al., 1987）。还有人开发了一种基于联结主义的的学习系统，通过在需要形成新概念时添加节点（"招募"）（Diederich, 1988）。也有人尝试过结合增长/剪枝的算法（Hirose et al., 1989）。

剪枝方法存在一些缺点。由于大部分训练时间花费在一个比必要更大的网络上，这种方法在计算上是浪费的。在实践中，由于解是未知的，通常会选择比所需大得多的网络作为起点（Sietma & Dow, 1988）。此外，许多具有不同拓扑结构的网络能够实现相同的映射。由于剪枝方法从大网络开始，它可能会卡在其中一个中等规模的解中。

有两种方法来训练动态增长额外节点或权重的网络。在这两种方法中，激活都向前传播通过整个网络（包括新添加的元素）。在反向传播过程中，一种方法是冻结现有网络，只通过反向传播训练新元素。另一种方法允许对整个网络进行完全重新训练。

冻结方法的优点是引入新元素时所需的重新训练工作量相对较小。然而，这种方法通常找不到期望的解。当引入额外的自由度时（例如，通过向网络添加一个新权重），保持现有网络值不变只允许在权重空间的一个仿射子集中找到解（见图1）。可以引入额外的自由度（额外的节点或权重）来允许这个仿射子集通过全局最小点（实际上，移动一些先前固定的坐标）。但由于涉及部分重复工作（即，需要两个或更多权重来确定单个维度的值），这样的网络不可能在规模上是最小的。

剩下的替代方案允许在添加新元素后重新训练整个网络。动态节点创建（DNC）方法就是为了在网络训练期间***向隐藏层添加节点***而开发的。用此过程生长一个新节点后，进行常规的BP训练，直到学习完期望的映射，或需要添加另一个节点。

#flashpaper
##### 新增节点后，原来已有的节点还要继续训练
?
<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//b37f1ac6-1b89-4f22-b9e0-f41bccafecaf/markdown_3/imgs/img_in_image_box_338_224_785_558.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-02T08%3A18%3A09Z%2F-1%2F%2Fd2fd2d7ee7eb431458f21d28b45bbf928dda9c1e3c8ce4882044ac81d20bcbd2" alt="Image" width="36%" /></div>

*图1. $P_{1}$ 表示由 $W_{1} \times W_{2}$ 定义的平面中误差最低的点。如果引入另一个维度（$W_{3}$），全局最优变为 $P_{2}$。但是冻结 $W_{1}$ 和 $W_{2}$ 的值只允许找到通过 $P_{1}$ 的直线上的解。推广到更高维度；只能在权重空间的特定仿射子集中找到解。*
<!--SR:!2026-08-05,3,250-->

---

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//b37f1ac6-1b89-4f22-b9e0-f41bccafecaf/markdown_3/imgs/img_in_chart_box_342_730_797_1076.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-02T08%3A18%3A09Z%2F-1%2F%2F83fc7aa5c7d97eaf8d510783effb8f059b301f1b5d8a35225ceb5140b75f612a" alt="Image" width="37%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图2. 当检测到平均平方误差曲线趋于平缓时，触发添加单个新隐藏节点。</div> </div>


## 模型

所研究的网络***只有一个***隐藏层，层与层之间完全前馈互连。输入层和输出层之间没有直接连接。在输入层之上的所有层都使用逻辑激活函数（Rumelhart et al., 1985）：
$$ \mathrm{Output}\ (I)=\frac{1}{1+\mathrm{e}^{-I}} $$

| 名称 | 定义 |
| :--- | :--- |
| $t_{0}$ | 上次向隐藏层添加节点的时间（以训练次数计，初始为0） |
| $t$ | 当前时间 |
| $w$ | 确定触发斜率的窗口宽度（以训练次数计） |
| $a_{t}$ | 时间 t 时每个输出节点的平均平方误差 |
| $m_{t}$ | 时间 t 时任意输出节点在任意模式上的最大平方误差 |
| $C_{a}$ | 平均平方误差的期望截止值 |
| $C_{m}$ | 最大平方误差的期望截止值 |
| $\Delta_{T}$ | 触发斜率——误差曲线平坦度的度量，低于此值应添加新节点 |
<div style="text-align: center;"><div style="text-align: center;color: lightgreen">表 I. DNC中涉及的变量</div> </div>

这种架构并不限制网络可以找到的映射类别。事实上，最近的结果（Hecht-Neilsen, 1988; Hornik et al., 1988）表明，具有三层的前馈网络可以用隐藏层中有限数量的节点以任意选定的精度建模任何感兴趣的函数。然而，这些有力的理论结果并未回答一个特定问题需要多少个节点，以及BP过程能否学习该映射的问题。关于多隐藏层架构的优势也存在一个悬而未决的问题。尽管这种拓扑结构在理论上并不更强大，但在实践中它们常常产生更小（且更稀疏）的网络，这些网络能很快收敛到解。

在DNC中，当平均误差曲线开始过快地趋于平缓时，会添加一个新的隐藏节点（见图2）。计算最近 $w$ 次试验中平方误差的下降量与上次添加节点时平方误差的比值。当该值低于用户定义的"触发斜率" $\Delta_{T}$ 时，就会添加一个新节点。

数学上表达，如果满足以下两个条件，则应添加一个新节点（术语定义见表I）：

1. 误差曲线已经相当平缓
$$ \frac{a_{t}-a_{t-w}}{a_{to}}<\Delta_{T} $$
2. 保证上面式子中的所有误差项都针对相同的拓扑结构
$$ t-w\ge t_{0} $$ 
另一个结果是，新节点只能在至少 $w$ 次试验之后添加。新节点接收来自输入的完整连接，并连接到所有输出。

当映射学习到用户指定的精度时，节点增长被关闭。如果不禁用节点增长，算法将继续添加节点，只是为了在输出上获得稍微更高的精度。禁用节点增长后，仍可进行正常的BP微调以提高精度。以下关系决定何时停止添加节点。注意：$C_{a}$ 和 $C_{m}$ 都是用户指定的。

 $$ a_{t}\leq C_{a}\qquad\mathrm{且}\qquad m_{t}\leq C_{m} $$ 

这两个常数允许为整体误差和最坏情况误差分别设置期望的精度水平。

| 名称 | 输入 | 输出 | 已知解（隐藏单元数量） |
| :--- | :--- | :--- | :--- |
| 编码器问题 (ENC) | N位二进制向量，1位为1 | 与输入相同 | $\log_2 N$ |
| 对称性 (SYM) | N位二进制向量 | 对称则为1，不对称则为0 | 2 |
| 奇偶校验 (PAR) | N位二进制向量 | 1的个数为奇数则为1，否则为0 | N |
| 二进制加法 (ADD) | 2N位二进制向量 | N位结果和1位进位 | 单层隐藏层无已知解 |
<div style="text-align: center;"><div style="text-align: center;color: lightgreen">表 II. 测试问题及隐藏层单元数量的经验上界</div> </div>

## 实现与测试问题

本研究的仿真器用标准C编写，在Unix操作系统下的多用户Sun 4上运行。所有权重初始化为-0.1666到+0.1666之间的小随机值。示例问题选自Rumelhart & McClelland（1986，第8章）。这些问题包括编码器问题、对称性、奇偶校验以及带进位的二进制加法。每次运行时，网络从隐藏层的一个节点开始。使用的学习率恒为0.5，动量值为0.9。整个过程中使用的触发斜率值为0.05，窗口宽度（$\omega$）为1000次完整遍历训练集。所有运行中 $C_a$ 和 $C_m$ 的值分别为0.001和0.01。表II总结了测试的问题以及隐藏层所需节点数量的已知上界。

选择了上述各类问题的代表进行研究。每种情况下，名称后面的数字指的是表II输入列中提到的N。这些问题包括ENC16、SYM4、SYM6、PAR2（异或）、PAR4、PAR5、PAR6、ADD2和ADD3。使用穷举的所有可能输入数据来训练每个网络。一次完整的数据呈现被视为一次时间试验。

## 结果

动态节点创建的基本有效性已通过所有测试问题得到验证。在每次运行中，都发现了手头问题的解。在所有情况下都找到了相当小的解。事实上，对于所有具有已知上界的问题都找到了最小解（见表II）。在ENC16的情况下，问题用三个隐藏节点而不是预期的四个解决。并非每次运行都产生最小拓扑结构。PAR6的一次运行（8次中的1次）需要九个节点而不是六个来学习映射，另一次需要七个。同样，PAR2在5次尝试中有2次需要三个节点而不是最小的两个，尽管触发斜率使用了0.01（而不是0.05）的值。这种额外的增长部分是由于 $\Delta_T$ 和 $w$ 的选择不当。如果局部最小值和全局最小值在权重空间中被分隔开，更多的隐藏单元可能被"置换"（White, 1987）以产生多个全局最小值并促进解的发现。这可能有助于解释为什么比最小规模稍大的网络通常更容易找到解。ADD2始终用四个节点解决，ADD3用七个节点解决。应检查这些结果是否代表单层隐藏层二进制加法的最小解。

计算开销与使用具有"正确"隐藏节点数量的BP网络学习相比具有竞争力。由于开销随网络规模的增加而几何增长，DNC受益于最初使用较小的网络。图3展示了与等效BP网络的训练开销比较。可以看出，DNC网络与BP网络具有竞争力，并且在找到良好拓扑结构的问题上给出了解。

预计DNC会比BP更昂贵。但即使在最坏的情况下，差异也只有百分之四十。在某些情况下，增长过程实际上比BP更便宜。这些结果是在每次都将DNC网络天真地从一个隐藏节点开始的情况下获得的。在实践中，针对更复杂任务的网络可以用更多的隐藏节点初始化，并允许从那里增长。同样重要的是要注意，尽管经过多次尝试和延长学习时间，BP仍无法找到几个问题的最小拓扑解。BP网络尝试了0.1、0.25和0.5的学习率，并允许运行50,000次试验。

单调递减平均误差的基本假设对大多数测试问题都成立。当它不成立时，误差曲线中的尖峰并未引起触发新节点增长的问题（见图4中的节点4）。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//8ea367ce-0415-4e9b-afc1-bbdafdd4446a/markdown_2/imgs/img_in_chart_box_401_224_877_563.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-02T08%3A17%3A16Z%2F-1%2F%2F031915f17a4a1ee77ddfc33d576b9a38b1af68e45be440fff175feb9ed615c29" alt="Image" width="38%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图3. 使用DNC和普通BP学习映射所需的计算工作量（以浮点乘法次数计）。显示的结果是DNC找到的最小拓扑结构。浅色条代表DNC，深色条代表BP。注意：SYM4、PAR5和PAR6的BP解在50,000次试验后未找到。</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//8ea367ce-0415-4e9b-afc1-bbdafdd4446a/markdown_2/imgs/img_in_chart_box_404_762_892_1129.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-02T08%3A17%3A16Z%2F-1%2F%2F05bbd36b840be813a8d12ed29468476f0fd65aa465d2a985715179583eb23620" alt="Image" width="39%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图4. ADD3的最佳情况平方误差图。虚线竖线表示新节点的创建。节点增长不受 $a_t$ 曲线中尖峰的影响（见节点4）。</div> </div>


**向网络添加新节点并未降低其性能。当添加新节点时，它连接到网络其余部分的权重很小。现有网络中的活跃权重通常至少大一个数量级。新节点最初既不帮助也不妨碍现有网络的活动。它随着权重的变化慢慢成长为新功能，并降低平均误差。**

## 讨论

理解训练较小网络对找到更大解的影响很重要。这可以通过考虑DNC网络在学习映射之前以其最大架构花费了多少次试验（完整的数据集呈现）来评估。如果低维训练使它偏离了实际解，这个数字预计会比训练具有相同拓扑结构的常规BP网络的时间更长。如果训练没有效果（导致权重的本质上随机分布），该数字应该与BP训练时间大致相同。事实上，在测试的每种情况下，DNC网络在最大规模上花费的训练时间都比其BP对应物少（见图5）。这表明在低维空间中的初始训练实际上有助于找到解。这也得到了以下事实的支持：SYM4、PAR5和PAR6的最小解是通过增长网络找到的，但在使用正常BP学习的反复尝试中未能找到。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//8ea367ce-0415-4e9b-afc1-bbdafdd4446a/markdown_3/imgs/img_in_chart_box_349_833_820_1168.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-08-02T08%3A17%3A18Z%2F-1%2F%2F84555f473640c2165983fbf279d66a0c09b2163be168a44df6a2dd1b7d013074" alt="Image" width="38%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图5. 在学习映射之前，以最终架构进行的试验次数。浅色条代表DNC，深色条代表BP。在所有情况下，低维DNC训练都有助于找到解。注意：SYM4、PAR5、PAR6的BP解在50,000次试验后未找到。</div> </div>


"无穷范数" $m_{t}$ 是决定映射何时被学习的重要标准。在所选的测试问题中，网络为了降低整体平均误差，至少在一个模式中完全弄错一个输出节点，这种情况非常常见。这种"牺牲一个为了多数的利益"的行为可以通过观察 $m_{t}$ 来发现（见图4）。$a_{t}$ 和 $m_{t}$ 的低水平应该是良好网络性能的指标。这个目标可能并非对所有问题都可能实现。它只会在本质上无噪声、确定性的环境中起作用。噪声数据也可能导致 $a_{t}$ 和 $m_{t}$ 之间的某种反向关系（即，强制降低平均误差可能导致最大误差上升，反之亦然）。

同样重要的是要注意，平均误差曲线不应使用训练样本来计算。应使用另一个测试集（从与训练集相同的总体中抽取）来计算误差值。对于测试的布尔函数来说，这不是问题，因为每种情况下都使用了完整的训练集。由于训练示例代表了所有可能数据点的整个空间，网络正在学习一个精确的关系。一般来说，当基于测试集的平均平方误差曲线开始上升时，应停止网络的训练。超过这一点，就会发生过训练，网络的泛化能力将下降。

在至少一种情况下（PAR4），DNC和BP网络观察到类似的误差曲线行为。图6a中曲线的形状基本上是图6b中曲线的压缩。两个网络的前2000次试验显示 $a_{t}$ 和 $m_{t}$ 值约为0.25，对应输出节点的误差为0.5。这表明只学习了奇偶校验函数的平均值（它在一半时间开启，另一半时间关闭）。对于DNC网络来说，这是可以理解的，因为它可能无法用其简单的初始架构建模更复杂的函数。但BP也首先做出了同样粗略的近似。它先尝试最简单（线性）的解，然后再进行正确的解，尽管从一开始就有所有四个隐藏节点和相关权重可用。

目前仍不清楚DNC找到的解是否与常规BP找到的解相似。尽管上述图表似乎表明了这一点，我们尚未对此进行调查。一个类似的问题是关于泛化的。如果DNC找到的网络类型与BP找到的不同，它们是否泛化得更好？由于每个问题都使用了完整的训练集，没有机会用当前的问题来测试这一点。

### 开放问题

本研究提出了几个值得进一步关注的问题。一个问题涉及所使用的回溯窗口（w）的选择。对于测试的布尔问题，1000次试验的宽度效果很好。然而，在某些运行中，这不足以防止超出已知最小值的额外节点增长。这可能是因为BP误差曲线在快速下降之前常常看起来几乎是平坦的很长一段时间。这类似于博弈中前瞻问题的地平线效应。通过其他不需要误差值回溯历史的平坦度启发式方法，可能可以完全避免这个问题。可能可以使用某种类型的衰减窗口积分器方程。然而，有人指出（G. Cottrell, 个人通信, 1988），方程(2)具有在训练初期当剩余绝对误差最高时更快添加节点的理想特性。

本仿真中的变化单位是一个完全前向和反向连接的节点。选择这一点部分是因为节点是连接通过的瓶颈，在某种意义上比单个权重更重要（Kruschke, 1988）。它还允许为仿真器编写非常高效的代码。一般来说，没有理由不使用单个权重和/或稀疏连接的节点来添加到网络中。增长方法找到的某些解在节点数量上是最小的，但在权重数量上不是。添加单个权重将导致更细粒度的增长（一次添加一个自由度而不是多个），并可能产生不那么复杂的网络。一旦找到解，也可以使用各种剪枝程序来消除未使用的权重。

添加单个权重还可以允许对任意拓扑和互连方案进行建模。一旦考虑到非均匀网络，开发算法来决定在何处以及何时添加新元素就变得很重要。我们目前正在进行研究，以确定具有多个隐藏层的网络中的一般节点分配策略。使用类似大脑拓扑约束和类似BP更新规则的工作已经证明对某些感知分类任务很有前景（Honavar & Uhr, 1988a, b）。这种方法根据识别每类对象的性能，向精心设计的分层网络添加新元素。

本文使用的测试问题都是离散（布尔）性质的。DNC已成功应用于实值云图像数据的压缩（McInerney, 1989）。然而，还需要进一步的工作来确定DNC对建模连续函数的有效性，以及其在构建更大网络中的实用性。

## 参考文献

Aoki, Chiye, Siekevitz & Philip (1988) 大脑发育中的可塑性. Scientific American, 259, 56–64.

Diederich, J. (1988) 知识密集型招募学习, 报告 TR-88-010. International Computer Science Institute.

Hanson, S.J. & Pratt, L.Y. (1989) 比较反向传播最小网络构造的偏差, CSL报告 36. Princeton University Cognitive Science Laboratory.

Hecht-Nielsen, R. (1989) 反向传播神经网络理论, Proceedings of the International Joint Conference on Neural Networks, I, 593-611.

Hirose, Y., Yamashita, K. & Hijiya, S. (1989) 改变隐藏单元数量的反向传播算法, 海报展示于 International Joint Conference on Neural Networks. Washington, DC, June.

Honavar, V. & Uhr, L. (1988a) 一个通过生成以及重新加权其连接来学习感知的神经元样单元网络, Computer Sciences Technical Report 793. University of Wisconsin, Madison.

Honavar, V. & Uhr, L. (1988b) 实验结果表明生成、局部感受野和全局收敛改善了联结主义网络中的感知学习, Computer Sciences Technical Report 805. University of Wisconsin, Madison.

Hornik, K., Stinchcombe, M. & White, H. (1988) 多层前馈网络是通用逼近器. Neural Networks, 1.

Huyser, K.A. & Horowitz, M.A. (1988) 数字函数中的泛化. 摘要发表于 Neural Networks, 1 (Suppl. 1), 101.

Kruschke, J.K. (1988) 在反向传播网络的隐藏层中创建局部和分布式瓶颈. In D. S. Touretzky (Ed.) Proceedings of the 1988 Connectionist Models Summer School, pp. 120–126. Los Altos, CA: Morgan Kaufmann.

McInerney, J.M. (1989) 用于图像压缩的动态节点创建, 初步手稿. Department of Computer Science and Engineering, University of California at San Diego.

Mozer, M.C. & Smolensky, P. (1989) 使用相关性自动减小网络规模. Connection Science, 1, 3–16.

Minsky, M. & Papert, S. (1988) Perceptrons (Expanded Edition). Cambridge, MA: MIT Press.

Reilly, D.L., Scofield, C., Elbaum, C. & Cooper, L.N. (1987) 由多个学习模块组成的学习系统架构. Proceedings of the IEEE First International Conference on Neural Networks. San Diego.

Rumelhart, D.E. (1988) 并行分布式处理. Plenary Lecture presented at the IEEE International Conference on Neural Networks. San Diego, July.

Rumelhart, D.E. & McClelland, J.L. (1986) Parallel Distributed Processing: Explorations in the Microstructure of Cognition, Vol. 1. Cambridge, MA: MIT Press.

Rumelhart, D.E., Hinton, G.E. & Williams, R.J. (1985) 通过误差传播学习内部表示, ICS报告 8506. Institute for Cognitive Science, University of California at San Diego.

Sietma, J. & Dow, R. (1988) 神经网络剪枝——为何及如何. Proceedings of IEEE International Conference on Neural Networks, I, 325–333.

Werbos, P.J. (1974) 超越回归：行为科学中预测和分析的新工具. 博士论文, 应用数学, Harvard University.

White, H. (1987) 反向传播的一些渐近结果. Proceedings of the IEEE First International Conference on Neural Networks, III, 261–266.

