Article

https://doi.org/10.1038/s41593-023-01514-1

# Inferring neural activity before plasticity as a foundation for learning beyond backpropagation

### Advantages of prospective configuration: reduced interference and faster learning

Here, we quantify interference in the above scenario and demonstrate how reduced interference translates into an advantage in performance. In all simulations in the main text, prospective configuration is implemented in predictive coding networks (other energy-based models are considered in the Supplementary Notes, Section 2.1). We also compare the performance of predictive coding networks against artificial neural networks (ANNs) trained with backpropagation because they are closely related, which makes the comparisons fair. In particular, although predictive coding networks include recurrent connections, they generate the same prediction for a given input (when inputs are constrained but outputs are not; Fig. 2a) as standard feedforward ANNs if their weights are set to corresponding values $^{12,14}$. Therefore, loss is the same function of weights in both models, so direct minimization of loss with gradient descent in predictive coding networks (which is not their natural way of training) would produce the same weight changes as backpropagation in ANNs. Hence, comparing predictive coding networks and backpropagation enables isolation of the effects of the learning algorithm (prospective configuration versus direct minimization of loss as in backpropagation).

【译文】在此，我们对上述场景中的干扰进行量化，并证明减少的干扰如何转化为性能优势。在主文本的所有模拟中，前瞻性配置均在预测编码网络中实现（其他能量基模型在补充注释第2.1节中考虑）。我们还将预测编码网络的性能与用反向传播训练的人工神经网络（ANN）进行比较，因为它们密切相关，这使得比较是公平的。特别是，尽管预测编码网络包含循环连接，但如果它们的权重被设置为相应的值，它们对于给定输入（当输入被约束但输出不被约束时；图2a）会生成与标准前馈ANN相同的预测$^{12,14}$。因此，在两个模型中，损失都是权重的相同函数，所以在预测编码网络中直接用梯度下降最小化损失（这不是它们自然的训练方式）会产生与ANN中反向传播相同的权重变化。因此，比较预测编码网络和反向传播可以将学习算法的效果分离出来（前瞻性配置 versus 反向传播中的直接最小化损失）。

In Fig. 3a, we compare the activity of output neurons in the example in Fig. 1 between backpropagation and prospective configuration. Initially both output neurons are active (top right), and the output should change toward a target in which one of the neurons is inactive (red vector). Learning with prospective configuration results in changes on the output (purple solid vector) that are aligned better with the target than those for backpropagation (purple dotted vector).

【译文】在图3a中，我们比较了图1示例中反向传播和前瞻性配置的输出神经元活动。初始时两个输出神经元都是活跃的（右上方），而输出应该朝着一个目标变化，即其中一个神经元变为不活跃（红色向量）。前瞻性配置的学习导致输出发生变化（紫色实线向量），其与目标的对齐程度优于反向传播（紫色虚线向量）。
在图3a中，我们比较了图1示例中反向传播和前瞻性配置的输出神经元活动。初始时两个输出神经元都是活跃的（右上方），而输出应该朝着一个目标变化，即其中一个神经元变为不活跃（红色向量）。前瞻性配置的学习导致输出发生变化（紫色实线向量），其与目标的对齐程度优于反向传播（紫色虚线向量）。

**对照理解**：

Following the first weight update, we simulate multiple iterations until the network is able to correctly predict the target. Here, 'iteration' refers to each time the agent is presented with stimuli and conducts one weight update because of the stimulus. Although the output from backpropagation can reach the target after multiple iterations, the output for the 'correct neuron' diverges from the target during learning and then comes back; this is a particularly undesired effect in biological learning, where networks can be 'tested' at any point during the learning process, because it may lead to incorrect decisions.

【译文】在第一次权重更新之后，我们模拟多次迭代，直到网络能够正确预测目标。在此，"迭代"指的是每次智能体被呈现刺激并因此执行一次权重更新。尽管反向传播的输出可以在多次迭代后达到目标，但"正确神经元"的输出在学习过程中会先偏离目标然后再回来；这在生物学习中是一个特别不理想的效果，因为在生物学习中网络可以在学习过程的任何时刻被"测试"，因为这可能导致错误的决策。

Although backpropagation modifies weights to directly reduce cost in the space of weights (that is, performs gradient descent), surprisingly, and rather subversively, it does not push the resulting output activity directly toward the target. To illustrate this, Fig. 3a visualizes the cost with contour lines. Changing the activity of output neurons according to the gradient of the cost would correspond to a change orthogonal to the contour lines, that is, that indicated by the red arrow. However, backpropagation changes the output in a different direction shown by a dashed arrow. Optimizing the weights independently, without considering the effect of updating other weights, leads to output activity not updating toward the target directly due to different weight updates to different layers interfering with each other. By contrast, prospective configuration considers the results of updating other weights by finding a desired configuration of neural activity first. Such a mechanism is missing in backpropagation but is natural in energy-based networks. Supplementary Fig. 2 shows a direct comparison of how these two models evolve in weight and output spaces during learning.
**尽管反向传播在权重空间中通过直接降低代价来修改权重（即执行梯度下降），但令人惊讶且颇具颠覆性的是，它并不会将结果输出活动直接推向目标。为了说明这一点，图3a用等高线可视化了代价。按照代价的梯度来改变输出神经元的活动，对应于垂直于等高线的变化，即红色箭头所示的方向。然而，反向传播以虚线箭头所示的不同方向改变了输出。独立地优化权重，而不考虑更新其他权重的效果，导致输出活动无法直接朝目标更新，因为不同层之间的权重更新会互相干扰。相比之下，前瞻性配置通过首先找到一个期望的神经活动配置，来考虑更新其他权重的结果。这种机制在反向传播中缺失，但在能量基网络中却很自然。补充图2直接比较了这两种模型在权重空间和输出空间中如何演化。**

Interference can be quantified by the angle between the direction of the target (from current output to target) and learning (from current output to output after learning, both measured without the target provided), and we define 'target alignment' as the cosine of this angle (Fig. 3b); hence, high interference corresponds to low target alignment (Fig. 3c).

【译文】干扰可以通过目标方向（从当前输出到目标）和学习方向（从当前输出到学习后的输出，两者都是在不提供目标的情况下测量的）之间的夹角来量化，我们将"目标对齐"定义为这个夹角的余弦值（图3b）；因此，高干扰对应于低目标对齐（图3c）。

It is useful to highlight that target alignment is affected little by the learning rate (Fig. 3d), demonstrating that the learning rate has little effect on the direction and trajectory that output neurons take. The difference in target alignment demonstrated in Fig. 3a is also present for deeper and larger (randomly generated) networks (Fig. 3e). When a network has no hidden layers, the target alignment is equal to 1 (Supplementary Notes, Section 2.4.1). The target alignment drops for backpropagation as the network gets deeper because changes in weights in one layer interfere with changes in other layers (Fig. 1), and the backpropagated errors do not lead to appropriate modification of weights in hidden layers (Supplementary Fig. 2). Because backpropagation modifies the weights in the direction reducing loss, it has positive target alignment for small learning rates but not necessarily

【译文】值得强调的是，目标对齐几乎不受学习率的影响（图3d），这表明学习率对输出神经元所取的方向和轨迹几乎没有影响。图3a中展示的目标对齐差异在更深、更大（随机生成）的网络中也存在（图3e）。当网络没有隐藏层时，目标对齐等于1（补充注释，第2.4.1节）。对于反向传播，随着网络变深，目标对齐会下降，因为一层的权重变化会干扰其他层的变化（图1），而且反向传播的误差不会导致隐藏层权重的适当修改（补充图2）。因为反向传播以降低损失的方向修改权重，所以对于较小的学习率它具有正的目标对齐，但不一定

<div style="text-align: center;"><div style="text-align: center;">a</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_0/imgs/img_in_chart_box_96_113_508_336.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A32Z%2F-1%2F%2F861700c52a9e512f2629e21ccd27ee33a57be47f3a8d70b0aacb40db0c5372e4" alt="Image" width="70%" /></div>


| 元素         | 含义                                   |
| ---------- | ------------------------------------ |
| 横坐标        | 预测听到水声（错误）                           |
| 纵坐标        | 预测闻到三文鱼（正确）                          |
| 初始状态（右上方）  | 两个输出神经元都活跃 → 熊同时"预测听到水声"和"预测闻到三文鱼"   |




<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_0/imgs/img_in_chart_box_523_107_776_336.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A32Z%2F-1%2F%2Fa7e93a97b8d8de41e82d7a00748baec0cfe62b90b7a1c11a918257f6275abd0f" alt="Image" width="21%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_0/imgs/img_in_chart_box_801_104_1117_332.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A32Z%2F-1%2F%2Fc4dbe4c228bb15d533c7dd76060cbfac5e70fa809f2299a8029acb9ab3007a85" alt="Image" width="26%" /></div>


<div style="text-align: center;"><div style="text-align: center;">e</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_0/imgs/img_in_chart_box_81_358_428_575.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A32Z%2F-1%2F%2Ff5deef1e0849e30d0675be67aa984cbb7fdb44699da3fa8168528b9bcab7b9fa" alt="Image" width="29%" /></div>


<div style="text-align: center;"><div style="text-align: center;">f</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_0/imgs/img_in_chart_box_446_360_671_575.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A32Z%2F-1%2F%2F891849703774f529322aab669e1cf2924b38ff7dbca035fab9c62434470417cd" alt="Image" width="18%" /></div>


<div style="text-align: center;"><div style="text-align: center;">h</div> </div>


<div style="text-align: center;"><div style="text-align: center;">g</div> </div>


<div style="text-align: center;"><div style="text-align: center;">Fig. 3 | Learning with prospective configuration changes the activity of output neurons in a direction more aligned toward the target. a, Simulation of the network from Fig. 1 showing changes in the correct and incorrect output neurons during training ('Iteration') trained with both learning rules. Here, learning with prospective configuration (purple solid vector) aligns better with the target (red vector) than learning with backpropagation (purple dashed vector). b, Interference can be quantified by 'target alignment', the cosine similarity of the direction of the target (red vector) and the direction of learning (purple vector). c, Higher target alignment indicates less interference and vice versa. d, The same experiment as in a repeated with a learning rate ranging from 0.005 to 0.5 represented by the size of the markers, where it is shown that the choice of learning rate changes the trajectories for both methods slightly, but the conclusion holds irrespective of the learning rate. e, Target alignment of randomly generated networks trained with both learning rules as a function</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_0/imgs/img_in_chart_box_694_357_907_573.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A32Z%2F-1%2F%2F7628ed4ac9992086855ac34d81788975511328707ba69e5a5a3e04e91e3e0844" alt="Image" width="17%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_0/imgs/img_in_chart_box_918_365_1121_575.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A32Z%2F-1%2F%2F1aa61dd0aff760c3410ec30c3a8677b4092f07b11f503bd3767d404b3dfc42d4" alt="Image" width="17%" /></div>


<div style="text-align: center;"><div style="text-align: center;">of depth of the network. Each symbol shows target alignment resulting from training on a single randomly generated pattern. f, Test error during training on the FashionMNIST $^{60}$ dataset containing images of clothing belonging to different categories for both learning rules with a deep neural network of 15 layers. Here, 'test error' refers to the ratio of incorrectly classified samples among all samples in the test set. g, Mean of the test error over training epochs (reflecting how fast test error drops) as a function of learning rate. Results in f and h are for the learning rates giving the minima of the corresponding curves in g, h, Mean of test error of other network depths. Each point is from a learning rate independently optimized for each learning rule in the corresponding setup of network depth. In e–h, prospective configuration demonstrates a notable advantage as the structure gets deeper. Each experiment in f–h was repeated with n = 3 random seeds. Error bars and bands represent the 68% confidence interval.</div> </div>


close to 1. By contrast, prospective configuration maintains a much higher value along the way. This higher target alignment of prospective configuration can be theoretically explained by the following: (1) there exists a close link between prospective configuration and an algorithm called target propagation $^{26}$ (shown in Supplementary Fig. 3 and Supplementary Notes, Section 2.2), and (2) under certain conditions, target propagation $^{26}$ has a target alignment of 1 (ref. 27; demonstrated in Supplementary Fig. 4 and Supplementary Notes, Section 2.4.2). Thus, the link with target propagation provides theoretical insight (with numerical验证) into why prospective configuration has a higher target alignment.

【译文】接近1。相比之下，前瞻性配置在此过程中始终保持高得多的值。前瞻性配置这种更高的目标对齐可以从理论上得到如下解释：（1）前瞻性配置与一种称为目标传播的算法$^{26}$之间存在密切联系（见补充图3和补充注释第2.2节），以及（2）在某些条件下，目标传播$^{26}$的目标对齐为1（参考文献27；在补充图4和补充注释第2.4.2节中证明）。因此，与目标传播的联系为"为什么前瞻性配置具有更高的目标对齐"提供了理论洞见（附数值验证）。

Higher target alignment directly translates to the efficiency of learning. Test error during training in a visual classification task with a deep neural network of 15 layers decreases faster for prospective configuration than for backpropagation (Fig. 3f).

【译文】更高的目标对齐直接转化为学习的效率。在视觉分类任务中，使用15层深度神经网络训练时，前瞻性配置的测试误差下降速度比反向传播更快（图3f）。

Throughout the data presented here, if learning rate is not presented in a plot, the plot corresponds to the best learning rate optimized independently for each rule under the setup via a grid search. The optimization target is either learning performance or similarity to experimental data (details can be found in the methods for each experiment). Thus, for example, Fig. 3f shows the test errors as training progress, with the learning rates optimized independently for each learning rule. The optimization target is the 'mean of test error' during training, reflecting how fast the test error decreases during training. Fig. 3g plots this mean of test error for different learning rates for both learning rules, and the learning rates giving the minima of the curves were used in Fig. 3f. Fig. 3h repeats the experiment on networks of other depths and shows the mean of the test error during training as a function of network depth. The mean error is higher for lower depths, as these networks are unable to learn the task, and for greater depths, as it takes longer to train deeper networks. Importantly, the gap between backpropagation and prospective configuration widens for deeper networks, paralleling the difference in target alignment. Efficient training with deeper networks is important for biological neural systems known to be deep, for example, the primate visual cortex $^{28}$.

【译文】在本文呈现的所有数据中，如果图中未展示学习率，则该图对应于在该设置下通过网格搜索为每种规则独立优化的最佳学习率。优化目标要么是学习性能，要么是与实验数据的相似性（详见每个实验的方法部分）。因此，例如，图3f展示了训练过程中的测试误差，其中学习率为每种学习规则独立优化。优化目标是训练期间"测试误差的均值"，反映了测试误差在训练期间下降的速度。图3g为两种学习规则绘制了不同学习率下的测试误差均值，而图3f中使用了使曲线达到最小值的学习率。图3h在其他深度的网络上重复了该实验，并展示了训练期间测试误差的均值随网络深度的变化。对于较低的深度，平均误差更高，因为这些网络无法学习该任务；对于较大的深度，平均误差也更高，因为训练更深的网络需要更长时间。重要的是，随着网络加深，反向传播与前瞻性配置之间的差距进一步扩大，这与目标对齐的差异平行。对更深网络的高效训练对于已知具有深度结构的生物神经系统来说很重要，例如灵长类动物的视觉皮层$^{28}$。

In Section 2.3 of the Supplementary Notes, we develop a formal theory of prospective configuration and provide further illustrations and analyses of its advantages. Supplementary Fig. 5 formally defines prospective configuration and demonstrates that it is indeed commonly observed in different energy-based networks. Supplementary Figs. 6 and 7 empirically verify and generalize the advantages expected from the theory and show that prospective configuration yields more accurate error allocation and less erratic weight modification, respectively.

【译文】在补充注释的第2.3节中，我们发展了前瞻性配置的正式理论，并提供了对其优势的进一步说明和分析。补充图5正式定义了前瞻性配置，并证明它确实在不同的能量基网络中普遍被观察到。补充图6和7从经验上验证并推广了理论所预期的优势，分别表明前瞻性配置能产生更准确的误差分配和更稳定的权重修改。

## Advantages of prospective configuration: effective learning in biologically relevant scenarios

Inspired by these advantages, we show empirically that prospective configuration indeed handles various learning problems that biological systems would face better than backpropagation. Because the field of machine learning has developed effective benchmarks for testing learning performance, we use variants of classic machine learning algorithms.

【译文】受这些优势的启发，我们从经验上证明前瞻性配置确实能比反向传播更好地处理生物系统将面临的各种学习问题。由于机器学习领域已经开发了用于测试学习性能的有效基准，我们使用了经典机器学习算法的变体。

<div style="text-align: center;"><div style="text-align: center;">a</div> </div>


Online learning

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_1/imgs/img_in_image_box_89_116_266_308.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A32Z%2F-1%2F%2F999e9e5381202404e372bd4f5fda62d638ded04f7647b28092f8e9b7bdbc7f2b" alt="Image" width="14%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_1/imgs/img_in_chart_box_276_122_442_319.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A32Z%2F-1%2F%2F6af21b97ab2573eb525c062d59fe0f3f60e68eb9bd3ea2cfe9fbd96d1585d4ec" alt="Image" width="13%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_1/imgs/img_in_chart_box_447_119_601_319.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A33Z%2F-1%2F%2F3cff45e0e4d215758e10312ab7524d199456d85ff44cd93fe9f291ece87e6b2c" alt="Image" width="12%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_1/imgs/img_in_chart_box_616_105_944_318.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A33Z%2F-1%2F%2F229644bb6a9b03a2b8a207832611cfa3f2c66925d6d9fee0cfac38c505552bd3" alt="Image" width="27%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_1/imgs/img_in_chart_box_959_124_1119_320.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A33Z%2F-1%2F%2F75e86b4c0418b07427832636a247bdc27f6f945c2fda635ed2f46619a931f331" alt="Image" width="13%" /></div>


Learning in changing environments (concept drifting)

<div style="text-align: center;"><div style="text-align: center;">f</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_1/imgs/img_in_image_box_91_360_417_544.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A33Z%2F-1%2F%2F81d03af8cf91de9c4ae25cb3998ae3bca4ff394e8be5a60e09d885188d78082b" alt="Image" width="27%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_1/imgs/img_in_chart_box_416_359_704_550.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A33Z%2F-1%2F%2Fd0925c035c1a3750f05f5cf7b15b4e809ce48bc503f6bd1167c0a02fad1155ad" alt="Image" width="24%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_1/imgs/img_in_chart_box_720_342_901_550.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A33Z%2F-1%2F%2F6bd6b35f21d8db1087b74e951b4ddecd90afd516ff712d510f578f11f2ee3640" alt="Image" width="15%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_1/imgs/img_in_chart_box_916_334_1111_552.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A33Z%2F-1%2F%2Ffd95ae33d58f5ea1553e2c7bd1e27cf9c966cac997b0f6b355bc6d20c224d697" alt="Image" width="16%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_1/imgs/img_in_chart_box_88_576_294_779.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A33Z%2F-1%2F%2Fe405b03d4186cfd7841db400f9f2e7821d2ba37afa994eae0dd113eb3020298c" alt="Image" width="17%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_1/imgs/img_in_image_box_84_567_705_778.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A33Z%2F-1%2F%2Fb2bf49f67b8de7b403d6797286d245c0d6875cfec516fb8e746245b1597cbb40" alt="Image" width="52%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_1/imgs/img_in_chart_box_305_562_704_774.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A33Z%2F-1%2F%2F019d9f8c89f75d75e630fa8300af0a90799ce61458bac254648763347e325414" alt="Image" width="33%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Fig. 4 | Prospective configuration achieves a superior performance over backpropagation in various learning situations faced by biological systems. a–k, Learning situations include online learning $^{29}$ (a–c), continual learning of multiple tasks $^{30}$ (d–e), learning in changing environments $^{31}$ (f–g), learning with a limited amount of training examples (h) and reinforcement learning $^{4}$ (k). Graphs corresponding to each situation are grouped together with the same background color. Simulations of each situation differ from the ‘default setup’ described in the Methods in a single aspect unique to this task. For example, the default setup involves training with minibatches, so the batch size was only set to 1 in a–c for investigating online learning, whereas it was set to a larger default value in rest of the groups. In supervised learning setups, fully connected networks (a–h) were evaluated on the FashionMNIST $^{50}$ dataset, and convolutional neural networks $^{35}$ (i and j) were evaluated on the CIFAR-10 (ref. 36) dataset. In the reinforcement learning setup (k), fully connected networks were evaluated on three classic control problems. If the learning rate was not presented, each point (a setup of an experiment) in the plot corresponds to the best learning rate optimized independently for each rule under that setup. a, Difference in training setup between computers that can average weight modifications for individual</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_1/imgs/img_in_chart_box_719_569_1117_779.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A33Z%2F-1%2F%2Ff27855f9cc080804dc09a8f0437800274b0f4b5d5e4d88b40a7b5879a25aede3" alt="Image" width="33%" /></div>


<div style="text-align: center;"><div style="text-align: center;">examples to get a 'statistically good' value and biological systems that must apply one modification before computing another. b, Mean of the test errors during training as a function of batch size. c, Minimum of test error during training as a function of learning rate. d, Test error during continual learning of two tasks. e, Mean of test error of both tasks during training as a function of learning rate. f, Test error during training when learning with concept drifting.</div> </div>


g, Mean of test error during training with concept drifting as a function of learning rate. h, Minimum of test error during training with different amounts of training examples (data points per class). i, Minimum of test error during training of a convolutional neural network trained with prospective configuration and backpropagation on the CIFAR-10 (ref. 36) dataset. j, Structure detail of the convolutional neural network used in i. k, Sum of rewards per episode during training on three classic reinforcement learning tasks (insets). An episode is a period from initialization of environment to reaching a terminate state. Each experiment in a-h was repeated with n = 10 random seeds. Each experiment in i-k was repeated with n = 3 random seeds because these experiments are more expensive. Error bars and bands represent the 68% confidence interval.

learning problems that share key features with learning in natural environments. Such problems include online learning, where weights must be updated after each experience (rather than a batch of training examples) $^{29}$, continual learning with multiple tasks $^{30}$, learning in changing environments $^{31}$, learning with a limited amount of training examples and reinforcement learning $^{4}$. In all aforementioned learning problems, prospective configuration demonstrates a notable superiority over backpropagation.

【译文】这些学习问题与自然环境中的学习具有共同的关键特征。这些问题包括在线学习（权重必须在每次经验后更新，而不是在一批训练样本后更新）$^{29}$、多任务持续学习$^{30}$、变化环境中的学习$^{31}$、有限训练样本的学习以及强化学习$^{4}$。在上述所有学习问题中，前瞻性配置都表现出对反向传播的显著优越性。

First, based on the example in Fig. 1, we expect prospective configuration to require fewer episodes for learning than backpropagation. Before presenting the comparison, we describe how backpropagation is used to train ANNs. Typically, the weights are only modified after a batch of training examples based on the average of updates derived from individual examples (Fig. 4a). In fact, backpropagation relies heavily on averaging over multiple experiences to reach human-level performance $^{32}$, as it needs to stabilize training $^{33}$. By contrast, biological systems must update the weights after each experience, and we compare learning performance in such a setting. Sampling efficiency can be quantified by mean of test error during training, which is shown in Fig. 4b as a function of batch size (number of experiences that the updates are averaged over). Efficiency strongly depends on batch size for backpropagation because it requires batch训练 to average out erratic weight updates, whereas this dependence is weaker for prospective configuration, where weight changes are intrinsically less erratic and batch averaging is required less (Supplementary Fig. 7). Importantly, prospective configuration learns faster with smaller batch sizes, as in

【译文】首先，基于图1中的例子，我们预期前瞻性配置需要比反向传播更少的学习轮次。在呈现比较之前，我们先描述反向传播如何用于训练ANN。通常，权重只在经过一批训练样本后才被修改，基于从单个样本推导出的更新的平均值（图4a）。事实上，反向传播严重依赖于对多次经验的平均才能达到人类水平的性能$^{32}$，因为它需要稳定训练$^{33}$。相比之下，生物系统必须在每次经验后更新权重，我们在这种设置下比较学习性能。采样效率可以通过训练期间的测试误差均值来量化，如图4b所示，它是批量大小（更新所平均的经验数量）的函数。对于反向传播，效率严重依赖于批量大小，因为它需要批量训练来平均掉不稳定的权重更新，而这种依赖性对于前瞻性配置则较弱，因为前瞻性配置的权重变化本质上不太不稳定，对批量平均的需求也更小（补充图7）。重要的是，前瞻性配置在较小的批量大小下学习更快，正如

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_2/imgs/img_in_chart_box_78_104_291_246.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A33Z%2F-1%2F%2Fba2bbc8121d415fb5e25db778cb0f454e29cd2351fbd629679192fdb7963cb51" alt="Image" width="17%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_2/imgs/img_in_image_box_80_103_629_357.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A33Z%2F-1%2F%2F2d6dd9d75a6072c9c17bc5510578e2b8ecb163925fb4074cc09814098d162eb5" alt="Image" width="46%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_2/imgs/img_in_image_box_646_128_1125_294.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A33Z%2F-1%2F%2F1c42521bcbb7478f23128aed4420b2dcd5022b0aa6ffcba5e7c3c894aae1d6c4" alt="Image" width="40%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_2/imgs/img_in_chart_box_98_369_629_559.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A33Z%2F-1%2F%2F5a277d8afb6564e0469975fc8cd9f02acb3ea1a10738cd6e0e14053e7ca51451" alt="Image" width="44%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_2/imgs/img_in_chart_box_658_302_1110_558.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A33Z%2F-1%2F%2F7c96012d577019719662fc834202bdcdb92accabe2c98a4ec7f754b50f485e11" alt="Image" width="37%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Fig. 5 | Prospective configuration explains contextual inference in human sensorimotor learning. a, Structure of an experimental trial where participants were asked to move a stick from the starting point to the target point while experiencing perturbations. b, The minimal network for the task, including six connections encoding the associations from the backgrounds (B and R) to the belief of contexts ([B] and [R]) and from the belief of contexts to the prediction of perturbations (+ and -). c–e, Sequence of sessions the participants experienced, including training (c), washout (d) and testing (e). Darker gray boxes show the</div> </div>


<div style="text-align: center;"><div style="text-align: center;">expected network after the session, where thickness represents the strength of connections. In the testing session, the darker box explains how the two learning rules learn differently on the R+ trial, leading to the differences in f.f. Predictions of the two learning rules compared to behavioral data measured from human participants, where prospective configuration reproduces the key patterns of data, but backpropagation does not. Each experiment was repeated with n=24 random seeds, as there were 24 participants in the behavioral experiment.</div> </div>


biological settings. Additionally, final performance can be quantified by the minimum of the test error, which is shown in Fig. 4c, when trained with a batch size equal to 1. Here, prospective configuration also demonstrates a notable advantage over backpropagation.

【译文】生物设置中一样。此外，当批量大小等于1时训练，最终性能可以用测试误差的最小值来量化，如图4c所示。在此，前瞻性配置也表现出对反向传播的显著优势。

Second, biological organisms need to sequentially learn multiple tasks, while ANNs show catastrophic forgetting. When trained on a new task, performance on previously learned tasks is largely destroyed $^{16,34}$. The data in Fig. 4d show performance when trained on two tasks alternately (task 1 is classifying five randomly selected classes in the FashionMNIST dataset, and task 2 is classifying the remaining five classes). Prospective configuration outperforms backpropagation both in terms of avoiding forgetting previous tasks and relearning current tasks. The results are summarized in Fig. 4e.

【译文】其次，生物体需要顺序学习多个任务，而ANN表现出灾难性遗忘。当在新任务上训练时，先前学习任务的性能在很大程度上被破坏$^{16,34}$。图4d中的数据展示了在两个任务上交替训练时的性能（任务1是分类FashionMNIST数据集中随机选择的五个类别，任务2是分类剩下的五个类别）。前瞻性配置在避免遗忘先前任务和重新学习当前任务两方面都优于反向传播。结果总结于图4e中。

Third, biological systems often need to rapidly adapt to changing environments. A common way to simulate this is ‘concept drifting’ $^{31}$, where a part of the mapping between the output neurons to the semantic meaning is shuffled regularly, each time a certain number of training iterations has passed (Fig. 4f). Test error during training with concept drifting is presented in Fig. 4f. Before epoch 0, both learning rules are initialized with the same pretrained model (trained with backpropagation); thus, epoch 0 is the first time the model experiences concept drift. The results are summarized in Fig. 4g and show that, for this task, there is a particularly large difference in mean error (for optimal学习率). This large advantage of prospective configuration is related to it being able to optimally detect which weights to modify (Supplementary Fig. 6) and to preserve existing knowledge while adapting to changes (Fig. 1). This ability to maintain important information while updating other information is critical for survival in natural environments that are bound to change, and prospective configuration has a very substantial advantage in this respect.

【译文】第三，生物系统通常需要快速适应变化的环境。一种常见的模拟方式是"概念漂移"$^{31}$，即输出神经元与语义意义之间的映射的一部分会定期被打乱，每次经过一定数量的训练迭代后发生一次（图4f）。图4f展示了在概念漂移下训练时的测试误差。在epoch 0之前，两种学习规则都用相同的预训练模型（用反向传播训练）初始化；因此，epoch 0是模型第一次经历概念漂移。结果总结于图4g中，表明对于这项任务，平均误差存在特别大的差异（对于最优学习率）。前瞻性配置的这种巨大优势与它能够最优地检测需要修改哪些权重（补充图6）以及在适应变化的同时保留已有知识（图1）有关。这种在更新其他信息的同时保持重要信息的能力对于在不断变化的自然环境中生存至关重要，而前瞻性配置在这方面具有非常显著的优势。

Furthermore, biological学习 is also characterized by limited data availability. Prospective configuration outperforms backpropagation when the model is trained with fewer examples (Fig. 4h).

【译文】此外，生物学习的另一个特点是数据可用性有限。当用较少的样本训练模型时，前瞻性配置优于反向传播（图4h）。

To demonstrate that the advantage of prospective configuration also scales up to larger networks and problems, we evaluated convolutional neural networks $^{35}$ on CIFAR-10 (ref. 36) trained with both learning rules (Fig. 4i), where prospective configuration showed notable advantages over backpropagation. The detailed structure of the convolutional networks is provided in Fig. 4j.

【译文】为了证明前瞻性配置的优势也可以扩展到更大的网络和问题，我们在CIFAR-10（参考文献36）上评估了用两种学习规则训练的卷积神经网络$^{35}$（图4i），其中前瞻性配置显示出对反向传播的显著优势。卷积网络的详细结构见图4j。

Another key challenge for biological systems is to decide which actions to take. Reinforcement learning theories (for example, Q learning) propose that it is solved by learning the expected reward resulting from different actions in different situations $^{37}$. Such prediction of rewards can be made by neural networks $^{4}$, which can be trained with prospective configuration or backpropagation. The sum of rewards per episode during training on three classic reinforcement learning tasks is reported in Fig. 4k, where prospective configuration demonstrates a notable advantage over backpropagation. This large advantage may arise because reinforcement learning is particularly sensitive to erratic changes in network weights (as the target output depends on reward predicted by the network itself for a new state; Methods).

【译文】生物系统面临的另一个关键挑战是决定采取哪些行动。强化学习理论（例如Q学习）提出，这可以通过学习在不同情境下不同行动所产生的预期奖励来解决$^{37}$。这种奖励预测可以由神经网络$^{4}$来完成，神经网络可以用前瞻性配置或反向传播来训练。图4k报告了在三个经典强化学习任务上训练时每回合的累计奖励，其中前瞻性配置表现出对反向传播的显著优势。这种巨大优势可能源于强化学习对网络权重的不稳定变化特别敏感（因为目标输出依赖于网络自身对新状态的奖励预测；见方法部分）。

Based on the superior learning performance of prospective configuration, we may expect that this learning mechanism has been favored by evolution; thus, in the next sections, we investigate if it can account for neural activity and behavior during learning better than backpropagation.

【译文】基于前瞻性配置优越的学习性能，我们可以预期这种学习机制受到了进化的青睐；因此，在接下来的章节中，我们研究它是否比反向传播更好地解释学习过程中的神经活动和行为。

## Evidence for prospective configuration: inferring the latent state during learning

Prospective configuration is related to theories proposing that before learning, the brain first infers a latent state of the environment from feedback $^{38-40}$. Here, we propose that this inference can be achieved in neural circuits through prospective配置, where, following feedback, neurons in ‘hidden layers’ converge to a prospective pattern of activity that encodes this latent state. We demonstrate that data from various previous studies, which involved the inference of a

【译文】前瞻性配置与一些理论有关，这些理论提出在学习之前，大脑首先从反馈中推断环境的潜在状态$^{38-40}$。在此，我们提出这种推断可以通过前瞻性配置在神经回路中实现，即在接受反馈后，"隐藏层"中的神经元收敛到一个编码该潜在状态的前瞻性活动模式。我们证明，来自各种先前研究的数据——这些研究涉及对

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_3/imgs/img_in_image_box_77_108_579_271.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A34Z%2F-1%2F%2F53fdc91799e6acc89da67975338efc5d494446674ccb0dc5b99e713d80ed0a94" alt="Image" width="42%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Fig. 6 | Prospective configuration can discover the underlying task structure during reinforcement learning. a, Reinforcement learning task. Human participants were required to choose between two options, leading to either reward (gaining coins) or punishment (losing coins) with different probabilities. The probability of reward was occasionally reversed between the two options. b, The minimal network encoding the essential elements of the task. c, Activity of the output neuron corresponding to the selected option from networks trained</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_3/imgs/img_in_chart_box_603_116_1126_276.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A34Z%2F-1%2F%2Fbd77902e4a44d62188ebdf6866db5d79d9ada45351f5d1c6b3646500ed158d57" alt="Image" width="43%" /></div>


<div style="text-align: center;"><div style="text-align: center;">with prospective configuration and backpropagation compared with fMRI data measured in human participants (that is, peak blood oxygenation level-dependent (%BOLD) signal in the mPFC). Prospective configuration reproduces the key finding that the expected value (encoded in %BOLD signal in the mPFC) increases if the next choice after a punishing trial is to switch to the other option. The number of trials is not mentioned in the original paper, so we simulated for n = 128 trials for both learning rules. Error bars represent the 68% confidence interval.</div> </div>


latent state, can be explained by prospective configuration. These data were previously explained by complex and abstract的机制, such as Bayesian models $^{38,39}$, whereas here, we mechanically show with prospective configuration how such inference can be performed by minimal networks encoding only the essential elements of the tasks.

【译文】潜在状态的推断——可以被前瞻性配置解释。这些数据此前被复杂而抽象的机制所解释，例如贝叶斯模型$^{38,39}$，而在此，我们用前瞻性配置机械地展示了这种推断如何能通过仅编码任务基本要素的最小网络来执行。

The dynamical inference of a latent state from feedback has been recently proposed to take place during sensorimotor learning $^{39}$. In this experiment, participants received different motor perturbations in different contexts and learned to compensate for these perturbations. Behavioral data suggest that, after receiving feedback, participants first used the feedback to infer context and then adapted the force for the inferred context. We demonstrate that prospective configuration is able to reproduce these behavioral data, whereas backpropagation cannot.

【译文】从反馈中动态推断潜在状态最近被提出发生在感觉运动学习期间$^{39}$。在该实验中，参与者在不同情境中接受了不同的运动扰动，并学习补偿这些扰动。行为数据表明，在接受反馈后，参与者首先利用反馈来推断情境，然后针对推断出的情境调整力度。我们证明前瞻性配置能够复现这些行为数据，而反向传播则不能。

Specifically, in the task (Fig. 5a), participants were asked to move a stick from a starting point to a target point while experiencing perturbations. The participants experienced a sequence of blocks of trials (Fig. 5c–e), including training, washout and testing. During the training session, different directions of perturbations, positive (+) or negative (-), were applied in different contexts, blue (B) or red (R) backgrounds, respectively. We denote these trials as B+ and R−. These trials may be associated with latent states, which we denote [B] and [R]; for example, the latent state [B] may be associated with both background B and perturbation +. The next stage of the task was designed to investigate if the latent state [B] can be activated by perturbation + even if no background B is shown. Thus, participants experienced different trials including R+ (that is, perturbation + but no background B). Specifically, after a washout session (during which no perturbation was provided), in the testing session, participants experienced one of the four possible test trials: B+, R+, B− and R−. To evaluate learning on the test trials, motor adaptation (that is, the difference between the final and target stick positions) was measured before and after the test trial in two trials with the blue background (Fig. 5e). Change in the adaptation between these two trials is a reflection of learning about blue context that occurred at the test trial. If participants only associated feedback with the background color (B), then the change in adaptation would only occur with test trials B+ and B−. However, experimental data (Fig. 5f) show that there was also substantial adaptation change with R+ trials (which was even bigger than with B− trials).

【译文】具体而言，在该任务中（图5a），参与者被要求将一根棍子从起点移动到目标点，同时经历扰动。参与者经历了一系列试次块（图5c–e），包括训练、消退和测试。在训练阶段，在不同情境中分别施加了不同方向的扰动，正方向（+）或负方向（-），蓝色（B）或红色（R）背景。我们将这些试次记为B+和R−。这些试次可能与潜在状态相关联，我们将其记为[B]和[R]；例如，潜在状态[B]可能与背景B和扰动+都相关联。任务的下一阶段旨在研究潜在状态[B]是否可以在不显示蓝色背景B的情况下被扰动+激活。因此，参与者经历了不同的试次，包括R+（即扰动+但没有蓝色背景B）。具体而言，在消退阶段（此阶段不提供扰动）之后，在测试阶段，参与者经历了四种可能的测试试次之一：B+、R+、B−和R−。为了评估测试试次上的学习，在测试试次前后的两次蓝色背景试次中测量了运动适应（即棍子最终位置与目标位置之间的差异）（图5e）。这两次试次之间适应的变化反映了在测试试次上发生的关于蓝色情境的学习。如果参与者只将反馈与背景颜色（B）相关联，那么适应的变化只会发生在B+和B−测试试次中。然而，实验数据（图5f）表明，在R+试次中也存在显著的适应变化（甚至比B−试次中的变化更大）。

To model learning in this task, we considered a neural network (Fig. 5b) where input nodes encode the background color, and outputs encode movement compensations in the two directions. Importantly, this network also includes hidden neurons encoding belief of being in the contexts associated with the two backgrounds ([B] and [R]). Trained with the exact procedure of the experiment $^{39}$ from randomly initialized weights, prospective configuration with this minimal network can reproduce the behavioral data, whereas backpropagation cannot (Fig. 5f).

【译文】为了对该任务中的学习进行建模，我们考虑了一个神经网络（图5b），其中输入节点编码背景颜色，输出编码两个方向上的运动补偿。重要的是，该网络还包括编码处于与两个背景相关联的情境中的信念的隐藏神经元（[B]和[R]）。用与实验$^{39}$完全相同的程序从随机初始化的权重开始训练，这个最小网络配合前瞻性配置可以复现行为数据，而反向传播则不能（图5f）。

Prospective configuration can produce change in adaptation with the R+ test trial because after + feedback, it is able to also activate context [B] that was associated with this feedback during training and then learn compensation for this latent state. To shed light on how this inference takes place in the model, schematics in Fig. 5c, d show evolution of the weights of the network over sessions (thickness represents the strength of connections). The schematic in Fig. 5e shows the difference between the two learning rules after exposure to R+; although B is not perceived, prospective configuration infers a moderate excitation of the belief of blue context [B] because the positive connection from [B] to + was built during the training session. The activity of [B] enables the learning of weights from [B] to + and −, while backpropagation does not modify any weights originating from [B].

【译文】前瞻性配置能够在R+测试试次中产生适应变化，因为在+反馈后，它还能够激活在训练期间与这一反馈相关联的情境[B]，然后学习对该潜在状态的补偿。为了阐明这种推断在模型中是如何发生的，图5c、d中的示意图展示了网络权重在不同阶段间的演变（粗细代表连接的强度）。图5e中的示意图展示了两种学习规则在暴露于R+后的差异；尽管B没有被感知到，前瞻性配置推断出蓝色情境[B]的信念受到适度兴奋，因为[B]到+的正向连接在训练阶段已经建立。[B]的活动使得从[B]到+和−的权重的学习成为可能，而反向传播则不会修改任何源自[B]的权重。

For simplicity of explanation, we presented simulations with minimal networks; however, Supplementary Fig. 8 shows that networks with a general fully connected structure and more hidden neurons can replicate the above data when using prospective configuration but not when using backpropagation.

【译文】为了解释简洁起见，我们展示了用最小网络进行的模拟；然而，补充图8表明，具有一般全连接结构和更多隐藏神经元的网络在使用前瞻性配置时可以复现上述数据，而在使用反向传播时则不能。

Studies of animal conditioning have also observed that feedback in learning tasks involving multiple stimuli may trigger learning about non-presented stimuli $^{41,42}$. One example is provided in Supplementary Fig. 9, where we show that it can be explained by prospective configuration but not by backpropagation.

【译文】动物条件作用的研究也观察到，在涉及多种刺激的学习任务中，反馈可能触发对未呈现刺激的学习$^{41,42}$。补充图9提供了一个例子，我们在其中表明它可以用前瞻性配置解释，但不能用反向传播解释。

### Evidence for prospective configuration: discovering task structure during learning

Prospective configuration is also able to discover the underlying task structure in reinforcement learning. Specifically, we consider a task where reward probabilities of different options were not independent $^{38}$. In this study, humans were choosing between two options where the reward probabilities were constrained such that one option had a higher reward probability than the other (Fig. 6a). Occasionally the reward probabilities were swapped, so if one probability was increased, the other was decreased by the same amount. Remarkably, the recorded functional magnetic resonance imaging (fMRI) data suggested that participants learned that the values of the two options were negatively correlated and on each trial updated the value estimates of both options in opposite ways. This conclusion was drawn from analysis of the signal from the medial prefrontal cortex (mPFC), which encoded the expected value of reward. The data presented in Fig. 6c compare this signal after making a choice on two consecutive trials: a trial in which the reward was not received ('punish trial') and the next trial. If the participant selected the same option on both trials ('stay'), the signal decreased, indicating that the reward expected by the participant was reduced. Remarkably, if the participant selected the other option on the next trial ('switch'), the signal increased, suggesting that negative feedback for one option increased the value estimate for

【译文】前瞻性配置还能够发现强化学习中的潜在任务结构。具体而言，我们考虑一个不同选项的奖励概率并非相互独立的任务$^{38}$。在该研究中，人类在两个选项之间进行选择，其中奖励概率受到约束，使得一个选项的奖励概率高于另一个（图6a）。奖励概率偶尔会互换，因此如果一个概率增加，另一个则减少相同的量。引人注目的是，记录到的功能性磁共振成像（fMRI）数据表明，参与者学会了两个选项的价值呈负相关，并且在每次试次中以相反的方式更新两个选项的价值估计。这一结论是通过分析内侧前额叶皮层（mPFC）的信号得出的，该信号编码了奖励的预期价值。图6c中呈现的数据比较了在连续两次试次做出选择后的该信号：一次未获得奖励的试次（"惩罚试次"）和下一次试次。如果参与者在两次试次中都选择了相同的选项（"停留"），信号降低，表明参与者预期的奖励减少了。引人注目的是，如果参与者在下一次试次中选择了另一个选项（"切换"），信号增加，表明对

the other. Such learning is not predicted by standard reinforcement learning models $^{38}$.

【译文】另一个选项的价值估计增加了。这种学习无法被标准强化学习模型预测$^{38}$。

This task can be conceptualized as having a latent state encoding which option is superior, and this latent state determines the reward probabilities for both options. Consequently, we consider a neural network reflecting this structure (Fig. 6b) that includes an input neuron encoding being in the task (equal to 1 in simulations), a hidden neuron encoding the latent state and two output neurons encoding the reward probabilities for the two options. Trained with the exact procedure of the experiment $^{38}$ from randomly initialized weights, prospective configuration with this minimal network can reproduce the data, whereas backpropagation cannot (Fig. 6c). In Supplementary Fig. 10, we show that prospective configuration reproduces these data because it can infer the rewarded choice by updating the activity of the hidden neuron based on feedback.

【译文】该任务可以被概念化为具有一个编码哪个选项更优的潜在状态，这个潜在状态决定了两个选项的奖励概率。因此，我们考虑了一个反映这种结构的神经网络（图6b），它包括一个编码处于任务中的输入神经元（在模拟中等于1）、一个编码潜在状态的隐藏神经元，以及两个编码两个选项奖励概率的输出神经元。用与实验$^{38}$完全相同的程序从随机初始化的权重开始训练，这个最小网络配合前瞻性配置可以复现数据，而反向传播则不能（图6c）。在补充图10中，我们证明前瞻性配置之所以能复现这些数据，是因为它可以通过基于反馈更新隐藏神经元的活动来推断被奖励的选项。

Taken together, the presented simulations illustrate that prospective configuration is a common principle that can explain a range of surprising learning effects in diverse tasks.

【译文】综上所述，所呈现的模拟表明，前瞻性配置是一种普遍的原理，能够解释各种任务中一系列令人惊讶的学习效应。

## Discussion

Our paper identifies the principle of prospective configuration, according to which learning relies on neurons first optimizing their pattern of activity to match the correct output and then reinforcing these prospective activities through synaptic plasticity. Although it was known that in energy-based networks the activity of neurons shifts before weight update, it has been previously thought that this shift is a necessary cost of error propagation in biological networks, and several methods have been proposed to suppress it $^{[1,12,14,20,21]}$ to approximate backpropagation more closely. By contrast, we demonstrate that this reconfiguration of neural activity is the key to achieving learning performance superior to that of backpropagation and to explaining experimental data from diverse learning tasks. Prospective configuration further offers a range of experimental predictions distinct from those of backpropagation (Supplementary Figs. 11 and 12). Together, we have demonstrated that prospective configuration enables more efficient learning than backpropagation by reducing interference, demonstrates superior performance in situations faced by biological organisms, requires only local computation and plasticity and matches experimental data across a wide range of tasks.

【译文】我们的论文识别了前瞻性配置原理，根据这一原理，学习依赖于神经元首先优化其活动模式以匹配正确输出，然后通过突触可塑性强化这些前瞻性活动。尽管已知在能量基网络中神经元活动在权重更新之前会发生变化，但此前人们一直认为这种变化是生物网络中误差传播的必要代价，并且已经提出了几种方法来抑制它$^{[1,12,14,20,21]}$，以更紧密地逼近反向传播。相比之下，我们证明这种神经活动的重配置是实现优于反向传播的学习性能以及解释来自各种学习任务的实验数据的关键。前瞻性配置进一步提供了一系列不同于反向传播的实验预测（补充图11和12）。总之，我们已经证明前瞻性配置通过减少干扰实现了比反向传播更高效的学习，在生物体面临的情境中表现出优越性能，仅需要局部计算和可塑性，并且与广泛任务中的实验数据相匹配。

Our theory addresses a long-standing question of how the brain solves the plasticity-stability dilemma, for example, how it is possible that, despite adjustment of representation in the primary visual cortex during learning $^{43}$, we can still understand the meaning of visual stimuli we learned over our lifetime. According to prospective configuration, when some weights are modified, compensatory changes are made to other weights to ensure the stability of correctly预测 outputs. Thus, prospective configuration reduces interference between different weight modifications while learning a single association. Previous computational models have proposed mechanisms that reduce interference between new and previously acquired information while learning multiple associations $^{34,44}$. It is highly likely that such mechanisms and prospective configuration operate in the brain in parallel to minimize both types of interference.

【译文】我们的理论解决了一个长期存在的问题，即大脑如何解决可塑性-稳定性困境，例如，尽管在学习过程中初级视觉皮层的表征会发生调整$^{43}$，但我们仍然能够理解一生中学习过的视觉刺激的含义，这是如何可能的。根据前瞻性配置，当某些权重被修改时，会对其他权重进行补偿性变化，以确保正确预测输出的稳定性。因此，前瞻性配置在学习单一关联时减少了不同权重修改之间的干扰。先前的计算模型提出了在学习多个关联时减少新信息与先前获得信息之间干扰的机制$^{34,44}$。很有可能这些机制和前瞻性配置在大脑中并行运作，以最小化这两种类型的干扰。

Prospective configuration is related to inference and learning procedures in statistical modeling. If the ‘energy’ in energy-based schemes is variational free energy, prospective configuration can be seen as an implementation of variational Bayes that subsumes inference and learning $^{45}$. For example, dynamic expectation maximization $^{46,47}$ can be regarded as a generalization of predictive coding networks in which the D-step optimizes representations of latent states (analogously to relaxation until convergence during inference) while the E-step optimizes model parameters (analogously to weight modification during learning).

【译文】前瞻性配置与统计建模中的推断和学习程序有关。如果能量基方案中的"能量"是变分自由能，前瞻性配置可以被视为一种变分贝叶斯的实现，它涵盖了推断和学习$^{45}$。例如，动态期望最大化$^{46,47}$可以被视为预测编码网络的推广，其中D步优化潜在状态的表征（类似于推断期间直到收敛的松弛），而E步优化模型参数（类似于学习期间的权重修改）。

Other recent work $^{48,49}$ also noticed that the natural form of energy-based networks ('strong control' in their words) performs different learning than backpropagation. Their analysis concentrates on an architecture of deep feedback control, and they demonstrated that a particular form of their model is equivalent to predictive编码 networks $^{49}$. The unique contribution of our paper is to show the benefits of such strong control and explain why they arise. The principle of prospective configuration is also present in other recent models. For example, Gilra and Gerstner $^{50}$ developed a spiking model in which feedback about the error on the output directly affects the activity of hidden neurons before plasticity takes place. Haider et al. $^{51}$ developed a faster inference algorithm for energy-based models that computes a value to which the activity is likely to converge, termed latent equilibrium $^{51}$. Iteratively setting each neuron's output based on its latent equilibrium leads to much faster inference $^{51}$ and enables efficient computation of the prospective configuration.

【译文】其他近期工作$^{48,49}$也注意到能量基网络的自然形式（用他们的话说叫"强控制"）执行的学习不同于反向传播。他们的分析集中于深度反馈控制的架构，并且他们证明了其模型的一种特定形式等价于预测编码网络$^{49}$。我们论文的独特贡献在于展示了这种强控制的好处并解释了它们为何产生。前瞻性配置原理也存在于其他近期模型中。例如，Gilra和Gerstner$^{50}$开发了一个脉冲模型，其中关于输出误差的反馈在可塑性发生之前直接影响隐藏神经元的活动。Haider等人$^{51}$为能量基模型开发了一种更快的推断算法，该算法计算活动可能收敛到的值，称为潜平衡$^{51}$。基于其潜平衡迭代设置每个神经元的输出会带来快得多的推断$^{51}$，并使得前瞻性配置的高效计算成为可能。

Predictive coding networks require symmetric forward and backward weights between layers of neurons, so a question arises concerning how such symmetry may develop in the brain. If predictive coding networks are initialized with symmetric weights (as in our simulations), the symmetry will persist because the changes in weight between neurons A and B are the same as those for feedback weight (between neurons B and A). Even if the weights are not initialized symmetrically, the symmetry may develop if synaptic decay is included in the model $^{52}$ because then the initial asymmetric values decay away, and weight values become more influenced by recent changes that are symmetric. Nevertheless, weight symmetry is not generally required for effective credit assignment $^{53,54}$.

【译文】预测编码网络需要神经元层之间对称的前向和反向权重，因此产生了一个问题：这种对称性如何在大脑中发展。如果预测编码网络用对称权重初始化（如我们的模拟中那样），对称性将会持续存在，因为神经元A和B之间权重的变化与反馈权重（神经元B和A之间）的变化相同。即使权重没有被对称初始化，如果模型中包含突触衰减$^{52}$，对称性也可能会发展出来，因为此时初始的不对称值会衰减掉，而权重值更多地受到最近对称变化的影响。尽管如此，权重对称性通常并不是有效信用分配所必需的$^{53,54}$。

Here, we assumed for simplicity that the convergence of neural activity to an equilibrium happens rapidly after the stimuli are provided so that the synaptic weight modification after convergence may take place while the stimuli are still present. Nevertheless, predictive coding networks can still work even if weight modification takes place while the neural activity is converging. Specifically, Song et al. demonstrated that if neural activities are only updated for the first few steps, the update of the weights is equivalent to that in backpropagation $^{14}$. As a reminder, we demonstrate here that if the neural activities are updated to equilibrium, the update of the weights follows the principle of prospective configuration and possesses the desirable demonstrated properties. Thus, a learning rule where neural activities and weights are updated in parallel will experience a weight update that is equivalent to backpropagation at the start and then move to prospective configuration as the system converges to equilibrium $^{55}$. Furthermore, predictive coding networks have been extended to describe recurrent structures $^{56-58}$ and it has been shown that such networks can learn to predict dynamically changing stimuli even if weights are modified before the activity converged for a given 'frame' of the stimulus $^{57}$.

【译文】在此，为了简化，我们假设神经活动在提供刺激后迅速收敛到平衡状态，以便收敛后的突触权重修改可以在刺激仍然存在时发生。然而，即使权重修改发生在神经活动收敛过程中，预测编码网络仍然可以工作。具体而言，Song等人证明，如果神经活动只更新最初的几个步骤，权重的更新等价于反向传播中的更新$^{14}$。提醒一下，我们在此证明，如果神经活动被更新到平衡状态，权重的更新遵循前瞻性配置原理，并具有所展示的令人满意的特性。因此，一种神经活动和权重并行更新的学习规则，将在开始时经历等价于反向传播的权重更新，然后随着系统收敛到平衡状态而过渡到前瞻性配置$^{55}$。此外，预测编码网络已被扩展以描述循环结构$^{56-58}$，并且已证明即使权重在给定刺激"帧"的活动收敛之前就被修改，这种网络也能学习预测动态变化的刺激$^{57}$。

The advantages of prospective configuration suggest that it may be profitably applied in machine learning to improve the efficiency and performance of deep neural networks. An obstacle for this is that the relaxation phase is computationally expensive. However, recent work demonstrated that by modifying weights after each step of relaxation, the model becomes comparably fast to backpropagation and easier for parallelization $^{55}$.

【译文】前瞻性配置的优势表明，它可以被有利地应用于机器学习，以提高深度神经网络的效率和性能。这方面的一个障碍是松弛阶段的计算成本高昂。然而，近期工作证明，通过在松弛的每一步后修改权重，模型可以达到与反向传播相当的速度，并且更容易并行化$^{55}$。

Most intriguingly, it has been demonstrated that the speed of energy-based networks can be greatly increased by implementing the relaxation on analog hardware $^{59}$, potentially resulting in energy-based networks being faster than backpropagation. Therefore, we anticipate that our discoveries may change the blueprint of next-generation machine learning hardware, switching from the current digital tensor base to analog hardware and being closer to the brain and potentially far more efficient.

【译文】最引人注目的是，已经证明通过在模拟硬件上实现松弛，能量基网络的速度可以大幅提高$^{59}$，这可能使得能量基网络比反向传播更快。因此，我们预期我们的发现可能会改变下一代机器学习硬件的蓝图，从当前的数字张量基础转向模拟硬件，更接近大脑，并且潜在地更加高效。

#### Online content

Any methods, additional references, Nature Portfolio reporting summaries, source data, extended data, supplementary information, acknowledgements, peer review information; details of author contributions and competing interests; and statements of data and code availability are available at https://doi.org/10.1038/s41593-023-01514-1.

【译文】任何方法、额外参考文献、Nature Portfolio报告摘要、源数据、扩展数据、补充信息、致谢、同行评审信息；作者贡献和利益冲突的详细信息；以及数据和代码可用性声明均可在 https://doi.org/10.1038/s41593-023-01514-1 获取。


