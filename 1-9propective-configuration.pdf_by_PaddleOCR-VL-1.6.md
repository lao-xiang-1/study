Article

https://doi.org/10.1038/s41593-023-01514-1

# Inferring neural activity before plasticity as a foundation for learning beyond backpropagation

Received: 18 May 2022

Accepted: 2 November 2023

Yuhang Song $^{1,2,3}$, Beren Millidge $^{2}$, Tommaso Salvatori $^{1,4,5}$,

Published online: 3 January 2024

Thomas Lukasiewicz $^{1,4}$, Zhenghua Xu $^{1,6}$ & Rafal Bogacz $^{2}$

Check for updates

For both humans and machines, the essence of learning is to pinpoint which components in its information processing pipeline are responsible for an error in its output, a challenge that is known as 'credit assignment'. It has long been assumed that credit assignment is best solved by backpropagation, which is also the foundation of modern machine learning. Here, we set out a fundamentally different principle on credit assignment called 'prospective configuration'. In prospective configuration, the network first infers the pattern of neural activity that should result from learning, and then the synaptic weights are modified to consolidate the change in neural activity. We demonstrate that this distinct mechanism, in contrast to backpropagation, (1) underlies learning in a well-established family of models of cortical circuits, (2) enables learning that is more efficient and effective in many contexts faced by biological organisms and (3) reproduces surprising patterns of neural activity and behavior observed in diverse human and rat learning experiments.

The credit assignment problem $^{1}$ lies at the very heart of learning. Backpropagation $^{2}$, as a simple yet effective credit assignment theory, has powered notable advances in artificial intelligence since its inception $^{3-5}$ and has also gained a predominant place in understanding learning in the brain $^{1,6-8}$. Due to this success, much recent work has focused on understanding how biological neural networks could learn in a way similar to backpropagation $^{9-12}$; although many proposed models do not implement backpropagation exactly, they nevertheless try to approximate backpropagation, and much emphasis is placed on how close this approximation is $^{9,11,13,14}$. However, learning in the brain is superior to backpropagation in many critical aspects. For example, compared to the brain, backpropagation requires many more exposures to a stimulus to learn $^{15}$ and suffers from catastrophic interference of newly and previously stored information $^{16}$. This raises the question of whether using backpropagation to understand learning in the brain should be the main focus of the field.

Here, we propose that the brain instead solves credit assignment with a fundamentally different principle, which we call 'prospective configuration'. In prospective configuration, before synaptic weights are modified, neural activity changes across the network so that output neurons better predict the target output; only then are the synaptic weights (hereafter termed 'weights') modified to consolidate this change in neural activity. By contrast, in backpropagation, the order is reversed; weight modification takes the lead, and the change in neural activity is the result that follows.

We identify prospective configuration as a principle that is implicitly followed by a well-established family of neural models with solid biological groundings, namely, energy-based networks.

 $^{1}$Department of Computer Science, University of Oxford, Oxford, UK.  $^{2}$Medical Research Council Brain Network Dynamics Unit, University of Oxford, Oxford, UK.  $^{3}$Fractile, Ltd., London, UK.  $^{4}$Institute of Logic and Computation, Vienna University of Technology, Vienna, Austria.  $^{5}$VERSES AI Research Lab, Los Angeles, CA, USA.  $^{6}$State Key Laboratory of Reliability and Intelligence of Electrical Equipment, School of Health Sciences and Biomedical Engineering, Hebei University of Technology, Tianjin, China. e-mail: yuhang.song@bndu.ox.ac.uk; thomas.lukasiewicz@cs.ox.ac.uk; zhenghua.xu@hebut.edu.cn; rafal.bogacz@ndcn.ox.ac.uk

These networks include Hopfield networks $^{17}$ and predictive coding networks $^{18}$, which have been successfully used to describe information processing in the cortex $^{19}$. To support the theory of prospective configuration, we show that it can both yield efficient learning, which humans and animals are capable of, and reproduce data from experiments on human and animal learning. Thus, on the one hand, we demonstrate that prospective configuration performs more efficient and effective learning than backpropagation in various situations faced by biological systems, such as learning with deep structures, online learning, learning with a limited amount of training examples, learning in changing environments, continual learning with multiple tasks and reinforcement learning. On the other hand, we demonstrate that patterns of neural activity and behavior in diverse human and animal learning experiments, including sensorimotor learning, fear conditioning and reinforcement learning, can be naturally explained by prospective configuration but not by backpropagation.

Guided by the belief that backpropagation is the foundation of biological learning, previous work showed that energy-based networks can closely approximate backpropagation. However, to achieve it, the networks were set up in an unnatural way, such that the neural activity was prevented from substantially changing before weight modification by constraining the supervision signal to be infinitely small (for example, as in equilibrium propagation $^{11}$ and in previous studies using predictive coding networks $^{12,20}$) or last an infinitely short time $^{14,21}$. By contrast, we reveal that energy-based networks without these unrealistic constraints follow the distinct principle of prospective configuration rather than backpropagation and are superior in both learning efficiency and accounting for data on biological learning.

Here, we introduce prospective configuration with an intuitive example, show how it originates from energy-based networks and describe its advantages and quantify them in a rich set of biologically relevant learning tasks. We show that prospective configuration naturally explains patterns of neural activity and behavior in diverse learning experiments.

## Results

##### Prospective configuration: an intuitive example

To optimally plan behavior, it is critical for the brain to predict future stimuli, for example, to predict sensations in some modalities on the basis of other modalities $^{22}$. If the observed outcome differs from the prediction, the weights in the whole network need to be updated so that predictions in the 'output' neurons are corrected. Backpropagation computes how the weights should be modified to minimize the error on the output, and this weight update results in a change in neural activity when the network next makes the prediction. By contrast, we propose that neural activity is first adjusted to a new configuration so that the output neurons better predict the observed outcome (target pattern); the weights are then modified to reinforce this configuration of neural activity. We call this configuration of neural activity 'prospective' because it is the neural activity that the network should produce to correctly predict the observed outcome. In agreement with the proposed mechanism of prospective configuration, it has indeed been widely observed in biological neurons that are presenting the outcome of a prediction triggers changes in neural activity; for example, in tasks requiring animals to predict a juice delivery, the reward triggers rapid changes in activity not only in the gustatory cortex but also in multiple cortical regions $^{23,24}$.

To highlight the difference between backpropagation and prospective configuration, consider a simple example (Fig. 1a). Imagine a bear seeing a river. In the bear's mind, the sight generates predictions of hearing water and smelling salmon. On that day, the bear indeed smelled the salmon but did not hear the water, perhaps due to an ear injury, and thus the bear needs to change its expectation related to the sound. Backpropagation (Fig. 1b) would proceed by backpropagating the negative error to reduce the weights on the path between the visual and auditory neurons. However, this also entails a reduction of the weights between visual and olfactory neurons that would compromise the expectation of smelling the salmon the next time the river is visited, even though the smell of salmon was present and correctly predicted. These undesired and unrealistic side effects of learning with backpropagation are closely related with the phenomenon of catastrophic interference, where learning a new association destroys previously learned memories $^{16}$. This example shows that, with backpropagation, even learning one new aspect of an association may interfere with the memory of other aspects of the same association.

By contrast, prospective configuration assumes that learning starts with the neurons being configured to a new state, which corresponds to a pattern enabling the network to correctly predict the observed outcome. The weights are then modified to consolidate this state. This behavior can 'foresee' side effects of potential weight modifications and compensate for them dynamically (Fig. 1c). To correct the negative error on the incorrect output, the hidden neurons settle to their prospective state of lower activity, and, as a result, a positive error is revealed and allocated to the correct output. Consequently, prospective configuration increases the weights connecting to the correct output, whereas backpropagation does not (Fig. 1b,c). Hence, prospective configuration is able to correct the side effects of learning an association effectively and efficiently and with little interference.

##### Origin of prospective configuration: energy-based networks

Origin of prospective configuration: energy-based networks

To show how prospective configuration naturally arises in energy-based networks, we introduce a physical machine analog, which provides an intuitive understanding of energy-based networks and how they produce the mechanism of prospective configuration.

Energy-based networks have been widely and successfully used in describing biological neural systems $^{17,25}$. In these models, a neural circuit is described by a dynamical system driven by reducing an abstract ‘energy’, for example, reflecting errors made by neurons (Methods). Neural activity and weights change to reduce this energy; hence, they can be considered ‘movable parts’ of the dynamical system. We show that energy-based networks are mathematically equivalent to a physical machine (we call it ‘energy machine’), where the energy function has an intuitive interpretation, and its dynamics are straightforward; the energy machine simply adjusts its movable parts to reduce energy.

The energy machine includes nodes sliding on vertical posts connected with each other via rods and springs (Fig. 2a,b). Translating from energy-based networks to the energy machine, neural activity maps to the vertical position of a solid node, a connection maps to a rod (blue arrow) pointing from one node to another (where the weight determines how the end position of the rod relates to the initial position), and the energy function maps to the elastic potential energy of springs with nodes attached on both ends (the natural length of the springs is 0). Different energy functions and network structures result in different energy-based networks, corresponding to energy machines with different configurations and combinations of nodes, rods and springs. In Fig. 2, we present the energy machine of predictive coding networks $^{12,18}$ because they are most accessible and are established to be closely related to backpropagation $^{12,14}$.

The dynamics of energy-based networks, which are driven by minimizing the energy function, map to relaxation of the energy machine, which is driven by reducing the total elastic potential energy on the springs. A prediction with energy-based networks involves clamping the input neurons to the provided stimulus and updating the activity of the other neurons, which corresponds to fixing one side of the energy machine and letting the energy machine relax by moving nodes (Fig. 2a). Learning with energy-based networks involves clamping the input and output neurons to the corresponding stimulus, first letting the activities of the remaining neurons converge and then updating weights, which corresponds to fixing both sides of the energy machine and letting the energy machine relax first by moving nodes and then tuning rods (Fig. 2b).

<div style="text-align: center;"><div style="text-align: center;">a</div> </div>


<div style="text-align: center;"><div style="text-align: center;">b</div> </div>


<div style="text-align: center;"><div style="text-align: center;">Backpropagation (conventional)</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//5976061a-e5ca-44d1-952f-84c8ce6f252b/markdown_2/imgs/img_in_image_box_81_108_1124_560.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A32Z%2F-1%2F%2F6c15c52eddb532b71de20a03b86c6f5ade8a153cadc64f55cc18b1a79caef5c7" alt="Image" width="87%" /></div>


<div style="text-align: center;"><div style="text-align: center;">C</div> </div>


<div style="text-align: center;"><div style="text-align: center;">Prospective configuration (proposed)</div> </div>


<div style="text-align: center;"><div style="text-align: center;">Fig. 1 Prospective configuration avoids interference during learning. a, Abstract (top) and concrete (bottom) examples of a task inducing interference during learning. One stimulus input (seeing the water) triggers two prediction outputs (hearing the water and smelling the salmon). One output is correct (smelling the salmon), whereas the other output is an error (not hearing the water). b,c, Backpropagation produces interference during learning; not hearing the water reduces the expectation of smelling the salmon (b), although the salmon was indeed smelled. Prospective configuration, on the other hand, avoids such interference (c). In backpropagation, negative error propagates.</div> </div>


<div style="text-align: center;"><div style="text-align: center;">from the error output to hidden neurons (b; left). This causes a weakening of some connections, which, on the next trial, improves the incorrect output but also reduces the prediction of the correct output, thus introducing interference (b; middle and right). In prospective configuration, neural activity settles into a new configuration (different intensities of purple) before weight modification (c; left). This configuration corresponds to the activity that should be produced after learning, that is, is 'prospective'. Hence, it foresees the positive error on the correct output and modifies the connections to improve the incorrect output while maintaining the correct output (c; middle and right).</div> </div>


The energy machine reveals the essence of energy-based networks; relaxation before weight modification lets the network settle to a new configuration of neural activity corresponding to the neural activity that would have occurred after the error was corrected by the modification of weights, that is, prospective activity (thus, we call this mechanism prospective configuration). For example, the second-layer ‘neuron’ in Fig. 2b increases its activity, and this increase in activity would also be caused by the subsequent weight modification (of the connection between the first and second neurons). In simple terms, relaxation in energy-based networks infers the prospective neural activity after learning, toward which the weights are then modified. This distinguishes it from backpropagation, where weight modification takes the lead, and the change in neural activity is the result that follows.

The bottom of Fig. 2c shows the connectivity of a predictive coding network $^{12,18}$, which has dynamics mathematically equivalent to those of the energy machine shown above it. Predictive coding networks include neurons (blue) corresponding to nodes on the posts and separate neurons encoding prediction errors (red) corresponding to springs. For details, see Methods and Supplementary Fig. 1, where we list equations describing predictive coding networks and show how they map on the neural implementation and the proposed energy machine.

Using the energy machine, Fig. 2d simulates the learning problem from Fig. 1. Here, we can see that prospective configuration indeed foresees the result of learning and its side effects through relaxation. Hence, it corrects the side effects within one iteration, which would otherwise take multiple iterations for backpropagation.

### Advantages of prospective configuration: reduced interference and faster learning

Here, we quantify interference in the above scenario and demonstrate how reduced interference translates into an advantage in performance. In all simulations in the main text, prospective configuration is implemented in predictive coding networks (other energy-based models are considered in the Supplementary Notes, Section 2.1). We also compare the performance of predictive coding networks against artificial neural networks (ANNs) trained with backpropagation because they are closely related, which makes the comparisons fair. In particular, although predictive coding networks include recurrent connections, they generate the same prediction for a given input (when inputs are constrained but outputs are not; Fig. 2a) as standard feedforward ANNs if their weights are set to corresponding values $^{12,14}$. Therefore, loss is the same function of weights in both models, so direct minimization of loss with gradient descent in predictive coding networks (which is not their natural way of training) would produce the same weight changes as backpropagation in ANNs. Hence, comparing predictive coding networks and backpropagation enables isolation of the effects of the learning algorithm (prospective configuration versus direct minimization of loss as in backpropagation).

In Fig. 3a, we compare the activity of output neurons in the example in Fig. 1 between backpropagation and prospective configuration. Initially both output neurons are active (top right), and the output should change toward a target in which one of the neurons is inactive (red vector). Learning with prospective configuration results in changes on the output (purple solid vector) that are aligned better with the target than those for backpropagation (purple dotted vector).
在图3a中，我们比较了图1示例中反向传播和前瞻性配置的输出神经元活动。初始时两个输出神经元都是活跃的（右上方），而输出应该朝着一个目标变化，即其中一个神经元变为不活跃（红色向量）。前瞻性配置的学习导致输出发生变化（紫色实线向量），其与目标的对齐程度优于反向传播（紫色虚线向量）。

**对照理解**：

| 元素          | 含义                                   |
| ----------- | ------------------------------------ |
| 初始状态（右上方）   | 两个输出神经元都活跃 → 熊同时"预测听到水声"和"预测闻到三文鱼"   |
| 目标状态（红色向量）  | 只有闻到三文鱼是对的，所以应该朝"听觉神经元不活跃、嗅觉神经元活跃"变化 |
| 紫色虚线（反向传播）  | 权重独立更新互相干扰 → 输出变化方向偏离目标              |
| 紫色实线（前瞻性配置） | 先统一推断理想神经活动 → 输出变化方向更对准目标            |

Following the first weight update, we simulate multiple iterations until the network is able to correctly predict the target. Here, 'iteration' refers to each time the agent is presented with stimuli and conducts one weight update because of the stimulus. Although the output from backpropagation can reach the target after multiple iterations, the output for the 'correct neuron' diverges from the target during learning and then comes back; this is a particularly undesired effect in biological learning, where networks can be 'tested' at any point during the learning process, because it may lead to incorrect decisions.

<div style="text-align: center;"><div style="text-align: center;">a</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//5976061a-e5ca-44d1-952f-84c8ce6f252b/markdown_3/imgs/img_in_image_box_106_115_336_250.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A32Z%2F-1%2F%2F2022dba6ab0b93e8cdc55d716a03cafa332d21f0f3e54e3264198a105aa5577b" alt="Image" width="19%" /></div>


Clamp input neuron

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//5976061a-e5ca-44d1-952f-84c8ce6f252b/markdown_3/imgs/img_in_image_box_352_119_584_244.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A32Z%2F-1%2F%2F6119b1d8c321ca1eab58b4a548a3a9ef6a496a737a7b735f84cc21788f91663b" alt="Image" width="19%" /></div>


Predict

Relaxation  $(\Delta x \sim -\partial E/\partial x)$

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//5976061a-e5ca-44d1-952f-84c8ce6f252b/markdown_3/imgs/img_in_image_box_599_117_832_242.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A32Z%2F-1%2F%2Ffe1078b799e7ac5cbfa8fe5652bccfba459bd766b78f67fb2816a04ddcf3fa30" alt="Image" width="19%" /></div>


Relaxation convergence

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//5976061a-e5ca-44d1-952f-84c8ce6f252b/markdown_3/imgs/img_in_image_box_895_114_1126_250.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A32Z%2F-1%2F%2F727db264f98098d4f5e32a83416bc7bc59a8aa7eef0d9d14b22b47e66b1221a6" alt="Image" width="19%" /></div>


<div style="text-align: center;"><div style="text-align: center;">C</div> </div>


Physical implementation

Clamp input and output neurons

<div style="text-align: center;"><div style="text-align: center;">b</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//5976061a-e5ca-44d1-952f-84c8ce6f252b/markdown_3/imgs/img_in_image_box_107_332_334_463.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A32Z%2F-1%2F%2Fe620cc557a156d2096b4930e423c99952934ddf8eed41ee1a94df12c07136a8a" alt="Image" width="19%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//5976061a-e5ca-44d1-952f-84c8ce6f252b/markdown_3/imgs/img_in_image_box_351_338_582_460.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A33Z%2F-1%2F%2Fe78789242ccaa1a57086c05cea68d709945f944bd9febe9a016c82b110ebc425" alt="Image" width="19%" /></div>


Relaxation until convergence

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//5976061a-e5ca-44d1-952f-84c8ce6f252b/markdown_3/imgs/img_in_image_box_598_338_832_460.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A33Z%2F-1%2F%2F656354d296de05bca38d571d657e53c5f8a4b908f9d83450872e255f8813e608" alt="Image" width="19%" /></div>


Learn

Weight modification ( $\Delta w \sim -\partial E/\partial w$)

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//5976061a-e5ca-44d1-952f-84c8ce6f252b/markdown_3/imgs/img_in_image_box_894_294_1126_410.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A33Z%2F-1%2F%2F9a771ae4f65f7203d3b6e586f46dc7571ff6aea769e962b83bf0175708f2eeef" alt="Image" width="19%" /></div>


Energy machine

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//5976061a-e5ca-44d1-952f-84c8ce6f252b/markdown_3/imgs/img_in_image_box_911_460_1122_491.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A33Z%2F-1%2F%2Fc687b69849a0730e7f1f0a114e664e614f6ffebb508b6b38fffeaac1df48127c" alt="Image" width="17%" /></div>


Neural implementation

<div style="text-align: center;"><div style="text-align: center;">d</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//5976061a-e5ca-44d1-952f-84c8ce6f252b/markdown_3/imgs/img_in_image_box_100_549_347_735.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A33Z%2F-1%2F%2Fb6391b4dce3a02f4e1b740a4ac87c85f46d0fbe1333bce4f96276584423b1464" alt="Image" width="20%" /></div>


Clamp input and output neurons

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//5976061a-e5ca-44d1-952f-84c8ce6f252b/markdown_3/imgs/img_in_image_box_357_558_601_719.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A33Z%2F-1%2F%2Fed3d3790ad4851bba44e9853df69e822e5eafdf06c9ce6a148d41c2c7b0233d5" alt="Image" width="20%" /></div>


Learn

Relaxation until convergence

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//5976061a-e5ca-44d1-952f-84c8ce6f252b/markdown_3/imgs/img_in_image_box_613_561_863_718.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A33Z%2F-1%2F%2F96384593813efca01cf9c2f733f7eeb92e596398954f26c4a405b4bee7586f27" alt="Image" width="20%" /></div>


Weight modification

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//5976061a-e5ca-44d1-952f-84c8ce6f252b/markdown_3/imgs/img_in_image_box_866_559_1125_719.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A33Z%2F-1%2F%2F4e857e4dfeff3a92af42974e41632a2b8fe55ab45b3423c5f4afb57b1054a296" alt="Image" width="21%" /></div>


Next time prediction

Predict

Fig. 2 | The energy machine reveals a new understanding of energy-based networks, the mechanism of prospective configuration and its theoretical advantages. A subset of energy-based networks can be visualized as mechanical machines that perform equivalent computations. Here, we present the energy machine corresponding to predictive coding networks $^{[12,18]}$. In the energy machine, the activity of a neuron corresponds to the height of a node (represented by a solid circle) sliding on a post. The input to the neuron is represented by a hollow node on the same post. A synaptic connection corresponds to a rod pointing from a solid node to a hollow node. The weight determines how the input to a postsynaptic neuron depends on the activity of a presynaptic neuron; hence, it influences the angle of the rod. In energy-based networks, relaxation (that is, neural dynamics) and weight modification (that is, weight dynamics) are both driven by minimizing the energy, which corresponds to relaxation of the energy machine by moving the nodes and tuning the rods, respectively. a,b, Predictions (a) and learning (b) in energy-based networks visualized by the energy machine. The pin indicates that neural activity is fixed to the input or target pattern. Here, it is revealed that relaxation infers prospective neural activity, toward which the weights are then modified, a mechanism that we call prospective configuration. c, Physical implementation (top) and connectivity of a predictive coding network $^{[12,18]}$ (bottom), which has dynamics mathematically equivalent to those of the energy machine in the middle (see Methods for details). d, The learning problem in Fig. 1 visualized by the energy machine, which learns to improve the incorrect output while not interfering with the correct output, thanks to the mechanism of prospective configuration.

affecting chances for survival. By contrast, prospective configuration substantially reduces this effect.

Although backpropagation modifies weights to directly reduce cost in the space of weights (that is, performs gradient descent), surprisingly, and rather subversively, it does not push the resulting output activity directly toward the target. To illustrate this, Fig. 3a visualizes the cost with contour lines. Changing the activity of output neurons according to the gradient of the cost would correspond to a change orthogonal to the contour lines, that is, that indicated by the red arrow. However, backpropagation changes the output in a different direction shown by a dashed arrow. Optimizing the weights independently, without considering the effect of updating other weights, leads to output activity not updating toward the target directly due to different weight updates to different layers interfering with each other. By contrast, prospective configuration considers the results of updating other weights by finding a desired configuration of neural activity first. Such a mechanism is missing in backpropagation but is natural in energy-based networks. Supplementary Fig. 2 shows a direct comparison of how these two models evolve in weight and output spaces during learning.
**尽管反向传播在权重空间中通过直接降低代价来修改权重（即执行梯度下降），但令人惊讶且颇具颠覆性的是，它并不会将结果输出活动直接推向目标。为了说明这一点，图3a用等高线可视化了代价。按照代价的梯度来改变输出神经元的活动，对应于垂直于等高线的变化，即红色箭头所示的方向。然而，反向传播以虚线箭头所示的不同方向改变了输出。独立地优化权重，而不考虑更新其他权重的效果，导致输出活动无法直接朝目标更新，因为不同层之间的权重更新会互相干扰。相比之下，前瞻性配置通过首先找到一个期望的神经活动配置，来考虑更新其他权重的结果。这种机制在反向传播中缺失，但在能量基网络中却很自然。补充图2直接比较了这两种模型在权重空间和输出空间中如何演化。**

Interference can be quantified by the angle between the direction of the target (from current output to target) and learning (from current output to output after learning, both measured without the target provided), and we define 'target alignment' as the cosine of this angle (Fig. 3b); hence, high interference corresponds to low target alignment (Fig. 3c).

It is useful to highlight that target alignment is affected little by the learning rate (Fig. 3d), demonstrating that the learning rate has little effect on the direction and trajectory that output neurons take. The difference in target alignment demonstrated in Fig. 3a is also present for deeper and larger (randomly generated) networks (Fig. 3e). When a network has no hidden layers, the target alignment is equal to 1 (Supplementary Notes, Section 2.4.1). The target alignment drops for backpropagation as the network gets deeper because changes in weights in one layer interfere with changes in other layers (Fig. 1), and the backpropagated errors do not lead to appropriate modification of weights in hidden layers (Supplementary Fig. 2). Because backpropagation modifies the weights in the direction reducing loss, it has positive target alignment for small learning rates but not necessarily

<div style="text-align: center;"><div style="text-align: center;">a</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_0/imgs/img_in_chart_box_96_113_508_336.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A32Z%2F-1%2F%2F861700c52a9e512f2629e21ccd27ee33a57be47f3a8d70b0aacb40db0c5372e4" alt="Image" width="34%" /></div>


<div style="text-align: center;"><div style="text-align: center;">b</div> </div>


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


close to 1. By contrast, prospective configuration maintains a much higher value along the way. This higher target alignment of prospective configuration can be theoretically explained by the following: (1) there exists a close link between prospective configuration and an algorithm called target propagation $^{26}$ (shown in Supplementary Fig. 3 and Supplementary Notes, Section 2.2), and (2) under certain conditions, target propagation $^{26}$ has a target alignment of 1 (ref. 27; demonstrated in Supplementary Fig. 4 and Supplementary Notes, Section 2.4.2). Thus, the link with target propagation provides theoretical insight (with numerical verification) into why prospective configuration has a higher target alignment.

Higher target alignment directly translates to the efficiency of learning. Test error during training in a visual classification task with a deep neural network of 15 layers decreases faster for prospective configuration than for backpropagation (Fig. 3f).

Throughout the data presented here, if learning rate is not presented in a plot, the plot corresponds to the best learning rate optimized independently for each rule under the setup via a grid search. The optimization target is either learning performance or similarity to experimental data (details can be found in the methods for each experiment). Thus, for example, Fig. 3f shows the test errors as training progress, with the learning rates optimized independently for each learning rule. The optimization target is the 'mean of test error' during training, reflecting how fast the test error decreases during training. Fig. 3g plots this mean of test error for different learning rates for both learning rules, and the learning rates giving the minima of the curves were used in Fig. 3f. Fig. 3h repeats the experiment on networks of other depths and shows the mean of the test error during training as a function of network depth. The mean error is higher for lower depths, as these networks are unable to learn the task, and for greater depths, as it takes longer to train deeper networks. Importantly, the gap between backpropagation and prospective configuration widens for deeper networks, paralleling the difference in target alignment. Efficient training with deeper networks is important for biological neural systems known to be deep, for example, the primate visual cortex $^{28}$.

In Section 2.3 of the Supplementary Notes, we develop a formal theory of prospective configuration and provide further illustrations and analyses of its advantages. Supplementary Fig. 5 formally defines prospective configuration and demonstrates that it is indeed commonly observed in different energy-based networks. Supplementary Figs. 6 and 7 empirically verify and generalize the advantages expected from the theory and show that prospective configuration yields more accurate error allocation and less erratic weight modification, respectively.

## Advantages of prospective configuration: effective learning in biologically relevant scenarios

Inspired by these advantages, we show empirically that prospective configuration indeed handles various learning problems that biological systems would face better than backpropagation. Because the field of machine learning has developed effective benchmarks for testing learning performance, we use variants of classic machine learning algorithms.

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

First, based on the example in Fig. 1, we expect prospective configuration to require fewer episodes for learning than backpropagation. Before presenting the comparison, we describe how backpropagation is used to train ANNs. Typically, the weights are only modified after a batch of training examples based on the average of updates derived from individual examples (Fig. 4a). In fact, backpropagation relies heavily on averaging over multiple experiences to reach human-level performance $^{32}$, as it needs to stabilize training $^{33}$. By contrast, biological systems must update the weights after each experience, and we compare learning performance in such a setting. Sampling efficiency can be quantified by mean of test error during training, which is shown in Fig. 4b as a function of batch size (number of experiences that the updates are averaged over). Efficiency strongly depends on batch size for backpropagation because it requires batch training to average out erratic weight updates, whereas this dependence is weaker for prospective configuration, where weight changes are intrinsically less erratic and batch averaging is required less (Supplementary Fig. 7). Importantly, prospective configuration learns faster with smaller batch sizes, as in

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_2/imgs/img_in_chart_box_78_104_291_246.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A33Z%2F-1%2F%2Fba2bbc8121d415fb5e25db778cb0f454e29cd2351fbd629679192fdb7963cb51" alt="Image" width="17%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_2/imgs/img_in_image_box_80_103_629_357.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A33Z%2F-1%2F%2F2d6dd9d75a6072c9c17bc5510578e2b8ecb163925fb4074cc09814098d162eb5" alt="Image" width="46%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_2/imgs/img_in_image_box_646_128_1125_294.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A33Z%2F-1%2F%2F1c42521bcbb7478f23128aed4420b2dcd5022b0aa6ffcba5e7c3c894aae1d6c4" alt="Image" width="40%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_2/imgs/img_in_chart_box_98_369_629_559.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A33Z%2F-1%2F%2F5a277d8afb6564e0469975fc8cd9f02acb3ea1a10738cd6e0e14053e7ca51451" alt="Image" width="44%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_2/imgs/img_in_chart_box_658_302_1110_558.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A33Z%2F-1%2F%2F7c96012d577019719662fc834202bdcdb92accabe2c98a4ec7f754b50f485e11" alt="Image" width="37%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Fig. 5 | Prospective configuration explains contextual inference in human sensorimotor learning. a, Structure of an experimental trial where participants were asked to move a stick from the starting point to the target point while experiencing perturbations. b, The minimal network for the task, including six connections encoding the associations from the backgrounds (B and R) to the belief of contexts ([B] and [R]) and from the belief of contexts to the prediction of perturbations (+ and -). c–e, Sequence of sessions the participants experienced, including training (c), washout (d) and testing (e). Darker gray boxes show the</div> </div>


<div style="text-align: center;"><div style="text-align: center;">expected network after the session, where thickness represents the strength of connections. In the testing session, the darker box explains how the two learning rules learn differently on the R+ trial, leading to the differences in f.f. Predictions of the two learning rules compared to behavioral data measured from human participants, where prospective configuration reproduces the key patterns of data, but backpropagation does not. Each experiment was repeated with n=24 random seeds, as there were 24 participants in the behavioral experiment.</div> </div>


biological settings. Additionally, final performance can be quantified by the minimum of the test error, which is shown in Fig. 4c, when trained with a batch size equal to 1. Here, prospective configuration also demonstrates a notable advantage over backpropagation.

Second, biological organisms need to sequentially learn multiple tasks, while ANNs show catastrophic forgetting. When trained on a new task, performance on previously learned tasks is largely destroyed $^{16,34}$. The data in Fig. 4d show performance when trained on two tasks alternately (task 1 is classifying five randomly selected classes in the FashionMNIST dataset, and task 2 is classifying the remaining five classes). Prospective configuration outperforms backpropagation both in terms of avoiding forgetting previous tasks and relearning current tasks. The results are summarized in Fig. 4e.

Third, biological systems often need to rapidly adapt to changing environments. A common way to simulate this is ‘concept drifting’ $^{31}$, where a part of the mapping between the output neurons to the semantic meaning is shuffled regularly, each time a certain number of training iterations has passed (Fig. 4f). Test error during training with concept drifting is presented in Fig. 4f. Before epoch 0, both learning rules are initialized with the same pretrained model (trained with backpropagation); thus, epoch 0 is the first time the model experiences concept drift. The results are summarized in Fig. 4g and show that, for this task, there is a particularly large difference in mean error (for optimal learning rates). This large advantage of prospective configuration is related to it being able to optimally detect which weights to modify (Supplementary Fig. 6) and to preserve existing knowledge while adapting to changes (Fig. 1). This ability to maintain important information while updating other information is critical for survival in natural environments that are bound to change, and prospective configuration has a very substantial advantage in this respect.

Furthermore, biological learning is also characterized by limited data availability. Prospective configuration outperforms backpropagation when the model is trained with fewer examples (Fig. 4h).

To demonstrate that the advantage of prospective configuration also scales up to larger networks and problems, we evaluated convolutional neural networks $^{35}$ on CIFAR-10 (ref. 36) trained with both learning rules (Fig. 4i), where prospective configuration showed notable advantages over backpropagation. The detailed structure of the convolutional networks is provided in Fig. 4j.

Another key challenge for biological systems is to decide which actions to take. Reinforcement learning theories (for example, Q learning) propose that it is solved by learning the expected reward resulting from different actions in different situations $^{37}$. Such prediction of rewards can be made by neural networks $^{4}$, which can be trained with prospective configuration or backpropagation. The sum of rewards per episode during training on three classic reinforcement learning tasks is reported in Fig. 4k, where prospective configuration demonstrates a notable advantage over backpropagation. This large advantage may arise because reinforcement learning is particularly sensitive to erratic changes in network weights (as the target output depends on reward predicted by the network itself for a new state; Methods).

Based on the superior learning performance of prospective configuration, we may expect that this learning mechanism has been favored by evolution; thus, in the next sections, we investigate if it can account for neural activity and behavior during learning better than backpropagation.

## Evidence for prospective configuration: inferring the latent state during learning

Prospective configuration is related to theories proposing that before learning, the brain first infers a latent state of the environment from feedback $^{38-40}$. Here, we propose that this inference can be achieved in neural circuits through prospective configuration, where, following feedback, neurons in ‘hidden layers’ converge to a prospective pattern of activity that encodes this latent state. We demonstrate that data from various previous studies, which involved the inference of a

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_3/imgs/img_in_image_box_77_108_579_271.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A34Z%2F-1%2F%2F53fdc91799e6acc89da67975338efc5d494446674ccb0dc5b99e713d80ed0a94" alt="Image" width="42%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Fig. 6 | Prospective configuration can discover the underlying task structure during reinforcement learning. a, Reinforcement learning task. Human participants were required to choose between two options, leading to either reward (gaining coins) or punishment (losing coins) with different probabilities. The probability of reward was occasionally reversed between the two options. b, The minimal network encoding the essential elements of the task. c, Activity of the output neuron corresponding to the selected option from networks trained</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//acb60c44-58a8-4520-be69-f62f556e17a7/markdown_3/imgs/img_in_chart_box_603_116_1126_276.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-14T02%3A59%3A34Z%2F-1%2F%2Fbd77902e4a44d62188ebdf6866db5d79d9ada45351f5d1c6b3646500ed158d57" alt="Image" width="43%" /></div>


<div style="text-align: center;"><div style="text-align: center;">with prospective configuration and backpropagation compared with fMRI data measured in human participants (that is, peak blood oxygenation level-dependent (%BOLD) signal in the mPFC). Prospective configuration reproduces the key finding that the expected value (encoded in %BOLD signal in the mPFC) increases if the next choice after a punishing trial is to switch to the other option. The number of trials is not mentioned in the original paper, so we simulated for n = 128 trials for both learning rules. Error bars represent the 68% confidence interval.</div> </div>


latent state, can be explained by prospective configuration. These data were previously explained by complex and abstract mechanisms, such as Bayesian models $^{38,39}$, whereas here, we mechanically show with prospective configuration how such inference can be performed by minimal networks encoding only the essential elements of the tasks.

The dynamical inference of a latent state from feedback has been recently proposed to take place during sensorimotor learning $^{39}$. In this experiment, participants received different motor perturbations in different contexts and learned to compensate for these perturbations. Behavioral data suggest that, after receiving feedback, participants first used the feedback to infer context and then adapted the force for the inferred context. We demonstrate that prospective configuration is able to reproduce these behavioral data, whereas backpropagation cannot.

Specifically, in the task (Fig. 5a), participants were asked to move a stick from a starting point to a target point while experiencing perturbations. The participants experienced a sequence of blocks of trials (Fig. 5c–e), including training, washout and testing. During the training session, different directions of perturbations, positive (+) or negative (-), were applied in different contexts, blue (B) or red (R) backgrounds, respectively. We denote these trials as B+ and R−. These trials may be associated with latent states, which we denote [B] and [R]; for example, the latent state [B] may be associated with both background B and perturbation +. The next stage of the task was designed to investigate if the latent state [B] can be activated by perturbation + even if no background B is shown. Thus, participants experienced different trials including R+ (that is, perturbation + but no background B). Specifically, after a washout session (during which no perturbation was provided), in the testing session, participants experienced one of the four possible test trials: B+, R+, B− and R−. To evaluate learning on the test trials, motor adaptation (that is, the difference between the final and target stick positions) was measured before and after the test trial in two trials with the blue background (Fig. 5e). Change in the adaptation between these two trials is a reflection of learning about blue context that occurred at the test trial. If participants only associated feedback with the background color (B), then the change in adaptation would only occur with test trials B+ and B−. However, experimental data (Fig. 5f) show that there was also substantial adaptation change with R+ trials (which was even bigger than with B− trials).

To model learning in this task, we considered a neural network (Fig. 5b) where input nodes encode the background color, and outputs encode movement compensations in the two directions. Importantly, this network also includes hidden neurons encoding belief of being in the contexts associated with the two backgrounds ([B] and [R]). Trained with the exact procedure of the experiment $^{39}$ from randomly initialized weights, prospective configuration with this minimal network can reproduce the behavioral data, whereas backpropagation cannot (Fig. 5f).

Prospective configuration can produce change in adaptation with the R+ test trial because after + feedback, it is able to also activate context [B] that was associated with this feedback during training and then learn compensation for this latent state. To shed light on how this inference takes place in the model, schematics in Fig. 5c, d show evolution of the weights of the network over sessions (thickness represents the strength of connections). The schematic in Fig. 5e shows the difference between the two learning rules after exposure to R+; although B is not perceived, prospective configuration infers a moderate excitation of the belief of blue context [B] because the positive connection from [B] to + was built during the training session. The activity of [B] enables the learning of weights from [B] to + and −, while backpropagation does not modify any weights originating from [B].

For simplicity of explanation, we presented simulations with minimal networks; however, Supplementary Fig. 8 shows that networks with a general fully connected structure and more hidden neurons can replicate the above data when using prospective configuration but not when using backpropagation.

Studies of animal conditioning have also observed that feedback in learning tasks involving multiple stimuli may trigger learning about non-presented stimuli $^{41,42}$. One example is provided in Supplementary Fig. 9, where we show that it can be explained by prospective configuration but not by backpropagation.

### Evidence for prospective configuration: discovering task structure during learning

Prospective configuration is also able to discover the underlying task structure in reinforcement learning. Specifically, we consider a task where reward probabilities of different options were not independent $^{38}$. In this study, humans were choosing between two options where the reward probabilities were constrained such that one option had a higher reward probability than the other (Fig. 6a). Occasionally the reward probabilities were swapped, so if one probability was increased, the other was decreased by the same amount. Remarkably, the recorded functional magnetic resonance imaging (fMRI) data suggested that participants learned that the values of the two options were negatively correlated and on each trial updated the value estimates of both options in opposite ways. This conclusion was drawn from analysis of the signal from the medial prefrontal cortex (mPFC), which encoded the expected value of reward. The data presented in Fig. 6c compare this signal after making a choice on two consecutive trials: a trial in which the reward was not received ('punish trial') and the next trial. If the participant selected the same option on both trials ('stay'), the signal decreased, indicating that the reward expected by the participant was reduced. Remarkably, if the participant selected the other option on the next trial ('switch'), the signal increased, suggesting that negative feedback for one option increased the value estimate for

the other. Such learning is not predicted by standard reinforcement learning models $^{38}$.

This task can be conceptualized as having a latent state encoding which option is superior, and this latent state determines the reward probabilities for both options. Consequently, we consider a neural network reflecting this structure (Fig. 6b) that includes an input neuron encoding being in the task (equal to 1 in simulations), a hidden neuron encoding the latent state and two output neurons encoding the reward probabilities for the two options. Trained with the exact procedure of the experiment $^{38}$ from randomly initialized weights, prospective configuration with this minimal network can reproduce the data, whereas backpropagation cannot (Fig. 6c). In Supplementary Fig. 10, we show that prospective configuration reproduces these data because it can infer the rewarded choice by updating the activity of the hidden neuron based on feedback.

Taken together, the presented simulations illustrate that prospective configuration is a common principle that can explain a range of surprising learning effects in diverse tasks.

## Discussion

Our paper identifies the principle of prospective configuration, according to which learning relies on neurons first optimizing their pattern of activity to match the correct output and then reinforcing these prospective activities through synaptic plasticity. Although it was known that in energy-based networks the activity of neurons shifts before weight update, it has been previously thought that this shift is a necessary cost of error propagation in biological networks, and several methods have been proposed to suppress it $^{[1,12,14,20,21]}$ to approximate backpropagation more closely. By contrast, we demonstrate that this reconfiguration of neural activity is the key to achieving learning performance superior to that of backpropagation and to explaining experimental data from diverse learning tasks. Prospective configuration further offers a range of experimental predictions distinct from those of backpropagation (Supplementary Figs. 11 and 12). Together, we have demonstrated that prospective configuration enables more efficient learning than backpropagation by reducing interference, demonstrates superior performance in situations faced by biological organisms, requires only local computation and plasticity and matches experimental data across a wide range of tasks.

Our theory addresses a long-standing question of how the brain solves the plasticity-stability dilemma, for example, how it is possible that, despite adjustment of representation in the primary visual cortex during learning $^{43}$, we can still understand the meaning of visual stimuli we learned over our lifetime. According to prospective configuration, when some weights are modified, compensatory changes are made to other weights to ensure the stability of correctly predicted outputs. Thus, prospective configuration reduces interference between different weight modifications while learning a single association. Previous computational models have proposed mechanisms that reduce interference between new and previously acquired information while learning multiple associations $^{34,44}$. It is highly likely that such mechanisms and prospective configuration operate in the brain in parallel to minimize both types of interference.

Prospective configuration is related to inference and learning procedures in statistical modeling. If the ‘energy’ in energy-based schemes is variational free energy, prospective configuration can be seen as an implementation of variational Bayes that subsumes inference and learning $^{45}$. For example, dynamic expectation maximization $^{46,47}$ can be regarded as a generalization of predictive coding networks in which the D-step optimizes representations of latent states (analogously to relaxation until convergence during inference) while the E-step optimizes model parameters (analogously to weight modification during learning).

Other recent work $^{48,49}$ also noticed that the natural form of energy-based networks ('strong control' in their words) performs different learning than backpropagation. Their analysis concentrates on an architecture of deep feedback control, and they demonstrated that a particular form of their model is equivalent to predictive coding networks $^{49}$. The unique contribution of our paper is to show the benefits of such strong control and explain why they arise. The principle of prospective configuration is also present in other recent models. For example, Gilra and Gerstner $^{50}$ developed a spiking model in which feedback about the error on the output directly affects the activity of hidden neurons before plasticity takes place. Haider et al. $^{51}$ developed a faster inference algorithm for energy-based models that computes a value to which the activity is likely to converge, termed latent equilibrium $^{51}$. Iteratively setting each neuron's output based on its latent equilibrium leads to much faster inference $^{51}$ and enables efficient computation of the prospective configuration.

Predictive coding networks require symmetric forward and backward weights between layers of neurons, so a question arises concerning how such symmetry may develop in the brain. If predictive coding networks are initialized with symmetric weights (as in our simulations), the symmetry will persist because the changes in weight between neurons A and B are the same as those for feedback weight (between neurons B and A). Even if the weights are not initialized symmetrically, the symmetry may develop if synaptic decay is included in the model $^{52}$ because then the initial asymmetric values decay away, and weight values become more influenced by recent changes that are symmetric. Nevertheless, weight symmetry is not generally required for effective credit assignment $^{53,54}$.

Here, we assumed for simplicity that the convergence of neural activity to an equilibrium happens rapidly after the stimuli are provided so that the synaptic weight modification after convergence may take place while the stimuli are still present. Nevertheless, predictive coding networks can still work even if weight modification takes place while the neural activity is converging. Specifically, Song et al. demonstrated that if neural activities are only updated for the first few steps, the update of the weights is equivalent to that in backpropagation $^{14}$. As a reminder, we demonstrate here that if the neural activities are updated to equilibrium, the update of the weights follows the principle of prospective configuration and possesses the desirable demonstrated properties. Thus, a learning rule where neural activities and weights are updated in parallel will experience a weight update that is equivalent to backpropagation at the start and then move to prospective configuration as the system converges to equilibrium $^{55}$. Furthermore, predictive coding networks have been extended to describe recurrent structures $^{56-58}$ and it has been shown that such networks can learn to predict dynamically changing stimuli even if weights are modified before the activity converged for a given 'frame' of the stimulus $^{57}$.

The advantages of prospective configuration suggest that it may be profitably applied in machine learning to improve the efficiency and performance of deep neural networks. An obstacle for this is that the relaxation phase is computationally expensive. However, recent work demonstrated that by modifying weights after each step of relaxation, the model becomes comparably fast to backpropagation and easier for parallelization $^{55}$.

Most intriguingly, it has been demonstrated that the speed of energy-based networks can be greatly increased by implementing the relaxation on analog hardware $^{59}$, potentially resulting in energy-based networks being faster than backpropagation. Therefore, we anticipate that our discoveries may change the blueprint of next-generation machine learning hardware, switching from the current digital tensor base to analog hardware and being closer to the brain and potentially far more efficient.

#### Online content

Any methods, additional references, Nature Portfolio reporting summaries, source data, extended data, supplementary information, acknowledgements, peer review information; details of author contributions and competing interests; and statements of data and code availability are available at https://doi.org/10.1038/s41593-023-01514-1.


