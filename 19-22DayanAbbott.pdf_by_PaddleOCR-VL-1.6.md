# 1 Neural Encoding I: Firing Rates and Spike Statistics

## 1.1 Introduction

Neurons are remarkable among the cells of the body in their ability to propagate signals rapidly over large distances. They do this by generating characteristic electrical pulses called action potentials or, more simply, spikes that can travel down nerve fibers. Neurons represent and transmit information by firing sequences of spikes in various temporal patterns. The study of neural coding, which is the subject of the first four chapters of this book, involves measuring and characterizing how stimulus attributes, such as light or sound intensity, or motor actions, such as the direction of an arm movement, are represented by action potentials.

The link between stimulus and response can be studied from two opposite points of view. Neural encoding, the subject of chapters 1 and 2, refers to the map from stimulus to response. For example, we can catalog how neurons respond to a wide variety of stimuli, and then construct models that attempt to predict responses to other stimuli. Neural decoding refers to the reverse map, from response to stimulus, and the challenge is to reconstruct a stimulus, or certain aspects of that stimulus, from the spike sequences it evokes. Neural decoding is discussed in chapter 3. In chapter 4, we consider how the amount of information encoded by sequences of action potentials can be quantified and maximized. Before embarking on this tour of neural coding, we briefly review how neurons generate their responses and discuss how neural activity is recorded. The biophysical mechanisms underlying neural responses and action potential generation are treated in greater detail in chapters 5 and 6.

#### Properties of Neurons

Neurons are highly specialized for generating electrical signals in response to chemical and other inputs, and transmitting them to other cells. Some important morphological specializations, seen in figure 1.1, are the dendrites that receive inputs from other neurons and the axon that carries the neuronal output to other cells. The elaborate branching structure of
the dendritic tree allows a neuron to receive inputs from many other neurons through synaptic connections. The cortical pyramidal neuron of figure 1.1A and the cortical interneuron of figure 1.1C each receive thousands of synaptic inputs, and for the cerebellar Purkinje cell of figure 1.1B the number is over 100,000. Figure 1.1 does not show the full extent of the axons of these neurons. Axons from single neurons can traverse large fractions of the brain or, in some cases, of the entire body. In the mouse brain, it has been estimated that cortical neurons typically send out a total of about 40 mm of axon and have approximately 4 mm of total dendritic cable in their branched dendritic trees. The axon makes an average of 180 synaptic connections with other neurons per mm of length and the dendritic tree receives, on average, 2 synaptic inputs per  $\mu$m. The cell body or soma of a typical cortical neuron ranges in diameter from about 10 to 50  $\mu$m.
神经元高度特化，能够响应化学信号及其他输入产生电信号，并将其传递给其他细胞。如图1.1所示，一些重要的形态学特化结构包括：树突（dendrite）——接收来自其他神经元的输入；以及轴突（axon）——将神经元的输出传递至其他细胞。树突复杂的分支结构使得神经元能够通过突触连接接收来自大量其他神经元的输入。图1.1A中的皮层锥体神经元（cortical pyramidal neuron）和图1.1C中的皮层中间神经元（cortical interneuron）各自接收数千个突触输入，而图1.1B中小脑浦肯野细胞（cerebellar Purkinje cell）的突触输入数量则超过100,000个。图1.1并未展示这些神经元轴突的完整延伸范围。单个神经元的轴突可以延伸穿过大脑的大部分区域，在某些情况下甚至可以贯穿整个身体。在小鼠脑中，据估计皮层神经元通常发出的轴突总长度约为40 mm，而其分支树突树中的树突总长度约为4 mm。轴突每毫米长度平均与其他神经元形成180个突触连接，而树突树平均每微米接收2个突触输入。典型皮层神经元的细胞体（胞体，soma）直径约为10至50 μm。

hyperpolarization and depolarization

Along with these morphological features, neurons have physiological specializations. Most prominent among these are a wide variety of membrane-spanning ion channels that allow ions, predominantly sodium (Na⁺), potassium (K⁺), calcium (Ca²⁺), and chloride (Cl⁻), to move into and out of the cell. Ion channels control the flow of ions across the cell membrane by opening and closing in response to voltage changes and to both internal and external signals.
除这些形态学特征外，神经元还具有生理学特化。其中最显著的是多种多样的跨膜离子通道（membrane-spanning ion channels），它们允许离子（主要是钠离子 Na⁺、钾离子 K⁺、钙离子 Ca²⁺ 和氯离子 Cl⁻）进出细胞。离子通道通过响应电压变化以及内部和外部信号而开启和关闭，从而控制离子跨细胞膜的流动。

The electrical signal of relevance to the nervous system is the difference in electrical potential between the interior of a neuron and the surrounding extracellular medium. Under resting conditions, the potential inside the cell membrane of a neuron is about -70 mV relative to that of the surrounding bath (which is conventionally defined to be 0 mV), and the cell is said to be polarized. Ion pumps located in the cell membrane maintain concentration gradients that support this membrane potential difference. For example, Na⁺ is much more concentrated outside a neuron than inside it, and the concentration of K⁺ is significantly higher inside the neuron than in the extracellular medium. Ions thus flow into and out of a cell due to both voltage and concentration gradients. Current in the form of positively charged ions flowing out of the cell (or negatively charged ions flowing into the cell) through open channels makes the membrane potential more negative, a process called hyperpolarization. Current flowing into the cell changes the membrane potential to less negative or even positive values. This is called depolarization.
与神经系统相关的电信号是神经元内部与周围细胞外介质（extracellular medium）之间的电位差。在静息状态下，神经元细胞膜内的电位相对于周围浴液（通常定义为 0 mV）约为 -70 mV，此时细胞处于**极化**（polarized）状态。位于细胞膜上的离子泵（ion pumps）维持着支持这一膜电位差的浓度梯度。例如，神经元外的 Na⁺ 浓度远高于内部，而 K⁺ 在神经元内的浓度则显著高于细胞外介质。因此，离子在电压梯度和浓度梯度的共同驱动下进出细胞。通过开放通道流出细胞的正离子（或流入细胞的负离子）所形成的电流使膜电位变得更负，这一过程称为**超极化（hyperpolarization）**。流入细胞的电流使膜电位变为不那么负、甚至为正值，这称为**去极化（depolarization）**。

refractory period

If a neuron is depolarized sufficiently to raise the membrane potential above a threshold level, a positive feedback process is initiated, and the neuron generates an action potential. An action potential is a roughly 100 mV fluctuation in the electrical potential across the cell membrane that lasts for about 1 ms (figure 1.2A). Action potential generation also depends on the recent history of cell firing. For a few milliseconds just after an action potential has been fired, it may be virtually impossible to initiate another spike. This is called the absolute refractory period. For a longer interval known as the relative refractory period, lasting up to tens of milliseconds after a spike, it is more difficult to evoke an action potential.
如果神经元被充分去极化，使膜电位升高至超过阈值水平，就会启动一个正反馈过程，神经元随即产生**动作电位（action potential）**。动作电位是跨细胞膜电位约100 mV的波动，持续约1毫秒（图1.2A）。动作电位的产生还取决于细胞近期的放电历史。在动作电位发放后的最初几毫秒内，几乎不可能引发另一个锋电位（spike），这称为**绝对不应期（absolute refractory period）**。在随后更长的一段时间内，即**相对不应期（relative refractory period）**，可在一次锋电位后持续长达数十毫秒，此时诱发动作电位变得更加困难。


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//9a1863c0-ee37-43ec-bcdd-3f788557224f/markdown_2/imgs/img_in_image_box_302_244_848_859.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-11T14%3A02%3A29Z%2F-1%2F%2F6e9e0565c3fd4f1b2c260fb4e09d59745615df9551808f279a6f1e8c52082979" alt="Image" width="45%" /></div>



<div style="text-align: center;"><div style="text-align: center;">Figure 1.1 Diagrams of three neurons. (A) A cortical pyramidal cell. These are the primary excitatory neurons of the cerebral cortex. Pyramidal cell axons branch locally, sending axon collaterals to synapse with nearby neurons, and also project more distally to conduct signals to other parts of the brain and nervous system. (B) A Purkinje cell of the cerebellum. Purkinje cell axons transmit the output of the cerebellar cortex. (C) A stellate cell of the cerebral cortex. Stellate cells are one of a large class of interneurons that provide inhibitory input to the neurons of the cerebral cortex. These figures are magnified about 150-fold. (Drawings from Cajal, 1911; figure from Dowling, 1992.)</div> </div>


Action potentials are of great importance because they are the only form of membrane potential fluctuation that can propagate over large distances. Subthreshold potential fluctuations are severely attenuated over distances of 1 mm or less. Action potentials, on the other hand, are regenerated actively along axon processes and can travel rapidly over large distances without attenuation.

Axons terminate at synapses where the voltage transient of the action potential opens ion channels, producing an influx of  $ Ca^{2+} $ that leads to the release of a neurotransmitter (figure 1.2B). The neurotransmitter binds to receptors at the signal-receiving or postsynaptic side of the synapse,

synapse

<div style="text-align: center;"><div style="text-align: center;">A</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//9a1863c0-ee37-43ec-bcdd-3f788557224f/markdown_3/imgs/img_in_image_box_285_212_534_568.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-11T14%3A02%3A30Z%2F-1%2F%2F9c9038c6e58bb9ae74e64262c9e4c98c7741c17ad7a1061164e17e8a5e851021" alt="Image" width="20%" /></div>


sharp and patch
electrodes

<div style="text-align: center;"><div style="text-align: center;">B</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//9a1863c0-ee37-43ec-bcdd-3f788557224f/markdown_3/imgs/img_in_image_box_539_214_843_566.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-11T14%3A02%3A30Z%2F-1%2F%2F3972efb5978a0864d4f690727b88a79aeef1f3663fc76eefa31daa6edd4622cf" alt="Image" width="25%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 1.2 (A) An action potential recorded intracellularly from a cultured rat neocortical pyramidal cell. (B) Diagram of a synapse. The axon terminal or bouton is at the end of the axonal branch seen entering from the top of the figure. It is filled with synaptic vesicles containing the neurotransmitter that is released when an action potential arrives from the presynaptic neuron. Transmitter crosses the synaptic cleft and binds to receptors on the dendritic spine, a process roughly 1  $ \mu $m long that extends from the dendrite of the postsynaptic neuron. Excitatory synapses onto cortical pyramidal cells form on dendritic spines as shown here. Other synapses form directly on the dendrites, axon, or soma of the postsynaptic neuron. (A recorded by L. Rutherford in the laboratory of G. Turrigiano; B adapted from Kandel et al., 1991.)</div> </div>


causing ion-conducting channels to open. Depending on the nature of the ion flow, the synapses can have either an excitatory, depolarizing, or an inhibitory, typically hyperpolarizing, effect on the postsynaptic neuron.

### Recording Neuronal Responses

Figure 1.3 illustrates intracellular and extracellular methods for recording neuronal responses electrically (they can also be recorded optically). Membrane potentials are measured intracellularly by connecting a hollow glass electrode filled with a conducting electrolyte to a neuron, and comparing the potential it records with that of a reference electrode placed in the extracellular medium. Intracellular recordings are made either with sharp electrodes inserted through the membrane into the cell, or patch electrodes that have broader tips and are sealed tightly to the surface of the membrane. After the patch electrode seals, the membrane beneath its tip is either broken or perforated, providing electrical contact with the interior of the cell. The top trace in figure 1.3 is a schematic of an intracellular recording from the soma of a neuron firing a sequence of action potentials. The recording shows rapid spikes riding on top of a more slowly varying subthreshold potential. The bottom trace is a schematic of an intracellular recording made some distance out on the axon of the neuron. These traces

