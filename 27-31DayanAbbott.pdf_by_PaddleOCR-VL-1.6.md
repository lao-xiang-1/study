In the same way that the response function  $\rho(t)$ can be averaged across trials to give the firing rate  $r(t)$, the spike-count firing rate can be averaged over trials, yielding a quantity that we refer to as the average firing rate. This is denoted by  $\langle r \rangle$ and is given by

 $$ \left\langle r\right\rangle=\frac{\left\langle n\right\rangle}{T}=\frac{1}{T}\int_{0}^{T}d\tau\left\langle\rho(\tau)\right\rangle=\frac{1}{T}\int_{0}^{T}d t\:\rhd(t)\:. $$ 

The first equality indicates that  $\langle r \rangle$ is just the average number of spikes per trial divided by the trial duration. The third equality follows from the equivalence of the firing rate and the trial-averaged neural response function within integrals (equation 1.6). The average firing rate is equal to both the time average of  $r(t)$ and the trial average of the spike-count rate r. Of course, a spike-count rate and average firing rate can be defined by counting spikes over any time period, not necessarily the entire duration of a trial.

The term “firing rate” is commonly used for all three quantities,  $r(t)$, r, and  $\langle r\rangle$. Whenever possible, we use the terms “firing rate”, “spike-count rate”, and “average firing rate” for  $r(t)$, r, and  $\langle r\rangle$, respectively, but when this becomes too cumbersome, the different mathematical notations serve to distinguish them. In particular, we distinguish the spike-count rate r from the time-dependent firing rate  $r(t)$ by using a different font and by including the time argument in the latter expression (unless  $r(t)$ is independent of time). The difference between the fonts is rather subtle, but the context should make it clear which rate is being used.

### Measuring Firing Rates

> **主要内容**：怎么将 测量到的动作电位峰值（冲激函数）转换成 连续的曲线

The firing rate  $r(t)$ cannot be determined exactly from the limited data available from a finite number of trials. In addition, there is no unique way to approximate  $r(t)$. A discussion of the different methods allows us to introduce the concept of a linear filter and kernel that will be used extensively in the following chapters. We illustrate these methods by extracting firing rates from a single trial, but more accurate results could be obtained by averaging over multiple trials.

average firing
rate  $\langle r \rangle$

<div style="text-align: center;"><div style="text-align: center;">A</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//570f5fc9-ffc4-429f-953a-0e87d75658fe/markdown_1/imgs/img_in_chart_box_289_217_858_759.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-12T16%3A54%3A22Z%2F-1%2F%2Fdd1a887574737097037fbc845c7042e03912b82e53360cc75ba4958452f28724" alt="Image" width="47%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 1.4 Firing rates approximated by different procedures. (A) A spike train from a neuron in the inferotemporal cortex of a monkey recorded while that animal watched a video on a monitor under free viewing conditions. (B) Discretetime firing rate obtained by binning time and counting spikes with  $\Delta t = 100$ ms. (C) Approximate firing rate determined by sliding a rectangular window function along the spike train with  $\Delta t = 100$ ms. (D) Approximate firing rate computed using a Gaussian window function with  $\sigma_{t} = 100$ ms. (E) Approximate firing rate using the window function of equation 1.12 with  $1/\alpha = 100$ ms. (Data from Baddeley et al., 1997.)</div> </div>

The jagged curve in figure 1.4C shows the result of sliding a 100 ms wide window along the spike train. 
图C用的方法和图B一样，只不过将采样间隔缩减到了100ms

The firing rate approximated in this way can be expressed as the sum of a window function：
$$ r_{\mathrm{a p p r o x}}(t)=\sum_{i=1}^{n}w\left(t-t_{i}\right) \quad(1.8)$$

Use of a sliding window avoids the arbitrariness of bin placement and produces a rate that might appear to have a better temporal resolution. However, it must be remembered that the rates obtained at times separated by less than one bin width are correlated because they involve some of the same spikes.

The sum in equation 1.8 can also be written as the integral of the window function times the neural response function (see equation 1.2):
$$ r_{\mathrm{a p p r o x}}(t)=\int_{-\infty}^{\infty}d\tau\;w(\tau)\rho(t-\tau) \quad(1.10) $$
- $\rho(t) = \sum_{i=1}^{n} \delta(t - t_i)$

The integral in equation 1.10 is called a linear filter, and the window function $w$, also called the filter kernel, specifies how the neural response function evaluated at time $t - \tau$ contributes to the firing rate approximated at time $t$.

The jagged appearance of the curve in figure 1.4C is caused by the discontinuous shape of the window function used. An approximate firing rate can be computed using virtually any window function  $w(\tau)$ that goes to 0 outside a region near  $\tau = 0$, provided that its time integral is equal to 1. For example, instead of the rectangular window function used in figure 1.4C,  $w(\tau)$ can be a Gaussian:
$$ w(\tau)=\frac{1}{\sqrt{2\pi}\sigma_{w}}\exp\left(-\frac{\tau^{2}}{2\sigma_{w}^{2}}\right). $$ 

In this case,  $\sigma_{w}$ controls the temporal resolution of the resulting rate, playing a role analogous to  $\Delta t$. A continuous window function like the Gaussian used in equation 1.8 generates a firing-rate estimate that is a smooth function of time (figure 1.4D).

Both the rectangular and the Gaussian window functions approximate the firing rate at any time, using spikes fired both before and after that time. A postsynaptic neuron monitoring the spike train of a presynaptic cell has access only to spikes that have previously occurred. An approximation of the firing rate at time $t$ that depends only on spikes fired before $t$ can be calculated using a window function that vanishes when its argument
矩形窗口函数和高斯窗口函数两者都可以近似任意时刻的放电率，它们使用的是该时刻之前和之后发放的spike。然而，一个监测突触前细胞spike序列的突触后神经元，只能获取到先前已经发生的spike。因此，仅依赖于时刻之前发放的spike的放电率近似值，可以使用一个当其自变量为负时归零的窗口函数来计算。
$$ w(\tau)=[\alpha^{2}\tau\exp(-\alpha\tau)]_{+} $$ - 其中，符号 $[\;]_+$ 表示半波整流运算（相当于relu）：
$$ [z]_{+}=\left\{\begin{array}{l l}{z}&{\mathrm{i f~}z\geq0}\\ {0}&{\mathrm{o t h e r w i s e}.}\end{array}\right. $$
 
where $1/\alpha$ determines the temporal resolution of the resulting firing-rate estimate. 
其中  $1/\alpha$ 决定了所得放电率估计的时间分辨率。

Figure 1.4E shows the firing rate approximated by such a causal scheme. Note that this rate tends to peak later than the rate computed in figure 1.4D using a temporally symmetric window function.
图1.4E展示了采用这种因果方案所得到的放电率近似值。注意，该放电率的峰值往往比图1.4D中采用时域对称窗口函数计算的放电率更晚出现。
 

## Tuning Curves

Neuronal responses typically depend on many different properties of a stimulus. In this chapter, we characterize responses of neurons as functions of just one of the stimulus attributes to which they may be sensitive. The value of this single attribute is denoted by s. In chapter 2, we consider more complete stimulus characterizations.

A simple way of characterizing the response of a neuron is to count the number of action potentials fired during the presentation of a stimulus. This approach is most appropriate if the parameter s characterizing the stimulus is held constant over the trial. If we average the number of action potentials fired over (in theory, an infinite number of) trials and divide by the trial duration, we obtain the average firing rate,  $\langle r \rangle$, defined in equation 1.7. The average firing rate written as a function of  $s$,  $\langle r \rangle = f(s)$, is called the neural response tuning curve. The functional form of a tuning curve depends on the parameter s used to describe the stimulus. The precise choice of parameters used as arguments of tuning curve functions is partially a matter of convention. Because tuning curves correspond to firing rates, they are measured in units of spikes per second or Hz.

Figure 1.5A shows extracellular recordings of a neuron in the primary visual cortex (V1) of a monkey. While these recordings were being made, a bar of light was moved at different angles across the region of the visual field where the cell responded to light. This region is called the receptive field of the neuron. Note that the number of action potentials fired depends on the angle of orientation of the bar. The same effect is shown in figure 1.5B in the form of a response tuning curve, which indicates how the average firing rate depends on the orientation of the light bar stimulus. The data have been fitted by a response tuning curve of the form

 $$ f(s)=r_{\mathrm{m a x}}\exp\left(-\frac{1}{2}\left(\frac{s-s_{\mathrm{m a x}}}{\sigma_{f}}\right)^{2}\right), $$ 

<div style="text-align: center;"><div style="text-align: center;">A</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//50f062e4-04a0-432b-817d-cd7c02e7645b/markdown_0/imgs/img_in_image_box_292_204_475_480.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-12T16%3A54%3A21Z%2F-1%2F%2F1f3867f24f8a718b6193c6d8509e048a963420e163f81f137f7cbe52552557bc" alt="Image" width="15%" /></div>


<div style="text-align: center;"><div style="text-align: center;">B</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//50f062e4-04a0-432b-817d-cd7c02e7645b/markdown_0/imgs/img_in_chart_box_488_221_866_500.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-12T16%3A54%3A21Z%2F-1%2F%2F00263c299d8cc73a0dac26e205952868cb24fd2b385d42ddfbce45049b021af3" alt="Image" width="31%" /></div>


<div style="text-align: center;"><div style="text-align: center;">Figure 1.5 (A) Recordings from a neuron in the primary visual cortex of a monkey. A bar of light was moved across the receptive field of the cell at different angles. The diagrams to the left of each trace show the receptive field as a dashed square and the light source as a black bar. The bidirectional motion of the light bar is indicated by the arrows. The angle of the bar indicates the orientation of the light bar for the corresponding trace. (B) Average firing rate of a cat V1 neuron plotted as a function of the orientation angle of the light bar stimulus. The curve is a fit using the function 1.14 with parameters  $r_{max} = 52.14$ Hz,  $s_{max} = 0^{\circ}$, and  $\sigma_{f} = 14.73^{\circ}$. (A adapted from Wandell, 1995, based on an original figure from Hubel and Wiesel, 1968; B data points from Henry et al., 1974.)</div> </div>


where $s$ is the orientation angle of the light bar, $s_{\max}$ is the orientation angle evoking the maximum average response rate $r_{\max}$ (with $s - s_{\max}$ taken to lie in the range between $-90^{\circ}$ and $+90^{\circ}$), and $\sigma_{f}$ determines the width of the tuning curve. The neuron responds most vigorously when a stimulus having $s = s_{\max}$ is presented, so we call $s_{\max}$ the preferred orientation angle of the neuron.

Response tuning curves can be used to characterize the selectivities of neurons in visual and other sensory areas to a variety of stimulus parameters. Tuning curves can also be measured for neurons in motor areas, in which case the average firing rate is expressed as a function of one or more parameters describing a motor action. Figure 1.6A shows an example of extracellular recordings from a neuron in primary motor cortex in a monkey that has been trained to reach in different directions. The stacked traces for each direction are rasters showing the results of five different trials. The horizontal axis in these traces represents time, and each mark indicates an action potential. The firing pattern of the cell, in particular the rate at which spikes are generated, is correlated with the direction of arm movement, and thus encodes information about this aspect of the motor action.

Figure 1.6B shows the response tuning curve of an M1 neuron plotted as a function of the direction of arm movement. Here the data points have been fitted by a tuning curve of the form

 $$ f(s)=r_{0}+\left(r_{\max}-r_{0}\right)\cos(s-s_{\max}), $$ 

where $s$ is the reaching angle of the arm, $s_{\mathrm{max}}$ is the reaching angle associ-

primary motor
cortex M1

cosine
tuning curve


