
#### 前课回顾——贝叶斯分类器

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//9cfff1dc-ced4-4595-a314-678f647d8f55/markdown_1/imgs/img_in_image_box_260_203_1079_631.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A00Z%2F-1%2F%2Ff462f462275c5454f097c58e2792c35a74e7a2bb2cf94e85992b4e7669a4c2d7" alt="Image" width="56%" /></div>


<div style="text-align: center;"><div style="text-align: center;">图 2.3 错误率</div> </div>


□ Bayes最小错误率决策使得每个观测值下的条件错误率最小，因而保证了（平均）错误率最小。

□ Bayes决策是一致最优决策

#### 前课回顾——Fisher 判别

□ Fisher准则的出发点：把所有样本都投影到一维空间（直线），要找到一个最合适的投影轴，使两类样本在该轴上投影之间的距离尽可能远，而每一类样本的投影尽可能紧凑，从而使分类效果为最佳

□ d维到一维的变换（线性判别函数）

 $$ \boldsymbol{y}_{n}=\mathbf{w}^{T}\mathbf{x}_{n},n=1,2,\mathrm{L},N_{i} $$ 

□ Fisher准则的描述：用投影后数据的统计性质——均值和离散度的函数作为判别优劣的标准。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//9cfff1dc-ced4-4595-a314-678f647d8f55/markdown_2/imgs/img_in_image_box_849_726_1194_957.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A00Z%2F-1%2F%2F9b99b2041c9abb5e3473a375d350fd561f5ecc4414c50ffac6e4945aa1720cb0" alt="Image" width="23%" /></div>


#### 前课回顾——分段线性分类器

推广→分段线性距离分类器：将各类别划分成相对密集的子类，每个子类以其均值作为代表点，然后按最小距离分类

判别函数定义： $ \omega_{i} $有 $ l_{i} $个子类，即属于 $ \omega_{i} $的决策域 $ R_{i} $分成 $ l_{i} $个子域 $ (R_{i}^{1}, R_{i}^{2}, \ldots, R_{i}^{li}) $，每个子区域用均值 $ m_{i}^{k} $作为代表点

 $$ g_{i}(\mathbf{x})=\min_{k=1,\ldots,l_{i}}\left\|\mathbf{x}-\mathbf{m}_{i}^{k}\right\| $$ 

判别规则

 $$ \underset{(i=1,...,c)}{j=\operatorname*{a r g m i n}g_{i}(\mathbf{x})} $$ 

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//9cfff1dc-ced4-4595-a314-678f647d8f55/markdown_3/imgs/img_in_image_box_744_343_1379_884.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A00Z%2F-1%2F%2F193e729e803925573ffca603bf6008adf6b67c721185191f7300c10a556691fa" alt="Image" width="44%" /></div>


<div style="text-align: center;"><div style="text-align: center;">I: 线性距离判别</div> </div>


<div style="text-align: center;"><div style="text-align: center;">Ⅱ：分段线性距离判别</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//9cfff1dc-ced4-4595-a314-678f647d8f55/markdown_3/imgs/img_in_image_box_1307_958_1427_1079.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A00Z%2F-1%2F%2Ff3448c80cca0a752c4046bb34dd468faf0d785dc79441a8f6df86c96e27a084c" alt="Image" width="8%" /></div>


#### 前课回顾——感知器

Neuron Model: 多输入，单输出，带偏置

1.  $ R $个输入 $ p_i \in \mathbb{R} $，即

 $ R $维输入矢量 $ \mathbf{p} $

2. n: net input, n=wp+b.

✓ R个权值 $ w_i \in \mathbb{R} $，即

 $ R $维权矢量 $ \mathbf{w} $

✓ 阈值b

3. 输出 $ a=f(n) $， $ f $:传递函数

神经元模型

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//8d3dd093-8a35-4634-a9e1-4de4538631b3/markdown_0/imgs/img_in_image_box_801_261_1317_709.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A17%3A59Z%2F-1%2F%2Facb13f60b2c639113809ba460dd62923122a9e5be788a6d5b33155495e345b03" alt="Image" width="35%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//8d3dd093-8a35-4634-a9e1-4de4538631b3/markdown_0/imgs/img_in_image_box_798_704_1288_968.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A00Z%2F-1%2F%2Fbfc8463e11d268c51a16cfe50e5c2485c307b60de22853b334106986ba6be4bf" alt="Image" width="34%" /></div>


McCulloch-Pitts模型

#### 前课回顾——神经网络

前馈型网络：节点按照一定的层次排列，信号按单一方向从一层节点传递到下一层节点，网络是单向的。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//8d3dd093-8a35-4634-a9e1-4de4538631b3/markdown_1/imgs/img_in_image_box_232_362_1217_927.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A00Z%2F-1%2F%2Fe9c93fa5d47002ec782983a8bf4a5055710c9f706d819e9ff1fc7def38033c65" alt="Image" width="68%" /></div>


输入层

 $$ \mathbf{a}^{3}=\mathbf{f}^{3}\mathbf{\Omega}\mathbf{L}\mathbf{W}^{3,2}\mathbf{f}^{2}\mathbf{\Omega}\mathbf{L}\mathbf{W}^{2,1}\mathbf{f}^{1}\mathbf{\Omega}\mathbf{I}\mathbf{W}^{1,1}\mathbf{p}+\mathbf{b}^{1})+\mathbf{b}^{2})+\mathbf{b}^{3}) $$ 

#### 前课回顾——支持向量机

在精度要求不高的情况下，我们可以采用上述核函数来作为支持向量机的核函数，但是，在某些特殊的情况下，我们为了提高支持向量机的分类精度，我们可以采用将更加高端的核函数与向量机结合来进行数学分类，比如小波核函数。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//8d3dd093-8a35-4634-a9e1-4de4538631b3/markdown_2/imgs/img_in_image_box_226_549_1210_949.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A01Z%2F-1%2F%2F20191971b37f727c504f63dc06ab2ce0b51aa34c1448452a3ba348db588a3da1" alt="Image" width="68%" /></div>


#### 前课回顾——最近邻法

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//8d3dd093-8a35-4634-a9e1-4de4538631b3/markdown_3/imgs/img_in_image_box_417_196_1028_758.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A02Z%2F-1%2F%2Fece7dc047ba8f11acfb8d929da60d7ad5844899ed7f3b900a4ccd0ec5f6774e1" alt="Image" width="42%" /></div>


在二维情况下，最近邻规则算法使得二维空间被分割成了许多Voronoi网格，每一个网格代表的类别就是它所包含的训练样本点所属的类别。

#### 前课回顾——K近邻法

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//a4eff1a8-fc8f-4160-a1b7-2d0165887b50/markdown_0/imgs/img_in_image_box_464_216_975_711.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A00Z%2F-1%2F%2F0cd4cf45033d0ce993c9fed70051ad06765c0577df9491d13b6191ce4fd9ea72" alt="Image" width="35%" /></div>


从样本点x开始生长，不断扩大区域，直到包含进k个训练样本点为止，并且把测试样本点x的类别归为这最近的k个训练样本点中出现频率最大的类别。

#### 前课回顾——决策树

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//a4eff1a8-fc8f-4160-a1b7-2d0165887b50/markdown_1/imgs/img_in_image_box_67_186_868_992.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A01Z%2F-1%2F%2F0d9f75ef824a0834a36f955c486b99d5cfb84f69e92f856b301755d189b9c320" alt="Image" width="55%" /></div>


女孩决定是否见一个约会对象的策略，其中绿色节点表示判断条件，橙色节点表示决策结果，箭头表示在一个判断条件在不同情况下的决策路径，图中红色箭头表示了上面例子中女孩的决策过程。

#### 前课回顾——随机森林

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//a4eff1a8-fc8f-4160-a1b7-2d0165887b50/markdown_2/imgs/img_in_image_box_90_160_1328_995.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A03Z%2F-1%2F%2F8792aadba746a2c0a76bbfbc90cc6e1bd453bfac52a8fbacf55a1a06f3aa3fac" alt="Image" width="85%" /></div>


#### 第7章 特征选择


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e627d79c-2f04-43a1-a45e-0c5652067fac/markdown_0/imgs/img_in_image_box_125_206_1316_553.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A17%3A59Z%2F-1%2F%2Fb18fb384f89af41f66b407006e9963799409a50c5151f104ab1c393ecb961473" alt="Image" width="82%" /></div>



##### □ 特征形成（特征获取、特征提取）

信号获取或测量→原始测量

原始特征

实例：

数字图象中的各像素灰度值

人体的各种生理指标

原始特征分析：

原始测量不能反映对象本质

高维原始特征不利于分类器设计：计算量大，冗余，样本分布十分稀疏

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//8dd299e0-bda6-4a3f-985b-bfdb2e566bf8/markdown_1/imgs/img_in_image_box_866_336_1429_673.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A03Z%2F-1%2F%2F2906f9b66e01d6234b427c1b7eac9799e2e5fd75c7a33974e56fa48c307acd23" alt="Image" width="39%" /></div>


特征空间过大导致计算量大，推广性差

提取有效信息、压缩特征空间的方法：

特征选择(selection)：从原始特征中挑选出一些最有代表性，分类性能最好的特征

原始特征空间： $ Y = \{y_1, y_2, \mathbf{L}, y_D\} $

精简特征空间： $ X = \{x_1, x_2, \ldots, x_d\} $

其中  $ x_{i}=y_{j} $

□ 特征提取 (extraction): 用映射(或变换)的方法把原始特征变换为较少的新特征，也称为特征变换、压缩

 $$  A:Y \to X $$ 

其中每个分量 $ x_{i} $是原特征向量各分量的函数，即

 $$ x_{i}=f_{i}(y_{1},y_{2},\mathrm{L},y_{D}) $$ 

## 7.2 特征的评价准则

##### 类别可分离性判据，类别可分性准则 $ _{ij} $

##### 希望满足的条件：

与错误率有单调关系，能较好满足反映分类目标

当特征独立时有可加性： $ J_{ij}(x_1,x_2,\ldots,x_d)=\sum_{k=1}^{d}J_{ij}(x_k) $

（数值越大，两类的分离程度越大）

度量特性： $ J_{ij} > 0 $, if  $ i = j $;  $ J_{ij} = 0 $, if  $ i \neq j $;  $ J_{ij} = J_{ji} $

▶ 判据对特征具有单调性：加入新的特征不会使判据减小

 $$ J_{ij}(x_{1},x_{2},...,x_{d})\leq J_{ij}(x_{1},x_{2},...,x_{d},x_{d+1}) $$ 

在实际应用并不一定能同时具备，但不影响它的实际使用价值

常见类别可分离性判据：基于距离、概率分布、熵函数、统计检验

##### 基于距离的可分性度量

### （一）点与点的距离

 $$ d(\vec{a},\vec{b})=\left[(\vec{a}-\vec{b})^{\mathrm{T}}(\vec{a}-\vec{b})\right]^{1/2}=\left[\sum_{k=1}^{n}(a_{k}-b_{k})^{2}\right]^{1/2} $$ 

### （二）点到点集的距离

用均方欧氏距离表示

点X到点集中所有点的距离平方和求平均

 $$ \overline{d}^{2}(\vec{x},\{\vec{a}_{k}^{(i)}\})=\frac{1}{N_{i}}\sum_{k=1}^{N_{i}}d^{2}(\vec{x},\vec{a}_{k}^{(i)}) $$ 

i 表示点集的序号，一共有  $ N_{i} $ 个点

# 基于距离的可分性度量

### （三）类内及总体的均值矢量

(三) 类内及总体的均值矢量

类的均值矢量： $\vec{m}^{(i)} = \frac{1}{N_i} \sum_{k=1}^{N_i} \vec{x}_k^{(i)}$  $i = 1,2,\cdots,c$

各类模式的总体均值矢量  $ \vec{m} = \sum_{i=1}^{c} P_i \vec{m}^{(i)} $

中心(重心)

某一类的样本集的中心(重心)

全体样本的中心(重心)

 $ P_i $ 为相应类的先验概率，当用统计量代替先验概率时，总体均值矢量可表示为：

 $$ \vec{m}=\sum_{i=1}^{c}P_{i}\vec{m}^{(i)}=\sum_{i=1}^{c}\frac{N_{i}}{N}\vec{m}^{(i)}=\frac{1}{N}\sum_{i=1}^{c}\sum_{k=1}^{N_{i}}\vec{x}_{k}^{(i)}=\frac{1}{N}\sum_{l=1}^{N}\vec{x}_{l} $$ 

### （四）类内距离

类内均方欧氏距离

一类样本每一点到

类中心的距离的平方和求平均

（五）类内离差矩阵

 $$ S_{\omega_{i}}=\frac{1}{N_{i}}\sum_{k=1}^{N_{i}}(\vec{x}_{k}^{(i)}-\vec{m}^{(i)})(\vec{x}_{k}^{(i)}-\vec{m}^{(i)})^{\mathrm{T}} $$ 

显然  $ \overline{d}^{2}(\omega_{i}) = Tr[S_{\omega_{i}}] $ 其中“ $ \mathbf{T}\mathbf{r}’ $”表示矩阵的迹（对角线元素的和）

##### 基于距离的可分性度量

### （六）两类之间的距离

 $$ \overline{d}(\omega_{i},\omega_{j})=\frac{1}{N_{i}N_{j}}\sum_{k=1}^{N_{i}}\sum_{l=1}^{N_{j}}d(\vec{x}_{k}^{(i)},\vec{x}_{l}^{(j)}) $$ 

两类样本之间任意连线的平均值

 $$ \overline{d}^{2}(\boldsymbol{\omega}_{i},\boldsymbol{\omega}_{j})=\frac{1}{N_{i}N_{j}}\sum_{k=1}^{N_{i}}\sum_{l=1}^{N_{j}}(\overrightarrow{x}_{k}^{(i)}-\overrightarrow{x}_{l}^{(j)})^{\mathrm{T}}(\overrightarrow{x}_{k}^{(i)}-\overrightarrow{x}_{l}^{(j)}) $$ 

两类样本之间任意

连线的平均平方和

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//e38a0f38-885f-4110-8328-44d2a4bc318f/markdown_3/imgs/img_in_image_box_562_544_653_646.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A03Z%2F-1%2F%2Fddc44888ed7fc6dc0a8a32605817e7b6f0c3275c1e65c4ed67a12e7822d32846" alt="Image" width="6%" /></div>


类内均方欧氏距离 $ \overline{d}^{2}(\omega_{i})=\frac{1}{N_{i}}\sum_{k=1}^{N_{i}}(\overrightarrow{x}_{k}^{(i)}-\overrightarrow{m}^{(i)})^{\mathrm{T}}(\overrightarrow{x}_{k}^{(i)}-\overrightarrow{m}^{(i)}) $

##### 基于距离的可分性度量

### （六）两类之间的距离

 $$ \overline{d}(\omega_{i},\omega_{j})=\frac{1}{N_{i}N_{j}}\sum_{k=1}^{N_{i}}\sum_{l=1}^{N_{j}}d(\vec{x}_{k}^{(i)},\vec{x}_{l}^{(j)}) $$ 

两类样本之间任意连线的平均值

 $$ \overline{d}^{2}(\omega_{i},\omega_{j})=\frac{1}{N_{i}N_{j}}\sum_{k=1}^{N_{i}}\sum_{l=1}^{N_{j}}(\vec{x}_{k}^{(i)}-\vec{x}_{l}^{(j)})^{\mathrm{T}}(\vec{x}_{k}^{(i)}-\vec{x}_{l}^{(j)}) $$ 

两类样本之间任意

连线的平均平方和

（七）各类模式之间的总的均方距离

 $$ \overline{d}^{2}(\overrightarrow{x})=\frac{1}{2}\sum_{i=1}^{c}P_{i}\sum_{j=1}^{c}P_{j}\frac{1}{N_{i}N_{j}}\sum_{k=1}^{N_{i}}\sum_{l=1}^{N_{j}}d^{2}(\overrightarrow{x}_{k}^{(i)},\overrightarrow{x}_{l}^{(j)}) $$ 

多类之间，两两求类间距离的加权和

##### 基于距离的可分性度量

### （六）两类之间的距离

两类样本之间任意连线的平均值

 $$ \overline{d}(\omega_{i},\omega_{j})=\frac{1}{N_{i}N_{j}}\sum_{k=1}^{N_{i}}\sum_{l=1}^{N_{j}}d(\vec{x}_{k}^{(i)},\vec{x}_{l}^{(j)}) $$ 

 $$ \overline{d}^{2}(\boldsymbol{\omega}_{i},\boldsymbol{\omega}_{j})=\frac{1}{N_{i}N_{j}}\sum_{k=1}^{N_{i}}\sum_{l=1}^{N_{j}}(\overrightarrow{x}_{k}^{(i)}-\overrightarrow{x}_{l}^{(j)})^{\mathrm{T}}(\overrightarrow{x}_{k}^{(i)}-\overrightarrow{x}_{l}^{(j)}) $$ 

（七）各类模式之间的总的均方距离

 $$ \overline{d}^{2}(\vec{x})=\frac{1}{2}\sum_{i=1}^{c}P_{i}\sum_{j=1}^{c}P_{j}\frac{1}{N_{i}N_{j}}\sum_{k=1}^{N_{i}}\sum_{l=1}^{N_{j}}d^{2}(\vec{x}_{k}^{(i)},\vec{x}_{l}^{(j)})^{ 多类之间 , 两两求  类间距离的加权和 } $$ 

当取欧氏距离时，总的均方距离为

 $$ \overline{d}^{2}(\overrightarrow{x})=\frac{1}{2}\sum_{i=1}^{c}P_{i}\sum_{j=1}^{c}P_{j}\frac{1}{N_{i}N_{j}}\sum_{k=1}^{N_{i}}\sum_{l=1}^{N_{j}}(\overrightarrow{x}_{k}^{(i)}-\overrightarrow{x}_{l}^{(j)})^{\mathrm{T}}(\overrightarrow{x}_{k}^{(i)}-\overrightarrow{x}_{l}^{(j)}) $$ 

##### 基于距离的可分性度量

（八） 多类情况下总的类内、类间及总体离差矩阵总的类内离差矩阵

 $$ S_{W}=\sum_{i=1}^{c}P_{i}S_{\omega_{i}}=\sum_{i=1}^{c}P_{i}\frac{1}{N_{i}}\sum_{k=1}^{N_{i}}(\vec{x}_{k}^{(i)}-\vec{m}^{(i)})(\vec{x}_{k}^{(i)}-\vec{m}^{(i)})^{\mathrm{T}} $$ 

类内离

差矩阵

 $ S_{\omega_i} = \frac{1}{N_i} \sum_{k=1}^{N_i} (\vec{x}_k^{(i)} - \vec{m}^{(i)}) (\vec{x}_k^{(i)} - \vec{m}^{(i)})^{\mathrm{T}} $

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//0bc57506-6d4b-4563-8b3d-13d6d82fd92f/markdown_2/imgs/img_in_image_box_1128_493_1297_666.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A02Z%2F-1%2F%2Fa536e84e04f9000dfa878669ef2b77a199ffb18511a666292b55734662e25986" alt="Image" width="11%" /></div>


总的类间离差矩阵

 $$ S_{B}=\sum_{i=1}^{c}P_{i}(\overrightarrow{m}^{(i)}-\overrightarrow{m})(\overrightarrow{m}^{(i)}-\overrightarrow{m})^{\mathrm{T}} $$ 

(八) 多类情况下总的类内、类间及总体离差矩阵（续

总体离差矩阵  $  S_{T} = \frac{1}{N} \sum_{l=1}^{N} (\vec{x}_{l} - \vec{m}) (\vec{x}_{l} - \vec{m})^{\mathrm{T}} = S_{W} + S_{B}  $ 不分类的样本协方差

 $$ \begin{array}{r l}&{S_{W}=\displaystyle\sum_{i=1}^{c}P_{i}\frac{1}{N_{i}}\displaystyle\sum_{k=1}^{N_{i}}(\vec{x}_{k}^{(i)}-\vec{m}^{(i)})(\vec{x}_{k}^{(i)}-\vec{m}^{(i)})^{\mathrm{T}}}\\ &{S_{B}=\displaystyle\sum_{i=1}^{c}P_{i}(\vec{m}^{(i)}-\vec{m})(\vec{m}^{(i)}-\vec{m})^{\mathrm{T}}}\end{array} $$ 

 $$ \begin{align*}S_{W}+S_{B}=&\sum_{i=1}^{c}P_{i}\frac{1}{N_{i}}\sum_{k=1}^{N_{i}}(\vec{x}_{k}^{(i)}-\vec{m}^{(i)})(\vec{x}_{k}^{(i)}-\vec{m}^{(i)})^{\mathrm{T}}+\sum_{i=1}^{c}P_{i}(\vec{m}^{(i)}-\vec{m})(\vec{m}^{(i)}-\vec{m})\\=&\sum_{i=1}^{c}P_{i}\Biggl[\frac{1}{N_{i}}\sum_{k=1}^{N_{i}}(\vec{x}_{k}^{(i)}-\vec{m}^{(i)})(\vec{x}_{k}^{(i)}-\vec{m}^{(i)})^{\mathrm{T}}+(\vec{m}^{(i)}-\vec{m})(\vec{m}^{(i)}-\vec{m})^{\mathrm{T}}\Biggr]\end{align*} $$ 

 $$ (\boldsymbol{A}-\boldsymbol{B})(\boldsymbol{A}-\boldsymbol{B})^{\mathrm{T}}=(\boldsymbol{A}-\boldsymbol{B})(\boldsymbol{A}^{\mathrm{T}}-\boldsymbol{B}^{\mathrm{T}})=\boldsymbol{A}\boldsymbol{A}^{\mathrm{T}}-\boldsymbol{A}\boldsymbol{B}^{\mathrm{T}}-\boldsymbol{B}\boldsymbol{A}^{\mathrm{T}}+\boldsymbol{B}\boldsymbol{B}^{\mathrm{T}} $$ 

 $$ \begin{aligned}S_{W}+S_{B}=&\sum_{i=1}^{c}P_{i}\left\lfloor\frac{1}{N_{i}}\sum_{k=1}^{N_{i}}\Big(\overrightarrow{x}_{k}^{(i)}\overrightarrow{x}_{k}^{(i)\mathrm{T}}-\overrightarrow{x}_{k}^{(i)}\overrightarrow{m}^{(i)\mathrm{T}}-\overrightarrow{m}^{(i)}\overrightarrow{x}_{k}^{(i)\mathrm{T}}+\overrightarrow{m}^{(i)}\overrightarrow{m}^{(i)\mathrm{T}}\Big)\right\rfloor\\+&\sum_{i=1}^{c}P_{i}\Big[\overrightarrow{m}^{(i)}\overrightarrow{m}^{(i)\mathrm{T}}-\overrightarrow{m}^{(i)}\overrightarrow{m}^{\mathrm{T}}-\overrightarrow{m}\overrightarrow{m}^{(i)\mathrm{T}}+\overrightarrow{m}\overrightarrow{m}^{\mathrm{T}}\Big]\end{aligned} $$ 

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//0bc57506-6d4b-4563-8b3d-13d6d82fd92f/markdown_3/imgs/img_in_image_box_1306_956_1431_1079.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A02Z%2F-1%2F%2Fb770b2132dcec51acecdd34340ec863f96c56995fe1dbe81beb2bd0322c439a3" alt="Image" width="8%" /></div>


(八) 多类情况下总的类内、类间及总体离差矩阵（续

总体离差矩阵  $  S_{T} = \frac{1}{N} \sum_{l=1}^{N} (\vec{x}_{l} - \vec{m}) (\vec{x}_{l} - \vec{m})^{\mathrm{T}} = S_{W} + S_{B}  $

 $$ \begin{aligned}S_{W}+S_{B}&=\sum_{i=1}^{c}P_{i}\left[\frac{1}{N_{i}}\sum_{k=1}^{N_{i}}\left(\overrightarrow{x}_{k}^{(i)}\overrightarrow{x}_{k}^{(i)\mathrm{T}}-\overrightarrow{x}_{k}^{(i)}\overrightarrow{m}^{(i)\mathrm{T}}-\overrightarrow{m}^{(i)}\overrightarrow{x}_{k}^{(i)\mathrm{T}}+\overrightarrow{m}^{(i)}\overrightarrow{m}^{(i)\mathrm{T}}\right)\right]\\&+\sum_{i=1}^{c}P_{i}\left[\overrightarrow{m}^{(i)}\overrightarrow{m}^{(i)\mathrm{T}}-\overrightarrow{m}^{(i)}\overrightarrow{m}^{\mathrm{T}}-\overrightarrow{m}\overrightarrow{m}^{(i)\mathrm{T}}+\overrightarrow{m}\overrightarrow{m}^{\mathrm{T}}\right]\end{aligned} $$ 

 $$ \frac{1}{N_{i}}\sum_{k=1}^{N_{i}}\vec{x}_{k}^{(i)}\vec{m}^{(i)\mathrm{T}}=\left(\frac{1}{N_{i}}\sum_{k=1}^{N_{i}}\vec{x}_{k}^{(i)}\right)\vec{m}^{(i)\mathrm{T}}=\vec{m}^{(i)}\vec{m}^{(i)\mathrm{T}},\frac{1}{N_{i}}\sum_{k=1}^{N_{i}}\vec{m}^{(i)}\vec{x}_{k}^{(i)\mathrm{T}}=\vec{m}^{(i)}\left(\frac{1}{N_{i}}\sum_{k=1}^{N_{i}}\vec{x}_{k}^{(i)}\right)^{\mathrm{T}}=\vec{m}^{(i)}\vec{m}^{(i)\mathrm{T}} $$ 

 $$ \frac{1}{N_{i}}\sum_{k=1}^{N_{i}}\overrightarrow{m}^{(i)}\overrightarrow{m}^{(i)\mathrm{T}}=\overrightarrow{m}^{(i)}\overrightarrow{m}^{(i)\mathrm{T}} $$ 

 $$ \begin{align*}S_{W}+S_{B}&=\sum_{i=1}^{c}P_{i}\left[\frac{1}{N_{i}}\sum_{k=1}^{N_{i}}\left(\overrightarrow{x}_{k}^{(i)}\overrightarrow{x}_{k}^{(i)\mathrm{T}}-\overrightarrow{m}^{(i)}\overrightarrow{m}^{\mathrm{T}}-\overrightarrow{m}\overrightarrow{m}^{(i)\mathrm{T}}+\overrightarrow{m}\overrightarrow{m}^{\mathrm{T}}\right)\right]\\&=\sum_{i=1}^{c}P_{i}\left[\frac{1}{N_{i}}\sum_{k=1}^{N_{i}}\left(\overrightarrow{x}_{k}^{(i)}\overrightarrow{x}_{k}^{(i)\mathrm{T}}-\overrightarrow{x}_{k}^{(i)}\overrightarrow{m}^{\mathrm{T}}-\overrightarrow{m}\overrightarrow{x}_{k}^{(i)\mathrm{T}}+\overrightarrow{m}\overrightarrow{m}^{\mathrm{T}}\right)\right]\end{align*} $$ 

# 基于距离的可分性度量

### （八） 多类情况下总的类内、类间及总体离差矩阵（续）

总体离差矩阵  $  S_{T} = \frac{1}{N} \sum_{l=1}^{N} (\vec{x}_{l} - \vec{m}) (\vec{x}_{l} - \vec{m})^{\mathrm{T}} = S_{W} + S_{B}  $

 $$ \begin{aligned}\boldsymbol{S}_{W}+\boldsymbol{S}_{B}=&\sum_{i=1}^{c}\boldsymbol{P}_{i}\left[\frac{1}{N_{i}}\sum_{k=1}^{N_{i}}\left(\overrightarrow{x}_{k}^{(i)}\overrightarrow{x}_{k}^{(i)\mathrm{T}}-\overrightarrow{m}^{(i)}\overrightarrow{m}^{\mathrm{T}}-\overrightarrow{m}\overrightarrow{m}^{(i)\mathrm{T}}+\overrightarrow{m}\overrightarrow{m}^{\mathrm{T}}\right)\right]\\=&\sum_{i=1}^{c}\boldsymbol{P}_{i}\left[\frac{1}{N_{i}}\sum_{k=1}^{N_{i}}\left(\overrightarrow{x}_{k}^{(i)}\overrightarrow{x}_{k}^{(i)\mathrm{T}}-\overrightarrow{x}_{k}^{(i)}\overrightarrow{m}^{\mathrm{T}}-\overrightarrow{m}\overrightarrow{x}_{k}^{(i)\mathrm{T}}+\overrightarrow{m}\overrightarrow{m}^{\mathrm{T}}\right)\right]\end{aligned} $$ 

 $$ (\boldsymbol{A}-\boldsymbol{B})(\boldsymbol{A}-\boldsymbol{B})^{\mathrm{T}}=(\boldsymbol{A}-\boldsymbol{B})(\boldsymbol{A}^{\mathrm{T}}-\boldsymbol{B}^{\mathrm{T}})=\boldsymbol{A}\boldsymbol{A}^{\mathrm{T}}-\boldsymbol{A}\boldsymbol{B}^{\mathrm{T}}-\boldsymbol{B}\boldsymbol{A}^{\mathrm{T}}+\boldsymbol{B}\boldsymbol{B}^{\mathrm{T}} $$ 

 $$ S_{W}+S_{B}=\sum_{i=1}^{c}P_{i}\left[\frac{1}{N_{i}}\sum_{k=1}^{N_{i}}(\vec{x}_{k}^{(i)}-\vec{m})(\vec{x}_{k}^{(i)}-\vec{m})^{\mathrm{T}}\right]=\frac{1}{N}\sum_{l=1}^{N}(\vec{x}_{l}-\vec{m})(\vec{x}_{l}-\vec{m})^{\mathrm{T}}=S_{T} $$ 

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//3ff71d66-912f-4902-a77c-a67ff46c97f0/markdown_1/imgs/img_in_seal_box_1310_956_1428_1079.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A02Z%2F-1%2F%2Fa049be6e90c16fcd6f23c77e53b877262acb85d1cf951a06763007852f0b9d37" alt="Image" width="8%" />  <div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//3ff71d66-912f-4902-a77c-a67ff46c97f0/markdown_1/imgs/img_in_seal_box_1310_956_1428_1079.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A02Z%2F-1%2F%2Fa049be6e90c16fcd6f23c77e53b877262acb85d1cf951a06763007852f0b9d37" alt="Image" width="8%" />  生命信息与仪器工程学院 L.I.S.I.E.  </div>   </div>


# 基于距离的可分性度量

可以用  $ S_{W} $、 $ S_{B} $、 $ S_{T} $ 构造不同的可分性判据：

类内类间

 $$ \boldsymbol{J}_{1}=T r\begin{bmatrix}\dot{\boldsymbol{S}}_{W}^{-1}\boldsymbol{S}_{B}\end{bmatrix} $$ 

类间离散度尽量大

类内离散度尽量小

 $ J_{3}=\frac{Tr[S_{B}]}{Tr[S_{W}]} $ └类间

 $$ J_{2}=\ln\left[\frac{\left|S_{B}\right|}{\left|S_{W}\right|}\right]\xleftarrow{ 郴类间 } 郴类内 $$ 

 $$ \overline{J_{4}}=\frac{\left|S_{W}+S_{B}\right|}{\left|S_{W}\right|}=\frac{\left|S_{T}\right|\leftrightarrow 总体 }{\left|S_{W}\right|\leftrightarrow 类内 } $$ 

可以证明  $ J_{1} $、 $ J_{2} $ 和  $ J_{4} $ 在任何非奇异线性变换下是不变的， $ J_{3} $ 与坐标系有关。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//3ff71d66-912f-4902-a77c-a67ff46c97f0/markdown_2/imgs/img_in_image_box_1307_956_1429_1079.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A04Z%2F-1%2F%2Fbb0052bb73d3300ff57bfb2654301b37864b5442f3296f617311748c63496411" alt="Image" width="8%" /></div>


##### 马氏距离

### Mahalanobis distances are calculated as:

 $$ \begin{array}{r}{\mathcal{O}^{2}=\left(\mathbf{x}-\mathbf{m}\right)^{\mathrm{T}}\mathbf{C}^{-1}\left(\mathbf{x}-\mathbf{m}\right)}\end{array} $$ 

where:

 $ D^{2} = Mahalanobis\ distance $

x = Vector of data

m = Vector of mean values of independent variables

 $ C^{-1} = \text{Inverse Covariance matrix of independent variables} $

T = Indicates vector should be transposed

##### 马氏距离

If we calculate Mahalanobis distances for each of these points and shade them according to their distance value, we see clear elliptical patterns emerge:

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//59fef1a3-f59c-4bc3-84f8-d8f91e0892e1/markdown_0/imgs/img_in_chart_box_461_446_979_959.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A00Z%2F-1%2F%2Ff5062628daad818c98031a1368419e93b34f61ea70d46a8f96c1c6e9f963c666" alt="Image" width="35%" /></div>


### 7.2.2 基于概率分布的可分性判据

☐ 基于距离的判据没有考虑样本的分布情况，很难与错误率建立起联系

用概率密度函数间的距离来度量

□ 考虑样本在特征空间的分布，用两类概密函数的重叠程度来度量可分性，距离越大越可分。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//59fef1a3-f59c-4bc3-84f8-d8f91e0892e1/markdown_1/imgs/img_in_image_box_168_532_670_945.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A01Z%2F-1%2F%2F7b9aa70e19a4109fe50913464e70d37cf7a8fd1602dfdd1c8a1aadc642fbb9c8" alt="Image" width="34%" /></div>


<div style="text-align: center;"><div style="text-align: center;">(a)</div> </div>


 $$ p(\vec{\pi}|\varpi_{1})=p(\vec{\pi}|\varpi_{2}) $$ 

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//59fef1a3-f59c-4bc3-84f8-d8f91e0892e1/markdown_1/imgs/img_in_image_box_869_592_1277_942.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A01Z%2F-1%2F%2F38031e6e31a9c788421b1632738d496e7502e98a86269a38571266f6a7751d1b" alt="Image" width="28%" /></div>


<div style="text-align: center;"><div style="text-align: center;">(B)</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//59fef1a3-f59c-4bc3-84f8-d8f91e0892e1/markdown_1/imgs/img_in_image_box_1307_955_1430_1078.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A01Z%2F-1%2F%2F615694a423048957fc7447ad3e73449fb4a0491f0a2577e6b6a3e5f2c7b7f219" alt="Image" width="8%" /></div>


##### 基于类的概率密度函数的可分性判据

### （一）Bhattacharyya 判据（ $ J_{B} $）

受相关概念与应用的启发，我们可以构造B-判据，它的计算式为

 $$ J_{_{B}}=-\ln\int\limits_{\Omega}\left[p(\overrightarrow{x}\big|\omega_{1})p(\overrightarrow{x}\big|\omega_{2})\right]^{1/2}d\overrightarrow{x} $$ 

满足对称性 $ J_{ij}=J_{ji} $

##### 基于类的概率密度函数的可分性判据

### （一）Bhattacharyya 判据（ $ J_{B} $）

受相关概念与应用的启发，我们可以构造B-判据，它的计算式为：

 $$ J_{_{B}}=-\ln\int\limits_{\Omega}\left[p(\overrightarrow{x}\big|\omega_{1})p(\overrightarrow{x}\big|\omega_{2})\right]^{1/2}d\overrightarrow{x} $$ 

满足对称性 $ J_{ij}=J_{ji} $

两种特殊情况：

 $$ p(\overrightarrow{x}|\omega_{1})=p(\overrightarrow{x}|\omega_{2})\Rightarrow J_{_{B}}=-{\ln1}=0 $$ 

 $ p(\vec{x}|\omega_1) $与 $ p(\vec{x}|\omega_2) $基本不重叠, $ p(\vec{x}|\omega_1)p(\vec{x}|\omega_2)\leq1,J_B=-\ln0_+\rightarrow\infty $

##### 基于类的概率密度函数的可分性判据

### （二）Chernoff 判据( $ J_{c} $)

我们可以构造比  $ J_{B} $ 更一般的判据，称其为 C-判据，其定义式为

 $$ \begin{aligned}J_{C}&=-\ln\int\limits_{\Omega}p(\vec{x}\big|\omega_{1})^{s}p(\vec{x}\big|\omega_{2})^{1-s}d\vec{x}=\boldsymbol{J}_{C}^{def}(s;x_{1},x_{2},\cdots,x_{n})\\&\overset{def}{=}\boldsymbol{J}_{C}(\omega_{1},\omega_{2};s)=\boldsymbol{J}_{C}(s)\quad&0<s<1\end{aligned} $$ 

##### 基于类的概率密度函数的可分性判据

 $ J_{C} $ 具有如下性质：

(1) 对于一切 0 < S < 1,  $ J_{c} \geq 0 $

(2) 对于一切  $ 0 < S < 1 $,  $ J_C = 0 \Leftrightarrow p(\vec{X} \mid \omega_1) = p(\vec{X} \mid \omega_2) $

(3) 当参数s和 $ (1-s) $互调时,有对称性,

 $$ \boldsymbol{J}_{C}\left(\omega_{1},\omega_{2};s\right)=\boldsymbol{J}_{C}\left(\omega_{2},\omega_{1};1-s\right) $$ 

(4) 当 $ \overrightarrow{X} $的各分量 $ x_1,x_2,\cdots,x_n $互相独立时

 $$ J_{c}\left(s;X\right)=\sum_{l=1}^{n}J_{c}\left(s;x_{l}\right) $$ 

##### 基于类的概率密度函数的可分性判据

### （三）JM 距离

可看作是B-判据的一种重要变体：

 $$ b(p,q)=-\ln\int\sqrt{p(x)q(x)}d x $$ 

 $$ J M_{p,q}=\sqrt{2\left(1-\sum_{i=1}^{N}\sqrt{p_{i}\cdot q_{i}}\right)} $$ 

JM距离范围：[0, $ \sqrt{2} $]，相比B判据，JM能够弱化极高和极低区分度的影响。

### 7.2.3 基于熵的可分性判据

□ 如果样本属于各类的后验概率越平均，则该特征越不利于分类；

□ 如果后验概率越集中于某一类，则特征越有利于分类

□ 为了衡量各类后验概率的集中程度，可借用信息论中熵的概念定义类别可分性的依据。

在信息论中，熵(Entropy)表示不确定性，熵越大不确定性越大；

□ 可以借用熵的概念来描述各类的可分性，衡量后验概率分布的集中程度

□ 熵函数

 $$ \boldsymbol{H}=\boldsymbol{J}_{c}\left[P(\omega_{1}\mid\mathbf{x}),...,\boldsymbol{P}(\omega_{c}\mid\mathbf{x})\right] $$ 

Shannon熵  $ J_{c}^{1} = -\sum_{i=1}^{c} P(\omega_{i} | \mathbf{x}) \log P(\omega_{i} | \mathbf{x}) $

另有Kolmogorov 熵，Topological 熵，Boltzmann 熵等

□ 熵函数期望表征类别的分离程度

 $$ J(\bullet)=E\left\{J_{c}\left[P(\omega_{1}\mid\mathbf{x}),...,\boldsymbol{P}(\omega_{c}\mid\mathbf{x})\right]\right\} $$ 

】越小，可分性越好

### 7.2.4 利用统计检验作为可分性判据

检验某一变量在两类样本间是否存在显著差异，可通过统计量来反映两类样本间的差别

□ 最常用的比较两组样本差别的方法是t-检验，须假设两类样本都服从正态分布，且方差相同

计算出实际t值后可推断在该特征上两类样本是否有显著差异。

但t检验属于参数化检验方法，对数据分布有一定的假设，必要时要验证样本分布是否属于该假设；

另一种非参数检验不对数据分布做特殊假设，适合更复杂的数据分布情况

□ 秩和检验没有对样本分布做任何假设，因此适用面更广；

其做法是-把两类样本混合，对所有样本按照所考查的特征从小到大排序，若出现特征取值相等的样本时则并列采用中间的顺序，两类样本中分别计算所得排序序号之和T1、T2，称为秩和。

□ 基本思想：如果一类样本的秩和显著地比另一类样本小（或大），则两类样本在所考查的特征上有显著差异。

##### 特征选择与分类器性能

##### 特征选择的方法：

是否直接考虑分类器性能

□ Filter方法：根据独立于分类器的指标J来评价所选择的特征子集S，在所有可能的特征子集中搜索出使得J最大的特征子集作为最优特征子集。不考虑所使用的学习算法

□ Wrapper方法：将特征选择和分类器结合在一起，在分类过程中表现优异（错误率）的特征子集会被选中

选择特征的顺序

自下而上：特征数从零逐步增加到d

自上而下：特征数从D开始逐步减少到d

##### 不同特征选择策略在作物病害监测上的应用

An severe outbreak of wheat rust happened in Henan Province, China in 2013

The feasibility of using high-resolution image (ZY-3) to map wheat rust

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//0e4c5b0d-b6b6-4dcf-8933-e20a0842552a/markdown_0/imgs/img_in_image_box_104_390_491_956.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A01Z%2F-1%2F%2F5c577d5ad6200076e4c74e306e0213dcba578231c8e21829e6fcbd74697e5a19" alt="Image" width="26%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//0e4c5b0d-b6b6-4dcf-8933-e20a0842552a/markdown_0/imgs/img_in_image_box_514_393_1335_1002.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A01Z%2F-1%2F%2Fd8be22a820091fa679a712d2971b44c9414cc7035bf4e24473e7434d17ed12e4" alt="Image" width="57%" /></div>


#### 不同特征选择策略在作物病害监测上的应用

Classifier accuracy:

filter feature selection 90%

wrapper feature selection with SVM 93%

wrapper feature selection with RF 92.5%

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//0e4c5b0d-b6b6-4dcf-8933-e20a0842552a/markdown_1/imgs/img_in_image_box_1309_956_1429_1078.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A03Z%2F-1%2F%2F7a8f323826335aa210abed1bdaa9aa18f0deda28cdc72aeb1509fb4cc0dbf8bf" alt="Image" width="8%" /></div>


##### 不同特征选择策略在作物病害监测上的应用

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//0e4c5b0d-b6b6-4dcf-8933-e20a0842552a/markdown_2/imgs/img_in_image_box_57_185_1032_993.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A04Z%2F-1%2F%2Ff1f04f15922a983c5d6105fb0ef9d6ea542540b7b90cae4bb46a9e10e8a7dc41" alt="Image" width="67%" /></div>


## 7.3 特征选择的最优算法

特征选择：从原始特征中挑选出一些最有代表性、分类性能最好的特征进行分类

单独最优特征组合

计算各特征单独使用时的可分性判据J并加以排队，取前d个作为选择结果

并非是最优结果

☐ 从D个特征中选取d个，共 $ C_{D}^{d} $种组合

- 典型的组合优化问题

 $$ \mathsf{C^{d}_{D}|_{D=20,d=10}=184756} $$ 



<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td rowspan="2">D=100</td><td style='text-align: center; word-wrap: break-word;'>d</td><td style='text-align: center; word-wrap: break-word;'>2</td><td style='text-align: center; word-wrap: break-word;'>3</td><td style='text-align: center; word-wrap: break-word;'>10</td><td style='text-align: center; word-wrap: break-word;'>50</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>$ C^{d}_{D} $</td><td style='text-align: center; word-wrap: break-word;'>4950</td><td style='text-align: center; word-wrap: break-word;'>161700</td><td style='text-align: center; word-wrap: break-word;'>1.731e+13</td><td style='text-align: center; word-wrap: break-word;'>1.0089e+29</td></tr></table>

许多特征选择算法力求解决搜索问题，经典算法有：

分支定界法：最优搜索，效率比盲目穷举法高

次优搜索

单独最优特征组合法：

顺序前进法

顺序后退法

其他组合优化方法

模拟退火法

遗传算法

### 7.3.1 最优搜索算法

□ 一种不需要进行穷举但仍能够取得最优解的方法是分枝定界法。该种方法具有回溯的过程，能够考虑到所有可能的组合。

☐ 基本思想：将所有可能特征组合成一个树状结构，按照特定规律对树进行搜索，使得搜索过程尽可能达到最优而不必遍历整个树。（准则判据对特征具有单调性）

##### 分枝定界法

寻求全局最优的特征选择的搜索过程可用一个树结构来描述，称其为搜索树或解树

总的搜索方案是沿着树自上而下、从右至左进行，由于树的每个节点代表一种特征组合，于是所有可能的组合都可以被考虑利用可分性判据的单调性采用分支定界策略和值左小右大的树结构，使得在实际上并不计算某些特征组合而又不影响全局寻优

□ 分支限界法的求解目标则是找出满足约束条件的一个解，或是在满足约束条件的解中找出使某一目标函数值达到极大或极小的解，即在某种意义下的最优解。

□ 分支限界法则以广度优先或以最小耗费优先的方式搜索解空间树。分支限界法的搜索策略是：在扩展结点处，首先生成其所有的儿子结点（分支），然后再从当前的结点表中选择下一个扩展结点。为了有效地选择下一扩展结点，加速搜索进程，在每一结点处，计算一个函数值（限界），并根据函数值，从当前结点表中选择一个最有利的结点作为扩展结点，使搜索朝着解空间树上有最优解的分支推进，以便尽快地找出一个最优解。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//9cee4028-6357-4e5c-b64e-c369471ead6d/markdown_3/imgs/img_in_image_box_114_292_783_758.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A05Z%2F-1%2F%2F133485f798c840afee1cf2ad8fb3cb7a3b1e6c5745f0895c65a759471d0bf0a7" alt="Image" width="46%" /></div>


<div style="text-align: center;"><div style="text-align: center;">(a)</div> </div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//9cee4028-6357-4e5c-b64e-c369471ead6d/markdown_3/imgs/img_in_image_box_812_296_1321_752.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A05Z%2F-1%2F%2Fe8ab330e7f3a2efc692a44ebe7ac5118383bb767d65d528515c28a54c3cf5fa8" alt="Image" width="35%" /></div>


<div style="text-align: center;"><div style="text-align: center;">(b)</div> </div>


<div style="text-align: center;"><div style="text-align: center;">(a) 搜索树</div> </div>


<div style="text-align: center;"><div style="text-align: center;">6选2的特征选择问题</div> </div>


<div style="text-align: center;"><div style="text-align: center;">(b) 搜索回溯示意图</div> </div>


□ 树的每个节点表示一种特征组合，树的每一级各节点表示从其父节点的特征组合中再去掉一个特征后的特征组合，其标号k表示去掉的特征是 $ x_{k} $

☐ 由于每一级只舍弃一个特征，因此整个搜索树除根节点的0级外，还需要n-d级，即全树有n-d级。在6个特征中选2个，故整个搜索树需要4级，第n-d级是叶节点，有 $ C_{n}^{d} $个叶节点

 $ \overline{X}_{s} $ 表示舍弃s个特征后余下的特征集合。

 $ \Psi_{s} $ 表示第s级当前节点上用来作为下一级可舍弃特征的集合。

 $ r_{s} $ 表示集合中元素的数目。

 $ q_{s} $ 表示当前节点的子节点数。

□ 由于从根节点要经历n-d级才能到达叶节点，s级某节点后继的每一个子节点分别舍弃 $ \psi_{s} $ 中互不相同的一个特征，从而考虑在 $ s+1 $级可以舍弃的特征 $ \overline{x} $ 方案数(即其子节点数) $ q_{s} $时，必须使这一级舍弃了特征 $ X_{s} $ 后还剩 $ (n-d)-(s+1) $个特征。除了从树的纵的方向上一级丢弃一个特征，实际上从树的横的方向上，一个分支也轮换丢弃一个特征。因此后继子节点数

 $$ q_{s}=r_{s}-(n-d-s-1) $$ 

##### ☐ 该算法的高效性能原因在于如下三个方面：

在构造搜索树时，同一父节点的各子节点为根的各子树右边的边要比左边的少，即树的结构右边比左边简单

在同一级中按最小的J值从左到右挑选舍弃的特征，即节点的J值是左小右大，而搜索过程是从右至左进行的

因J的单调性，树上某节点如A的可分性判据值 $ J_A \leq B $，则A的子树上各节点的J值都不会大于B，因此该子树各节点都可以不去搜索

从(1)、(2)和(3)可知，有很多的特征组合不需计算仍能求得全局最优解

## 7.4 次优搜索算法

□ 采用分枝定界法的计算量可能仍然很大，可采用计算量小的次优法

##### 单独最优特征组合

计算各特征单独使用时的可分性判据J并加以排队，取前d个作为选择结果

并非是最优结果

当可分性判据对各特征具有(广义)可加性，该方法可以选出一组最优的特征来，例如：

各类具有正态分布

各特征统计独立

可分性判据基于Mahalanobis距离

 $$ J_{ij}(x_{1},x_{2},...,x_{d})=\sum_{k=1}^{n}J_{ij}(x_{k}) $$ 

 $$ \boldsymbol{J}_{D}(\mathbf{x})=(\boldsymbol{\mu}_{i}-\boldsymbol{\mu}_{j})^{T}\boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_{i}-\boldsymbol{\mu}_{j}) $$ 

##### 顺序前进法(Sequential forward selection)

自下而上搜索方法

每次从未入选的特征中选择一个特征，使得它与已入选的特征组合在一起时所得的J值为最大，直至特征数增加到d为止

 $$ J(X_{k}+x_{1})\geq J(X_{k}+x_{2})\geq\cdots\geq J(X_{k}+x_{n-k}) $$ 

考虑了所选特征与已入选特征之间的相关性，但无剔除环节

顺序后退法(Sequential backward selection)

从全体特征开始，每次剔除一个特征，使得所保留的特征集合有最大的分类识别率

增 $ l $减 $ r $法

逐步选入l个，再逐步剔除r个

广义（Generalized）方法：GSFS、GSBS

### 模拟退火

#### Simulated Annealing

##### 模拟退火

##### 算法的提出

模拟退火算法最早的思想由Metropolis等（1953）提出，1983年Kirkpatrick等将其应用于组合优化。

##### 算法的目的

解决NP复杂性问题：

克服优化过程陷入局部极小；

克服初值依赖性。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//c1aa9c9e-7f46-4727-952a-12eef984c33a/markdown_2/imgs/img_in_image_box_944_525_1221_937.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A02Z%2F-1%2F%2F718a46747c94ec49a8f46a3ca12d379c89a5bc34966b7ed0aca6c48de2ef61a0" alt="Image" width="19%" /></div>


Nick Metropolis

##### NP(nondeterministic poly-nomial)问题

什么是非确定性问题？有些计算问题是确定性的，比如加减乘除之类，只要按照公式推导，就可以得到结果。但是，有些问题是无法按部就班直接地计算出来。比如，找大 $ \underline{\text{质数}} $的问题。这种问题的答案，是无法直接计算得到的，只能通过间接的“猜算”来得到结果。这也就是非确定性问题。而这些问题的通常有个算法，它不能直接告诉你答案是什么，但可以告诉你，某个可能的结果是正确的答案还是错误的。这个可以告诉你“猜算”的答案正确与否的算法，假如可以在 $ \underline{\text{多项式}} $（polynomial）时间内算出来，就叫做多项式非确定性问题。多项式非确定性问题可以用 $ \underline{\text{穷举法}} $得到答案，一个个检验下去，最终便能得到结果。但是这样算法的复杂程度，是指数关系，因此计算的时间随问题的复杂程度成指数的增长，很快便变得不可计算了。

##### 模拟退火

##### 什么是退火：

退火是指将固体加热到足够高的温度，使分子呈随机排列状态，然后逐步降温使之冷却，最后分子以低能状态排列，固体达到某种稳定状态。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//09d7efc4-4101-44f5-a137-cd0e443877d7/markdown_0/imgs/img_in_image_box_764_586_1318_975.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A00Z%2F-1%2F%2F9c86952e28e9637136203f116379dd8a95a5187bd6f9844b3d4c24a9d0473aa8" alt="Image" width="38%" /></div>


##### 模拟退火

退火工艺：退火是将金属和合金加热到适当温度，保持一定时间，然后缓慢冷却的热处理工艺。退火后组织亚共析钢是铁素体加片状珠光体；共析钢或过共析钢则是粒状珠光体。总之退火组织是接近平衡状态的组织。

退火的目的：①降低钢的硬度，提高塑性，以利于切削加工及冷变形加工。②细化晶粒，消除因铸、锻、焊引起的组织缺陷，均匀钢的组织和成分，改善钢的性能或为以后的热处理作组织准备。③消除钢中的内应力，以防止变形和开裂

##### 模拟退火

##### 加温过程——增强粒子的热运动，消除系统原先可能

存在的非均匀态；

等温过程——对于与环境换热而温度不变的封闭系统，系统状态的自发变化总是朝自由能减少的方向进行，当自由能达到最小时，系统达到平衡态；

冷却过程——使粒子热运动减弱并渐趋有序，系统能量逐渐下降，从而得到低能的晶体结构。

##### 模拟退火

##### 热力学中的退火指物体逐渐降温时发生的物理现象

温度越低，物体的能量状态越低，到达足够的低点时，液体开始冷凝与结晶，在结晶状态时，系统的能量状态最低。缓慢降温（退火）时，可达到最低能量状态；但如果快速降温（淬火），会导致不是最低能态的非晶形。

##### □ 大自然知道慢工出细活

缓缓降温，使得物体分子在每一温度时，能够有足够时间找到安顿位置，则逐渐地，到最后可得到最低能态，系统最稳定。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//09d7efc4-4101-44f5-a137-cd0e443877d7/markdown_3/imgs/img_in_image_box_363_737_1275_981.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A02Z%2F-1%2F%2F0f12003d8f731a2d7b29c205b75ed6ff47bba54d2b7e8ce880f97d6089775066" alt="Image" width="63%" /></div>


##### 模拟退火

□ 模仿自然界退火现象而得，利用了物理中固体物质的退火过程与一般优化问题的相似性

从某一初始 $ \underline{\text{温度}} $开始，伴随温度的不断下降，结合 $ \underline{\text{概率突跳}} $特性在解空间中 $ \underline{\text{随机}} $寻找 $ \underline{\text{全局最优解}} $

##### 相似性：

金属 ←→ 问题

能量状态  $ \leftarrow $ 成本函数

温度  $ \leftarrow $ 控制参数

- 完整排列的晶体结构  $ \leftarrow $ 问题的最优解

##### 模拟退火

相似性：

- 金属 ←→ 问题

- 能量状态  $ \leftarrow\rightarrow $ 成本函数

- 温度  $ \leftarrow $ 控制参数

– 完整排列的晶体结构 ←→ 问题的最优解

##### 模拟退火

##### 数学表述

在温度T，分子停留在状态r满足Boltzmann概率分布

 $$ P\{\overline{E}=E(r)\}=\frac{1}{Z(T)}\exp\left(-\frac{E(r)}{k_{B}T}\right) $$ 

 $ \overline{E} $表示分子能量的一个随机变量， $ E(r) $表示状态r的能量， $ k_{B}>0 $为Boltzmann常数。 $ Z(T) $为概率分布的标准化因子：

 $$ Z(T)=\sum_{s\in D}\exp\left(-\frac{E(s)}{k_{B} T}\right) $$ 

##### 模拟退火

##### Boltzman概率分布告诉我们：

（1）在同一个温度，分子停留在能量小状态的概率大于停留在能量大状态的概率

（2）温度越高，不同能量状态对应的概率相差越小；温度足够高时，各状态对应概率基本相同。

（3）随着温度的下降，能量最低状态对应概率越来越大；温度趋于0时，其状态趋于1

##### 模拟退火

Metropolis准则（1953）——以概率接受新状态

若在温度T，当前状态i → 新状态j

若 $ E_{j}<E_{i} $，则接受j为当前状态；

否则，若概率  $ p=\exp[-(E_{j}-E_{i})/k_{B}T] $ 大于 [0,1) 区间的随机数，则仍接受状态 j 为当前状态；若不成立则保留状态 i 为当前状态。

##### Metropolis准则（1953）——以概率接受新状态

 $$ p{=}\exp[{-(E_{i}{-}E_{i})/k_{B}T}] $$ 

在高温下，可接受与当前状态能量差较大的新状态；在低温下，只接受与当前状态能量差较小的新状态。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//4628091f-9125-463a-b483-c9e285a0c7d3/markdown_1/imgs/img_in_image_box_744_685_1357_971.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A01Z%2F-1%2F%2Fa9f663b1078f46e68174813ae71dd7aef41a7c56bdef6fed9f9644d3f4bb213a" alt="Image" width="42%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//4628091f-9125-463a-b483-c9e285a0c7d3/markdown_1/imgs/img_in_image_box_1307_947_1430_1079.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A01Z%2F-1%2F%2F13f2b00c3380e1327c05ea6f1e6abfdbdd50c79fb1137604b1c2dd2f76ec7752" alt="Image" width="8%" /></div>


#### 爬山算法

爬山算法是一种简单的贪心搜索算法，该算法每次从当前解的临近解空间中选择一个最优解作为当前解，直到达到一个局部最优解

☐ 爬山算法实现很简单，其主要缺点是会陷入局部最优解，而不一定能搜索到全局最优解

假设C点为当前解，爬山算法搜索到A点这个局部最优解就会停止搜索，因为在A点无论向那个方向小幅度移动都不能得到更优的解

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//4628091f-9125-463a-b483-c9e285a0c7d3/markdown_2/imgs/img_in_image_box_395_816_1016_952.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A02Z%2F-1%2F%2Fc615329cc3200a4ef6eafbe61269c65157eee923204cbfa9315fd842de826baf" alt="Image" width="43%" /></div>


##### 模拟退火

要从局部最优逃出，必须上行（up-step）

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//4628091f-9125-463a-b483-c9e285a0c7d3/markdown_3/imgs/img_in_image_box_97_265_1431_633.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A03Z%2F-1%2F%2F3018d6a6dc42923ec714e8b361ef717d2451db39a50c03ff6ab3156401a1352b" alt="Image" width="92%" /></div>


SA的上行机制：  $ \exp(\Delta/T) > \text{random}[0,1] $

 $ \triangle= $（当前解）—（下一个解）

T为温度，即SA的控制参数

爬山法是完全的贪心法，每次都鼠目寸光的选择一个当前最优解，因此只能搜索到局部的最优值。模拟退火其实也是一种贪心算法，但是它的搜索过程引入了随机因素。

□ 模拟退火算法以一定的概率来接受一个比当前解要差的解，因此有可能会跳出这个局部的最优解，达到全局的最优解。

□ 模拟退火算法是一种随机算法，并不一定能找到全局的最优解，可以比较快的找到问题的近似最优解。如果参数设置得当，模拟退火算法搜索效率比穷举法要高。

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//c039d364-5b74-448e-b9bf-399b78d2e5ef/markdown_0/imgs/img_in_image_box_1307_957_1428_1079.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A01Z%2F-1%2F%2Ff8b90c6155895619623108b8a6787711950baa22bed21293c656272574b38a39" alt="Image" width="8%" /></div>


##### 模拟退火

##### 特征选择的模拟退火算法

外循环

Step1: 令i=0, k=0, 给出初始温度 $ T_{0} $和初始特征组合x(0)

内循环

Step2: 在 $ x(k) $的邻域 $ N(x(k)) $中选择一个状态 $ x' $，即新特征组合。计算其可分性判据 $ J(x') $，并按概率P接受 $ x(k+1)=x' $

Step3: 如果在 $ T_{i} $下还未达到平衡，则转到Step2

Step4: 如果 $ T_{i} $已经足够低，则结束，当时的特征组合即为算法的结果。否则继续

Step5: 根据温度下降方法计算新的温度 $ T_{i+1} $。转到Step2

□ 所要注意的问题

理论上，降温过程的快慢

结束准则：连续m次转换没有状态改变

确定可行解的邻域和温度下降方法

##### 模拟退火

# 30城市TSP问题（ $ d^{*}=423.741 $）

旅行商问题，即TSP问题（Travelling

Salesman Problem）又译为旅行推销员问题、货郎担问题，是数学领域中著名问题之一。假设有一个旅行商人要拜访n个城市，他必须选择所要走的路径，路径的限制是每个城市只能拜访一次，而且最后要回到原来出发的城市。路径的选择目标是要求得的路径路程为所有路径之中的最小值。

#### 简化的TSP算例（6城市）



<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>City to city</td><td style='text-align: center; word-wrap: break-word;'>1</td><td style='text-align: center; word-wrap: break-word;'>2</td><td style='text-align: center; word-wrap: break-word;'>3</td><td style='text-align: center; word-wrap: break-word;'>4</td><td style='text-align: center; word-wrap: break-word;'>5</td><td style='text-align: center; word-wrap: break-word;'>6</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>1</td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'>12</td><td style='text-align: center; word-wrap: break-word;'>4</td><td style='text-align: center; word-wrap: break-word;'>7</td><td style='text-align: center; word-wrap: break-word;'>9</td><td style='text-align: center; word-wrap: break-word;'>10</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>2</td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'>11</td><td style='text-align: center; word-wrap: break-word;'>20</td><td style='text-align: center; word-wrap: break-word;'>13</td><td style='text-align: center; word-wrap: break-word;'>8</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>3</td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'>6</td><td style='text-align: center; word-wrap: break-word;'>17</td><td style='text-align: center; word-wrap: break-word;'>13</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>4</td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'>6</td><td style='text-align: center; word-wrap: break-word;'>9</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>5</td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'>15</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>6</td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'></td><td style='text-align: center; word-wrap: break-word;'></td></tr></table>

#### 简化的TSP算例（6城市）

参数设置：

☐ Th=200

☐ t=10

☐ r=0.6

☐ N=2

□ 生成新的解：随机选择两个位置，交换其表示的城市

T: 温度

Th: 最高温度

t: 最低温度

BS: 已经找到的

最好解

N:某一温度下达到平衡的搜索次数

#### 简化的TSP算例（6城市）



<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>初始解</td><td style='text-align: center; word-wrap: break-word;'>温度T=200</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Sequence</td><td style='text-align: center; word-wrap: break-word;'>The length of the route</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>132456</td><td style='text-align: center; word-wrap: break-word;'>28</td></tr><tr><td colspan="2">BS</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Sequence</td><td style='text-align: center; word-wrap: break-word;'>The length of the route</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>132456</td><td style='text-align: center; word-wrap: break-word;'>28</td></tr></table>

#### 简化的TSP算例（6城市）

#### 当前解



<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Sequence</td><td style='text-align: center; word-wrap: break-word;'>The length of the route</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>132456</td><td style='text-align: center; word-wrap: break-word;'>28</td></tr></table>

新的解



<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Sequence</td><td style='text-align: center; word-wrap: break-word;'>The length of the route</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>123456</td><td style='text-align: center; word-wrap: break-word;'>30</td></tr></table>

Exp((当前解一新的解)/T)=exp(-2/200)

Random[0,1]=0.7

接受新的解

T: 温度

Th: 最高温度

t: 最低温度

BS: 已经找到的

最好解

N:某一温度下达到平衡的搜索次数

T: 温度

Th: 最高温度

t: 最低温度

BS: 已经找到的

最好解

N:某一温度下达到平衡的搜索次数

#### 简化的TSP算例（6城市）



<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>当前解</td><td style='text-align: center; word-wrap: break-word;'>温度T=200</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Sequence</td><td style='text-align: center; word-wrap: break-word;'>The length of the route</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>123456</td><td style='text-align: center; word-wrap: break-word;'>30</td></tr></table>

<div style="text-align: center;"><div style="text-align: center;">BS</div> </div>




<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Sequence</td><td style='text-align: center; word-wrap: break-word;'>The length of the route</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>132456</td><td style='text-align: center; word-wrap: break-word;'>28</td></tr></table>

T: 温度

Th: 最高温度

t: 最低温度

BS: 已经找到的

最好解

N:某一温度下达到平衡的搜索次数

#### 简化的TSP算例（6城市）

当前解



<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Sequence</td><td style='text-align: center; word-wrap: break-word;'>The length of the route</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>123456</td><td style='text-align: center; word-wrap: break-word;'>30</td></tr></table>

新的解



<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Sequence</td><td style='text-align: center; word-wrap: break-word;'>The length of the route</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>123546</td><td style='text-align: center; word-wrap: break-word;'>36</td></tr></table>

Exp((当前解一新的解)/T)=exp(-6/200)

Random[0,1]=0.99，拒绝新的解

T: 温度

Th: 最高温度

t: 最低温度

BS: 已经找到的

最好解

N:某一温度下达到平衡的搜索次数

#### 简化的TSP算例（6城市）

当前解



<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Sequence</td><td style='text-align: center; word-wrap: break-word;'>The length of the route</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>123456</td><td style='text-align: center; word-wrap: break-word;'>30</td></tr></table>

新的解



<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Sequence</td><td style='text-align: center; word-wrap: break-word;'>The length of the route</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>123465</td><td style='text-align: center; word-wrap: break-word;'>31</td></tr></table>

Exp((新的解一当前解)/T)=exp(-1/200)

Random[0,1]=0.6

接受新的解

T: 温度

Th: 最高温度

t: 最低温度

BS: 已经找到的

最好解

N:某一温度下达到平衡的搜索次数

T: 温度

Th: 最高温度

t: 最低温度

BS: 已经找到的

最好解

N:某一温度下达到平衡的搜索次数

#### 简化的TSP算例（6城市）



<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>当前解</td><td style='text-align: center; word-wrap: break-word;'>温度T=120</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>Sequence</td><td style='text-align: center; word-wrap: break-word;'>The length of the route</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>123456</td><td style='text-align: center; word-wrap: break-word;'>31</td></tr></table>

<div style="text-align: center;"><div style="text-align: center;">新的解</div> </div>




<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Sequence</td><td style='text-align: center; word-wrap: break-word;'>The length of the route</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>213456</td><td style='text-align: center; word-wrap: break-word;'>27</td></tr></table>

#### 新的解优于当前解，接受新的解

T: 温度

Th: 最高温度

t: 最低温度

BS: 已经找到的

最好解

N:某一温度下达到平衡的搜索次数

#### 简化的TSP算例（6城市）

当前解 温度T=120



<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Sequence</td><td style='text-align: center; word-wrap: break-word;'>The length of the route</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>213456</td><td style='text-align: center; word-wrap: break-word;'>27</td></tr></table>

<div style="text-align: center;"><div style="text-align: center;">BS</div> </div>




<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>Sequence</td><td style='text-align: center; word-wrap: break-word;'>The length of the route</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>213456</td><td style='text-align: center; word-wrap: break-word;'>27</td></tr></table>

##### 产生新解的3种策略：1交换

☐ 交换又称2-OPT算法，是最常用的一种算法，其主要思想是在所有城市中随机选取两个城市，然后将2个城市在路径中的序列交换位置产生新的路径。

□ 例如：解空间S是边防每个城市恰好一次的所有路径，解乐意表示为{W1，W2，⋯⋯Wn}是1,2，⋯⋯的一个排列，表明W1城市出发，经过W2，⋯⋯Wn城市，在返回W1城市。新路径的产生：随机产生1和n之间的两相异数k和m。不妨设k<m，则将原路径

□ W1, W2⋯,Wk, W(k+1),⋯,Wm, W(m+1),⋯ Wn

变为新路径：

□ W1, W2⋯⋯, Wm, W(k+1), ⋯·Wk, W(m+1), ⋯ Wn

##### 产生新解的3种策略：2置换

□ 置换的主要思想是：随即在路径中选出两个城市，然后将两个城市之间的城市顺序完全倒置得出新的路径。如上例：产生两个相异数k和m（假设k<m），则将原路径

□ W1, W2⋯W(k-1), Wk, W(k+1), ⋯W(m-1), Wm, W(m+1), ⋯Wn

变为新路径：

□ W1, W2⋯W(k-1), Wm, W(m-1), ⋯W(k+1), Wk, W(m+1), ⋯ Wn

##### 产生新解的3种策略

□ 移位的主要思想是随机选出两个城市，两城市之间的城市同时向右移一位。如：随机选出城市 Wk和Wm（假设k<m），则将原路径

□ W1, W2⋯W(k-1),Wk, W(k+1),⋯W(m-1),Wm, W(m+1),⋯Wn

变为新路径：

□ W1, W2⋯W(k-1),W(m+1),Wk,⋯W(m-1),Wm,⋯Wn

##### TSP寻优过程

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//69d6b321-63b7-487b-b55d-e7f29ad6e6ad/markdown_3/imgs/img_in_chart_box_406_261_1111_848.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A03Z%2F-1%2F%2Fc2e11fa6fd0e66d570a44df17e48e927310c181ccc595fe4516d931098c0e851" alt="Image" width="48%" /></div>


##### TSP寻优过程

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//6f7c8759-d591-433e-a903-c409a2c7dcbe/markdown_0/imgs/img_in_chart_box_406_262_1111_848.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A01Z%2F-1%2F%2Fb94256fe52c8738d5d68b91c58c5f748b51a4b6e0e784514f725a39042cd958e" alt="Image" width="48%" /></div>


##### TSP寻优过程

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//6f7c8759-d591-433e-a903-c409a2c7dcbe/markdown_1/imgs/img_in_chart_box_416_262_1118_849.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A03Z%2F-1%2F%2Fe428fdf92ccd403852ad27b39c6c9f6397a19b136e899f26db9d9112b6444c3f" alt="Image" width="48%" /></div>


##### TSP寻优过程

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//6f7c8759-d591-433e-a903-c409a2c7dcbe/markdown_2/imgs/img_in_chart_box_406_262_1110_848.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A04Z%2F-1%2F%2Fa7d364ba66233f338badc62c79aaa7b896365e3dd975d935e05b9b7fde1ed7c0" alt="Image" width="48%" /></div>


##### TSP寻优过程

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//6f7c8759-d591-433e-a903-c409a2c7dcbe/markdown_3/imgs/img_in_chart_box_406_262_1110_848.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A06Z%2F-1%2F%2F5609e89b3aeefc3488772126c0e0cecc899dead50549f18bd73db3f2174a38c3" alt="Image" width="48%" /></div>


### 遗传算法

#### Genetic Algorithms

##### 遗传算法

近代科学技术发展的显著特点之一是生命科学与工程科学的相互交叉、相互渗透、相互促进。

遗传算法是20世纪60~70年代主要由美国Michigan大学John Holland教授提出.其内涵哲理启迪于自然界生物从低级、简单到高级、复杂，乃至人类这样一个漫长而绝妙的进化过程.借鉴Darwin的物竞天择、优胜劣汰、适者生存的自然选择和自然遗传的机理.其本质是一种求解问题的高效并行全局搜索方法，它能在搜索过程中自动获取和积累有关搜索空间的知识，并自适应地控制搜索过程以求得最优解.

##### 遗传算法

##### 基本概念

GA 的基本思想：

从一初始化的群体出发，通过随机选择(复制)（使群体中优秀的个体有更多的机会传给下一代），交叉（体现了自然界中群体内个体之间的信息交换），和变异（在群体中引入新的变种确保群体中信息的多样性）等遗传操作，使最具有生存能力的染色体以最大可能生存，群体一代一代地进化到搜索空间中越来越好的区域。

##### 生物遗传概念在遗传算法中的对应关系



<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>生物遗传概念</td><td style='text-align: center; word-wrap: break-word;'>遗传算法中的作用</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>适者生存</td><td style='text-align: center; word-wrap: break-word;'>最优目标值的解有最大可能留住</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>个体</td><td style='text-align: center; word-wrap: break-word;'>解</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>染色体</td><td style='text-align: center; word-wrap: break-word;'>解的编码（字符串、向量等）</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>基因</td><td style='text-align: center; word-wrap: break-word;'>解中每一分量的特征（如分量的值）</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>适应性</td><td style='text-align: center; word-wrap: break-word;'>适应函数值</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>群体</td><td style='text-align: center; word-wrap: break-word;'>选定的一组解</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>种群</td><td style='text-align: center; word-wrap: break-word;'>根据适应函数值选取的一组解</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>交叉（基因重组）</td><td style='text-align: center; word-wrap: break-word;'>通过交叉原则产生一组新解的过程</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>变异</td><td style='text-align: center; word-wrap: break-word;'>编码的某一个分量发生变化的过程</td></tr></table>

##### 遗传算法

染色体

基因

Example 1 某快餐店女作出以下三项决定。

价格 汉堡包的价格应定在1美元还是50美分；

饮料 和汉堡包一起供应的应该是酒还是可乐；服务速度 饭店应提供慢的还是快的服务速度。

Solution：编码 有3个决策变量共有 $ 2^{3}=8 $种方案

用三位 0-1 数串，表示一个方案  $ (a_{1}\quad a_{2}\quad a_{3})\quad a_{i}=0,1 $

 $ a_{1} $ 表示价格 0——高价格 1——低价格

 $ a_{2} $ 表示饮料 0——酒 1——可乐

 $ a_{3} $ 表示速度 0——慢 1——快

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//b58b6f31-cfe6-4481-85ad-682d9ceffdc8/markdown_0/imgs/img_in_image_box_1273_956_1430_1078.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A02Z%2F-1%2F%2Ff00939ad3395abec833aed6edc36990ffebbfce444d4d513ec416cf5d1cd5ea5" alt="Image" width="10%" /></div>


##### 遗传算法

适应函数 即目标函数就取每种方案实行后的利润，为简单起，每种方案所对应的利润（适应值）恰为这三位二进制所对应的十进制数值.

确定群体规模 N 取 N=4，在 8 个方案中随机抽取 4 个方案作为初始群体（第 0 代）.



<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>$ x_i $</td><td style='text-align: center; word-wrap: break-word;'>011</td><td style='text-align: center; word-wrap: break-word;'>001</td><td style='text-align: center; word-wrap: break-word;'>110</td><td style='text-align: center; word-wrap: break-word;'>010</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>$ f(x_i) $</td><td style='text-align: center; word-wrap: break-word;'>3</td><td style='text-align: center; word-wrap: break-word;'>1</td><td style='text-align: center; word-wrap: break-word;'>6</td><td style='text-align: center; word-wrap: break-word;'>2</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>总和</td><td style='text-align: center; word-wrap: break-word;'>12</td><td colspan="3">GA 不是一个个的考虑方案，而是\n届时考虑1N 平均集，3体现最大值性.</td></tr></table>

##### 轮盘赌选择：

步骤：1、求群体中所有个体的适应值的总和 S；

2、产生一个在0与S之间的随机数m；

3、从群体中编号为1的个体开始，将其适应值与后继个体的适应值相加，直到累加和等于或大于m，则停止.其中那个最后加进去的个体即为选择的个体.

如 随机数 5 2 12 9

选择的个体 110 011 010 110

由选择算子产生的新群体（可能有重复）称为种群，其规模仍为N.

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//b58b6f31-cfe6-4481-85ad-682d9ceffdc8/markdown_2/imgs/img_in_image_box_1105_693_1417_1061.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A03Z%2F-1%2F%2Fc2fae760b23c6381cf3a27ff5518f09ef10d01dd7011ea2c962035bb655defdb" alt="Image" width="21%" /></div>


<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//b58b6f31-cfe6-4481-85ad-682d9ceffdc8/markdown_2/imgs/img_in_image_box_759_970_1428_1075.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A03Z%2F-1%2F%2Fb2b7242d4fcd8474bb6324ce18bc3088a6b24724da922eaa9ff67c621c7bbffe" alt="Image" width="46%" /></div>


##### 遗传算法

 $$ x_{i} $$ 

 $$ f(x_{i}) $$ 

总和 17 最小值 2 平均值 4.25 最大值 6

选择算子作用的效果是提高了群体的平均适应值及最差的适应值，低适应值的个体趋于被淘汰，高适应值的个体趋于被复制。但是以损失群体的多样性为代价，选择算子并没有产生新的个体，当然群体中最好个体的适应值不会改进。

假设  $ P_{c} = 50\% $

##### 交叉（杂交）算子

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//624cd3b0-0c74-44e5-8e70-5474710a6d1a/markdown_0/imgs/img_in_image_box_87_264_976_438.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A02Z%2F-1%2F%2F2132501b34b9215ac63e9ec4e957e3aaec163e172c146c6489f8187a034a7659" alt="Image" width="61%" /></div>


<div style="text-align: center;"><div style="text-align: center;">总和 17 最小值 2 平均值 4.25 最大值 7</div> </div>


个数）的随机数  $ i $ (交叉点)，然后配对的两个个体相互交换从  $ i+1 $ 到  $ l $ 的位子．如对  $ x_1 $， $ x_2 $ 配对且交叉点选

在2，则

 $$ \begin{array}{l}11|0\rightarrow111\\01|1\quad010\end{array} $$ 

对种群要确定交叉概率  $ P_{c} $

随机选择  $ N \times P_{c} $ 个个体进行交叉，其余不变.

<div style="text-align: center;"><img src="https://pplines-online.bj.bcebos.com/deploy/official/paddleocr/pp-ocr-vl-16-online//624cd3b0-0c74-44e5-8e70-5474710a6d1a/markdown_0/imgs/img_in_image_box_1304_955_1428_1078.jpg?authorization=bce-auth-v1%2FALTAKDN8mY5KlNI7zaRpLmOqrw%2F2026-07-02T13%3A18%3A03Z%2F-1%2F%2F0d5dc9040fb3c4aa5925151fbcc53c3a512efad72846d9248ea631a3734f837f" alt="Image" width="8%" /></div>


##### 遗传算法

显然，利用选择、交叉算子可以产生具有更高平均适应值和更好个体的群体．但如果仅仅如此，很容易近亲繁殖，会早熟（局部最优解）．

##### 变异算子

以一个很小的变异概率  $ P_m (= 0.02) $ 随机地改变染色体上的某个基因  $ \begin{pmatrix} 0 \to 1 \\ 1 \to 0 \end{pmatrix} $，具有增加群体多样性的效果。

如：选择  $ x_3 $ 第 2 位，则  $ \begin{array}{c} 0 \\ 1 \end{array} \to 000 $ 得到新的群体。

称为第 1 代，再进行选择、交叉、变异……

##### 遗传算法

Example 2 用 GA 求

共有16种方案

群体规模为4

Solution：对连续变量求解，要解决如何编码问题。

假设对解的误差要求是  $ \frac{1}{16} $，则可采用4位二进制编码，对应关系  $ (a \ b \ c \ ^{\circ} d) \leftrightarrow \frac{a}{2} + \frac{b}{4} + \frac{c}{8} + \frac{d}{16} $

一次迭代的结果．(交叉概率  $ P_{c}=1 $，变异概率  $ P_{m}=0.02 $)



<table border=1 style='margin: auto; word-wrap: break-word;'><tr><td style='text-align: center; word-wrap: break-word;'>x值</td><td style='text-align: center; word-wrap: break-word;'>群体 $ x_i $</td><td style='text-align: center; word-wrap: break-word;'>$ f(x_i) $</td><td style='text-align: center; word-wrap: break-word;'>概率分布</td><td style='text-align: center; word-wrap: break-word;'>种群</td><td style='text-align: center; word-wrap: break-word;'>交叉位</td><td style='text-align: center; word-wrap: break-word;'>交叉结果</td><td style='text-align: center; word-wrap: break-word;'>变异?</td><td style='text-align: center; word-wrap: break-word;'>新群体</td><td style='text-align: center; word-wrap: break-word;'>x值</td><td style='text-align: center; word-wrap: break-word;'>$ f(x_i) $</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>1/16</td><td style='text-align: center; word-wrap: break-word;'>0001</td><td style='text-align: center; word-wrap: break-word;'>0.99</td><td style='text-align: center; word-wrap: break-word;'>0.318</td><td style='text-align: center; word-wrap: break-word;'>0001</td><td style='text-align: center; word-wrap: break-word;'>00|01</td><td style='text-align: center; word-wrap: break-word;'>0000</td><td style='text-align: center; word-wrap: break-word;'>N</td><td style='text-align: center; word-wrap: break-word;'>0000</td><td style='text-align: center; word-wrap: break-word;'>0</td><td style='text-align: center; word-wrap: break-word;'>1</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>1/4</td><td style='text-align: center; word-wrap: break-word;'>0100</td><td style='text-align: center; word-wrap: break-word;'>0.93</td><td style='text-align: center; word-wrap: break-word;'>0.299</td><td style='text-align: center; word-wrap: break-word;'>0100</td><td style='text-align: center; word-wrap: break-word;'>01|00</td><td style='text-align: center; word-wrap: break-word;'>0101</td><td style='text-align: center; word-wrap: break-word;'>Y</td><td style='text-align: center; word-wrap: break-word;'>1101</td><td style='text-align: center; word-wrap: break-word;'>13/16</td><td style='text-align: center; word-wrap: break-word;'>0.34</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>3/16</td><td style='text-align: center; word-wrap: break-word;'>0011</td><td style='text-align: center; word-wrap: break-word;'>0.96</td><td style='text-align: center; word-wrap: break-word;'>0.308</td><td style='text-align: center; word-wrap: break-word;'>0001</td><td style='text-align: center; word-wrap: break-word;'>0001</td><td style='text-align: center; word-wrap: break-word;'>0001</td><td style='text-align: center; word-wrap: break-word;'>N</td><td style='text-align: center; word-wrap: break-word;'>0001</td><td style='text-align: center; word-wrap: break-word;'>1/16</td><td style='text-align: center; word-wrap: break-word;'>0.99</td></tr><tr><td style='text-align: center; word-wrap: break-word;'>7/8</td><td style='text-align: center; word-wrap: break-word;'>1110</td><td style='text-align: center; word-wrap: break-word;'>0.23</td><td style='text-align: center; word-wrap: break-word;'>0.075</td><td style='text-align: center; word-wrap: break-word;'>0011</td><td style='text-align: center; word-wrap: break-word;'>0011</td><td style='text-align: center; word-wrap: break-word;'>0011</td><td style='text-align: center; word-wrap: break-word;'>N</td><td style='text-align: center; word-wrap: break-word;'>0011</td><td style='text-align: center; word-wrap: break-word;'>3/16</td><td style='text-align: center; word-wrap: break-word;'>0.96</td></tr></table>

第0代 总和3.133 最小值0.234 平均值0.7833 最大值0.996;

第1代总和3.301最小值0.340平均值0.8253最大值1.

#### 遗传算法的步骤

step 1 选择问题解的一个编码，给出一个有 N 个染色体的初始群体 pop(1)，t = 1;

step 2 对群体中每一个染色体  $ pop_{i}(t) $ 计算它的适应函数值  $ f_{i}=fitness(pop_{i}(t)) $;

step 3 若停止规则满足，则算法停止，否则计算概率

 $$ p_{i}=\frac{f_{i}}{\sum\limits_{j=1}^{N}f_{j}}\qquad i=1\left[\begin{array}{l l l l}{1}\\ {\Box}\\ {N}\end{array}\right] $$ 

并以此概率分布，从 pop(t) 中随机选 N 个染色体构成一个种群， $ \text{newpop}(t) $；

##### 遗传算法的步骤

step 4 通过交叉(交叉概率)为  $ P_{c} $，得到有 N 个染色体的  $ crosspop(t+1) $;

step 5 以较小的概率（变异概率） $ P_{m} $ 使得某染色体的一个基因发生变异，形成新的群体  $ mutpop(t+1) $ 令  $ t = t+1 $  $ pop(t) = nutpop(t) $ go to step 2.

# 遗传算法的优越性：

1、作为数值求解方法，具有普遍性

可以不连续、不规则、伴有噪声，甚至不一定要显式写出。

乎没有要求，总能以极大的概率找到全局最优解；

2、GA 在求解很多组合优化问题这是大键很高的技巧和对问题有非常深入的了解，在给问题的决策变量编码后，其计算过程是比较简单的，且可较快地得到一个满意解；

3、与其他启发式算法有较好的兼容性，易与别的技术相结合，形成性能更优的问题求解方法.

##### 遗传算法的问题：

# 1. 编码问题

编码是 GA 中的基础工作之一，GA 不能直接处理解空间的解数据，必须通过编码表成遗传空间的基因型数据。比较直观和常规的方法是 0、1 二进制编码，称为常规码。这种编码方法使算法的算子构造比较简单。

# 2. 欺骗问题

在遗传进化的初期，通常会产生一些超常个体，按比例选择，这些个体竞争力太强，而控制了选择过程，影响算法的全局优化性能；在遗传进化的后期，即算法接近收敛时，由于种群中个体适应度差异较小时，继续优化的潜能降低，可能获得某个局部最优解。

## 谢谢！

