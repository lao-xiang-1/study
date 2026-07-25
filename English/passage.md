#en/Passage
	This module provides a small set of utility functions for working with tree-like
data structures, such as nested tuples, lists, and dicts. We call these
structures pytrees. They are trees in that they are defined recursively (any
non-pytree is a pytree, i.e. a leaf, and any pytree of pytrees is a pytree) and
can be operated on recursively (object identity equivalence is not preserved by
mapping operations, and the structures cannot contain reference cycles).
	The set of Python types that are considered pytree nodes (e.g. that can be
mapped over, rather than treated as leaves) is extensible. There is a single
module-level registry of types, and class hierarchy is ignored. By registering a
new pytree node type, that type in effect becomes transparent to the utility
functions in this file.
	The primary purpose of this module is to enable the interoperability between
user defined data structures and JAX transformations (e.g. `jit`). This is not
meant to be a general purpose tree-like data structure handling library.
?
1. 本模块提供了一小组用于处理树状数据结构（tree-like data structures）的实用函数，这些数据结构包括嵌套元组、列表和字典。我们将这些结构称为 pytree。它们是树，因为它们是递归定义的（任何非 pytree 都是 pytree，即叶子节点；任何 pytree 的 pytree 也是 pytree），并且可以递归操作（映射操作不保留对象标识等价性，且这些结构不能包含引用循环）。
2. 被视为 pytree 节点（即可以被映射遍历，而非当作叶子节点处理）的 Python 类型集合是可扩展的。存在一个模块级别的类型注册表，且类继承层次结构被忽略。通过注册一个新的 pytree 节点类型，该类型实际上对本文件中的实用函数变为透明的。
3. 本模块的主要目的是实现用户自定义数据结构与 JAX 变换（如 `jit`）之间的互操作性。它并非一个通用的树状数据结构处理库。
<!--SR:!2026-07-26,3,250-->

#en/Passage
If checked, and you or the deck author altered the schema of a note type, Anki will merge the two versions instead of keeping both.
Altering a note type's schema means adding, removing, or reordering fields or templates, or changing the sort field. As a counterexample, changing the front side of an existing template does _not_ constitute a schema change.
Warning: This will require a one-way sync, and may mark existing notes as modified.
?
如果勾选此项，且您或卡片组作者修改了笔记类型的结构，Anki 将合并两个版本，而不是同时保留两者。
修改笔记类型的结构是指添加、删除或重新排列字段或模板，或者更改排序字段。反例：更改现有模板的正面内容并不构成结构变更。
警告：这将需要进行单向同步，并可能将现有笔记标记为已修改。
<!--SR:!2026-07-27,2,210-->

#en/Passage
During inference, all prediction errors and feedback terms are computed first using the current network state, and only then are the latent variables  $\mathbf{x}^{(l)}$ updated. This ensures that each update step is based on a consistent energy landscape and avoids using partially updated states within the same iteration. Conceptually, this corresponds to a synchronous update scheme where all neurons compute their next state based on the same network snapshot.
?
在推断过程中，首先利用当前网络状态计算所有预测误差和反馈项，然后才对隐变量 $\mathbf{x}^{(l)}$ 进行更新。这确保了每次更新步骤都基于一致的能量景观，并避免在同一次迭代中使用部分更新的状态。从概念上讲，这对应于一种同步更新方案，其中所有神经元都基于同一网络快照计算其下一状态。
<!--SR:!2026-07-27,3,250-->

#en/Passage
One of the original motivations behind predictive coding is its potential biological plausibility: that the brain could implement something akin to deep hierarchical learning using local computations. Locality typically refers to whether a computation depends only on information from a given layer and its immediate neighbors. This concept is important both for computational efficiency and biological plausibility.
?
预测编码最初的动机之一是其潜在的生物学合理性（biological plausibility）：即大脑可以利用局部计算实现类似于深层层级学习的功能。**局部性**（locality）通常指一个计算是否仅依赖于来自给定层及其直接相邻层的信息。这一概念对于计算效率和生物学合理性都很重要。
<!--SR:!2026-07-26,3,250-->

#en/Passage
Remark. The model can also support **anytime inference**（**随时推断**）, where well-predicted inputs converge in fewer steps, and additional inference steps are taken to improve predictions in ambiguous cases. This adaptivity offers potential energy savings in embedded or neuromorphic deployments.
The algorithm can be modified in this spirit by choosing a sufficiently large maximum step count $T_{infer}$, and running the inference loop until either $T_{infer}$ steps have been performed or convergence has been detected. Here convergence means, for instance, that the norm of the latest update (or updates over a longer patience window) across all latent variables falls below a preset threshold. In machine learning terminology, this could be phrased as inference with sample-wise early stopping.
?
**备注。** 该模型还支持**随时推断（anytime inference）**：对于容易预测的输入，只需更少的步骤即可收敛；而对于模糊或不确定的情况，则可以执行额外的推断步骤来改善预测结果。这种自适应特性为嵌入式或神经形态硬件部署提供了潜在的节能空间。
按照这一思路，可以对算法进行修改：设置一个足够大的最大步数 ，然后运行推断循环，直到执行了  步或检测到收敛为止。这里的收敛可以定义为，例如，所有隐变量的最新更新（或更长耐心窗口内的更新）的范数低于某个预设阈值。用机器学习的术语来说，这可以表述为**带样本级早停的推断（sample-wise early stopping）**。
<!--SR:!2026-07-26,2,230-->