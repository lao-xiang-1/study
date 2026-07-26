# PCX 教程 0：预测编码网络 (Predictive Coding Networks) — 详细解释

## 一、项目背景

**PCX** 是一个基于 JAX 的 Python 库，用于构建可高度配置的**预测编码网络 (Predictive Coding Networks, PCNs)**。预测编码是一种受神经科学启发的学习框架，其核心思想是：网络通过最小化**能量 (energy)** 来学习，能量的定义基于网络中每层对输入的预测与真实值之间的差异。

与传统的反向传播不同，PCN 在每次权重更新之前会进行多步**推理 (inference)**，让网络的内部状态（即每层的激活值）收敛到一个能量较低的状态。

---

## 二、逐 Cell 详解

### Cell 2 — 导入库

```python
import jax, jax.numpy as jnp, equinox as eqx
import pcx as px
import pcx.predictive_coding as pxc
import pcx.nn as pxnn
import pcx.functional as pxf
import pcx.utils as pxu

px.RKG.seed(0)  # 设置随机种子以保证可复现性
```

关键模块说明：

- **`pcx.predictive_coding`** — 核心模块，提供 `EnergyModule`、`Vode`、能量函数等
- **`pcx.nn`** — 神经网络层（如 `Linear`）
- **`pcx.functional`** — 函数式变换（`vmap`、`jit`、`value_and_grad` 等），封装了 JAX 的对应函数并添加了对 PCX 参数追踪的支持
- **`pcx.utils`** — 工具函数（优化器 `Optim`、掩码 `Mask`、参数保存/加载等）

### Cell 3 — 模型定义（核心）

```python
class Model(pxc.EnergyModule):
```

这是整个教程最核心的部分。`EnergyModule` 是 PCX 的基类，任何 PC 模型都必须继承它，从而获得**能量 (energy)** 的概念。

#### 构造函数参数

|参数|含义|
|---|---|
|`input_dim=2`|输入维度（Two Moons 是二维数据）|
|`hidden_dim=32`|隐藏层维度|
|`output_dim=2`|输出维度（二分类）|
|`nm_layers=3`|总层数（1 输入 + 1 隐藏 + 1 输出 = 3）|
|`act_fn`|激活函数（如 `leaky_relu`）|

#### 关键组件

**1. `Vode` — PCN 的核心抽象**

`Vode`（可以理解为 "Variable Node" 或 "Vector Node"）是预测编码网络的基本计算单元。每个 `Vode` 维护以下状态：

- **`u`** — 当前层的输入激活值（前一层输出经过权重矩阵后的结果）
- **`h`** — 当前层的目标/标签值（在训练时可以固定为真实标签）
- **`x`** — 当前层的输出激活值（推理后的值）
- 内部**能量 (energy)** — 衡量 `u` 和 `h` 之间差异的标量

默认规则集 `{"STATUS.INIT": ("h, u <- u",)}` 的含义是：在初始化状态 (`STATUS.INIT`) 下，当设置 `u` 时，同时将 `u` 的值复制到 `h` 和 `x`。这等价于普通前向传播的行为。

**2. 最后一层的特殊处理**

```python
self.vodes = [
    pxc.Vode() for _ in range(nm_layers - 1)
] + [pxc.Vode(pxc.ce_energy)]  # 最后一层使用交叉熵能量
self.vodes[-1].h.frozen = True  # 冻结最后一层的 h（标签）
```

- 前两层使用**默认能量函数**（通常是 MSE 风格的能量）
- 最后一层使用 **`ce_energy`**（交叉熵能量），等价于分类任务中的交叉熵损失
- `frozen = True` 表示最后一层的 `h`（目标值）在推理过程中**不会被更新**，因为它是固定的标签

**3. 前向传播 `__call__`**

```python
def __call__(self, x, y):
    for v, l in zip(self.vodes[:-1], self.layers[:-1]):
        x = v(self.act_fn(l(x)))  # 线性变换 → 激活 → Vode
    x = self.vodes[-1](self.layers[-1](x))  # 最后一层（无激活函数）
    if y is not None:
        self.vodes[-1].set("h", y)  # 训练时固定标签
    return self.vodes[-1].get("u")  # 返回最后一层的输入激活作为 logits
```

**数据流**：`输入 x → Linear → 激活函数 → Vode(保存状态+计算能量) → ... → 最后一层 Vode → 返回 u (logits)`

### Cell 4 — `vmap` 批量化

```python
@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=(0, 0), out_axes=0)
def forward(x, y, *, model: Model):
    return model(x, y)
```

这是 PCX 对 JAX `vmap` 的封装，关键创新是 **`kwargs_mask`**（通过 `pxu.M(...)` 创建）：

- **`VodeParam` 和 `VodeParam.Cache`**（每个样本独立的状态）→ 沿第 0 维 batch（`0`）
- **其他参数**（如权重 `LayerParam`，所有样本共享）→ 不 batch（`None`）

这解决了 PCN 中的一个关键问题：**权重是跨样本共享的，但 Vode 的激活状态是每个样本独立的**。

第二个函数 `energy` 类似，但：

- `out_axes=(None, 0)` — 能量是标量（求和后的结果），logits 是 batched
- 使用 `jax.lax.psum` 将各设备的能量求和（用于多设备训练）

### Cell 5 — 训练步骤（最关键的部分）

```python
@pxf.jit(static_argnums=0)
def train_on_batch(T, x, y, *, model, optim_w, optim_h):
```

`T` 是推理步数（inference steps），这是 PCN 特有的超参数。整个训练分三步：

#### 第一步：初始化 (Init)

```python
with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
    forward(x, y, model=model)
```

触发 `STATUS.INIT` 规则：`h, u <- u`，即用前向传播的值初始化所有 Vode 的状态。同时清除 Vode 的缓存。

#### 第二步：推理 (Inference) — PCN 的核心

```python
optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))  # 初始化状态优化器

for _ in range(T):  # 执行 T 步推理
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        (e, y_), g = pxf.value_and_grad(
            pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True]),  # 只对非冻结的 Vode 求导
            has_aux=True
        )(energy)(x, model=model)
    optim_h.step(model, g["model"])  # 用梯度更新 Vode 的状态（不是权重！）
```

**这是 PCN 区别于传统神经网络的关键**：

- 这里计算的是能量对 **Vode 状态 (`u`)** 的梯度，而不是对权重的梯度
- 通过梯度下降更新每层的激活值，使网络的总能量最小化
- 这模拟了大脑中"感知推理"的过程：给定输入和标签，调整内部表示以最好地解释观察
- `frozen=True` 的 Vode（即最后一层的标签 `h`）不会被更新

#### 第三步：权重更新 (Weight Update)

```python
with pxu.step(model, clear_params=pxc.VodeParam.Cache):
    (e, y_), g = pxf.value_and_grad(
        pxu.M(pxnn.LayerParam).to([False, True]),  # 只对权重参数求导
        has_aux=True
    )(energy)(x, model=model)

optim_w.step(model, g["model"], scale_by=1.0/x.shape[0])  # 更新权重
```

推理完成后，计算能量对**权重 (`LayerParam`)** 的梯度，并用 AdamW 优化器更新权重。

**总结 PCN 训练循环**：

```
对每个 batch:
  1. 前向传播，初始化 Vode 状态
  2. 重复 T 次：计算能量 → 求 Vode 状态的梯度 → 更新 Vode 状态（推理）
  3. 计算能量 → 求权重的梯度 → 更新权重（学习）
```

### Cell 6 — 评估函数

```python
@pxf.jit()
def eval_on_batch(x, y, *, model):
    model.eval()
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        y_ = forward(x, None, model=model).argmax(axis=-1)  # y=None，不固定标签
    return (y_ == y).mean(), y_
```

评估时：

- 只做一次前向传播（不进行推理迭代）
- `y=None`，所以最后一层的 `h` 保持为 `u`（即用自己的预测作为目标）
- 直接取 `argmax` 作为分类结果

### Cell 7-8 — 模型和优化器实例化

```python
model = Model(input_dim=2, hidden_dim=32, output_dim=2, nm_layers=3, act_fn=jax.nn.leaky_relu)

optim_w = pxu.Optim(lambda: optax.adamw(1e-2), pxu.M(pxnn.LayerParam)(model))
# 权重优化器：AdamW，lr=1e-2，只优化 LayerParam

optim_h = pxu.Optim(lambda: optax.sgd(1e-2, momentum=0.5, nesterov=True))
# 状态优化器：SGD + momentum + Nesterov，用于推理时更新 Vode 状态
```

两个优化器的区别：

- **`optim_w`** — 更新权重，在创建时就指定了目标参数（`LayerParam`）
- **`optim_h`** — 更新 Vode 状态，每个 batch 需要重新初始化（因为状态是 batch-dependent 的），所以在 Cell 5 中才调用 `optim_h.init(...)`

### Cell 9-10 — 数据准备

```python
X, y = make_moons(n_samples=1024, noise=0.2, random_state=42)
train_dl = list(zip(X.reshape(-1, batch_size, 2), y.reshape(-1, batch_size)))
```

生成 "Two Moons" 数据集并将其重组为 batch 格式。这是一个经典的二分类问题，两个类别的数据点形成两个半月形状。

### Cell 11 — 训练循环

```python
for e in range(nm_epochs):
    random.shuffle(train_dl)
    train(train_dl, T=8, model=model, optim_w=optim_w, optim_h=optim_h)
    a, y = eval(test_dl, model=model)
```

- `T=8`：每次训练进行 8 步推理
- 由于 JIT 编译，`"Training!"` 只会在第一次调用和结构变化时打印（JAX 会追踪并编译计算图）

### Cell 12 — 模型保存/加载

```python
pxu.save_params(model, "model")
pxu.load_params(model, "model")
```

PCX 提供工具来保存/加载模型参数。默认行为是保存所有权重 (`LayerParam`)，忽略 Vode 状态。

### Cell 13 — 可视化决策边界

```python
model.clear_params(pxc.VodeParam)  # 清除 Vode 状态以使用不同的 batch size
```

在网格上评估模型以可视化决策边界。由于之前训练时 Vode 状态被固定为 batch_size=32，这里需要先清除 Vode 状态，然后才能用不同的 batch size（96×96 个网格点）进行前向传播。

---

## 三、PCN 与传统神经网络的关键区别

```
传统神经网络:
  输入 → 前向传播 → 计算损失 → 反向传播(求权重梯度) → 更新权重

预测编码网络 (PCN):
  输入 → 前向传播(初始化 Vode) → 重复 T 步:
    ├─ 求 Vode 状态梯度 → 更新 Vode 状态 (推理/感知)
  → 求权重梯度 → 更新权重 (学习)
```

PCN 的额外推理步骤使得网络能够在权重更新之前"思考"当前样本，调整内部表示以最小化能量。这种机制更接近生物大脑的工作方式，并且理论上可以支持更灵活的推理和学习模式。