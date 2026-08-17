# XLA (Accelerated Linear Algebra)

## 1. 概述

XLA（Accelerated Linear Algebra，加速线性代数）是由 Google 开发的**领域特定编译器（Domain-Specific Compiler）**，用于加速数值计算，特别是机器学习工作负载。

- **起源**：最初为 TensorFlow 设计
- **现状**：JAX 深度依赖 XLA 作为其后端编译器
- **核心思想**：将高级数值计算图编译为高度优化的机器码，在 CPU/GPU/TPU 上高效执行

---

## 2. 核心架构

### 2.1 编译流程

```
Python 代码
    ↓
JAX Trace → JAXPR（JAX 中间表示）
    ↓
XLA HLO（High Level Operations）
    ↓
XLA 编译器优化
    ↓
LLVM IR（CPU）/ PTX（NVIDIA GPU）/ TPU 指令
    ↓
机器码执行
```

### 2.2 关键中间表示

| 表示层级 | 名称 | 说明 |
|---------|------|------|
| JAX 层 | JAXPR | JAX 的函数式 IR，基于 JAX 的 tracing 生成 |
| XLA 层 | HLO | High Level Operations，XLA 的核心 IR |
| 底层 | LLVM / PTX / TPU | 目标平台的机器码 |

### 2.3 HLO 运算类型

HLO（High Level Operations）是 XLA 的核心中间表示，主要包括：

- **Element-wise 运算**：`add`, `multiply`, `sin`, `log` 等
- **归约运算**：`reduce`, `reduce_window`（用于 sum、mean、max pooling 等）
- **矩阵运算**：`dot`, `convolution`
- **形状操作**：`reshape`, `transpose`, `broadcast`, `slice`, `concatenate`
- **控制流**：`conditional`, `while`, `call`, `map`
- **集合通信**（多设备）：`all_reduce`, `all_gather`, `all_to_all`, `collective_permute`

---

## 3. XLA 在 JAX 中的角色

### 3.1 JAX 的执行模型

JAX 默认是**即时执行（eager execution）**，但通过 `jax.jit` 可以触发 XLA 编译：

```python
import jax
import jax.numpy as jnp

def f(x):
    return jnp.sin(x) ** 2 + jnp.cos(x) ** 2

# eager 执行：逐运算符在 GPU/CPU 上执行
result = f(jnp.ones(1000))

# XLA 编译执行：将整个函数编译为一个融合内核
f_jit = jax.jit(f)
result = f_jit(jnp.ones(1000))  # 第一次调用编译，后续直接执行
```

### 3.2 XLA 带来的优势

1. **算子融合（Fusion）**
   - 将多个 element-wise 运算合并为单个内核
   - 减少内存带宽消耗：例如 `y = exp(sin(x) + 1)` 只需一次内存读写

2. **布局优化（Layout Optimization）**
   - 自动选择最优的内存布局（row-major / column-major）

3. **代数化简（Algebraic Simplification）**
   - 自动识别并简化数学表达式，如 `sin²(x) + cos²(x) → 1`

4. **自动并行化**
   - 自动生成向量化代码（SIMD）
   - GPU/TPU 上的线程调度优化

5. **内存优化**
   - 减少中间结果的内存分配
   - 优化缓冲区复用

---

## 4. 关键 XLA 特性与 JAX API

### 4.1 `jax.jit` —— 单设备编译

```python
@jax.jit
def matmul_and_relu(x, w):
    return jnp.maximum(x @ w, 0)
```

- 将 Python 函数编译为 XLA HLO
- 首次调用有编译开销（compilation overhead）
- 后续调用直接执行编译好的二进制

### 4.2 `jax.pmap` —— 多设备并行

```python
@jax.pmap
def step(params, batch):
    # 在多个设备上并行执行
    return update(params, batch)
```

- 使用 XLA 的集合通信原语（`all_reduce` 等）
- SPMD（Single Program Multiple Data）执行模型

### 4.3 `jax.grad` + `jax.jit`

```python
@jax.jit
def loss_fn(params, x, y):
    return jnp.mean((model(params, x) - y) ** 2)

grad_fn = jax.jit(jax.grad(loss_fn))  # XLA 也能编译反向传播
```

- XLA 自动微分：JAX 生成前向图的反向 HLO
- 整个前向+反向被编译为一个优化的计算图

### 4.4 `jax.vmap` —— 自动向量化

```python
# batched 推理：将单样本函数自动转换为批处理版本
batched_predict = jax.vmap(predict, in_axes=(None, 0))
```

- `vmap` 本身不直接调用 XLA，但与 `jit` 配合时能生成高效的批处理内核

---

## 5. XLA 的编译开销

### 5.1 编译缓存

XLA 编译结果会被缓存：
- **进程内缓存**：同一 Python 进程中重复调用相同形状的函数无需重新编译
- **持久化缓存**（Persistent Compilation Cache）：JAX 0.4.14+ 支持将编译结果缓存到磁盘

```python
import jax

# 启用持久化编译缓存
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
```

### 5.2 重编译触发条件

以下变化会触发 XLA 重新编译：
- 输入张量的**形状（shape）**改变
- 输入张量的**数据类型（dtype）**改变
- 静态参数（static arguments）的值改变
- 代码逻辑改变

> **注意**：JAX 对形状敏感，但不对具体数值敏感（除非是 `static_argnums` 标记的参数）

### 5.3 避免重编译的技巧

```python
# ❌ 坏：每次调用形状不同，反复编译
for i in range(10):
    jax.jit(fn)(jnp.ones(i))  # 编译 10 次

# ✅ 好：使用 padded 批次或固定形状
@partial(jax.jit, static_argnums=(1,))
def fn(x, length):
    return x[:length]
```

---

## 6. 调试 XLA

### 6.1 查看生成的 HLO

```python
# 查看 JAXPR（JAX 中间表示）
print(jax.make_jaxpr(fn)(*args))

# 查看 XLA HLO（文本形式）
compiled = jax.jit(fn).lower(*args).compile()
print(compiled.as_text())

# 查看优化后的 HLO
print(jax.jit(fn).lower(*args).compiler_ir(dialect="hlo"))

# 查看 MHLO / StableHLO（MLIR 格式）
print(jax.jit(fn).lower(*args).compiler_ir(dialect="mhlo"))
```

### 6.2 性能分析

```python
# 生成性能分析文件
with jax.profiler.trace("/tmp/jax-trace"):
    result = jax.jit(fn)(x)

# 在 TensorBoard 中查看
# tensorboard --logdir=/tmp/jax-trace
```

### 6.3 常用环境变量

```bash
# 打印编译的 HLO
export XLA_FLAGS="--xla_dump_to=/tmp/hlo_dump"

# 禁用某些优化以调试
export XLA_FLAGS="--xla_disable_all_hlo_passes"

# TPU 特定调试
export JAX_LOG_COMPILES=1  # 打印每次 XLA 编译的信息
```

---

## 7. XLA vs 其他编译器

| 特性 | XLA | TorchDynamo / torch.compile | TVM |
|------|-----|---------------------------|-----|
| 所属生态 | JAX / TensorFlow | PyTorch | Apache（独立） |
| IR | HLO / StableHLO | TorchFX Graph → Inductor | Relay / TIR |
| 融合能力 | 强 | 强 | 极强（可搜索调度） |
| 自定义算子 | 需 Custom Call | 较灵活 | 灵活 |
| Auto-Tuning | 有限 | 有限 | Ansor / MetaSchedule |
| 多后端 | CPU/GPU/TPU | CPU/GPU | CPU/GPU/FPGA/ASIC |

---

## 8. StableHLO 与 MLIR

### 8.1 StableHLO

- **StableHLO** 是 HLO 的可移植、稳定版本
- 基于 MLIR（Multi-Level Intermediate Representation）框架
- 目标：成为不同 ML 框架之间的标准化编译 IR
- JAX 可以通过 `dialect="stablehlo"` 导出 StableHLO

### 8.2 MLIR 生态

```
Python
  ↓
JAXPR
  ↓
StableHLO / MHLO  (MLIR Dialect)
  ↓
XLA 优化 Passes
  ↓
目标后端 (LLVM / TPU / GPU)
```

---

## 9. 常见问题

### Q: 为什么 `jax.jit` 第一次调用很慢？
A: XLA 需要时间将 Python 函数编译为机器码。这是**编译开销（compilation overhead）**，后续调用会快很多。

### Q: 为什么修改了数组值还需要重新编译？
A: 实际上只关心**形状和 dtype**，不关心具体数值。如果形状没变，不会重编译。

### Q: XLA 能处理动态形状吗？
A: XLA 传统上偏好**静态形状**。JAX 有有限的动态形状支持，但最佳实践仍是尽量使用静态形状。

### Q: 如何在 XLA 中使用自定义 CUDA 内核？
A: 通过 JAX 的 **Custom Call** 机制，可以调用外部 C++/CUDA 代码。

```python
from jax.lib import xla_client
# 或更现代的 jax.extend.ffi
```

---

## 10. 参考资源

- [XLA 官方文档](https://www.tensorflow.org/xla)
- [JAX 文档 - JIT Compilation](https://jax.readthedocs.io/en/latest/jit-compilation.html)
- [StableHLO GitHub](https://github.com/openxla/stablehlo)
- [OpenXLA 项目](https://openxla.org/)
