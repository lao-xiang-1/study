
## document
*官方网址*：[uv-dependency](https://uv.doczh.com/concepts/projects/dependencies/#_9).

## example
```toml
# uv-specific configuration: use CUDA 12.6 wheels for torch/torchvision on Windows/Linux
[tool.uv.sources]
torch = [
    { index = "pytorch-cu126", marker = "sys_platform == 'win32' or sys_platform == 'linux'" },
]
torchvision = [
    { index = "pytorch-cu126", marker = "sys_platform == 'win32' or sys_platform == 'linux'" },
]

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

# 换源（默认使用清华源）
[[tool.uv.index]]
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true
```

## 依赖源

`tool.uv.sources` 表用**替代依赖源**（Alternative Dependency Sources）扩展标准依赖表，用于开发过程。它为 `project.dependencies` 标准不支持的常见模式提供支持，例如**可编辑安装**（Editable Install）和**相对路径**。例如，从相对于项目根目录的目录安装 `foo`：

```toml
[project]
name = "example"
version = "0.1.0"
dependencies = ["foo"]

[tool.uv.sources]
foo = { path = "./packages/foo" }
```

uv 支持以下依赖源：

- **索引（Index）**：从特定包索引解析的包。
- **Git**：一个 Git 仓库。
- **URL**：远程 wheel 或源分发包。
- **路径（Path）**：本地 wheel、源分发包或项目目录。
- **工作区成员（Workspace Member）**：当前工作区的成员。

> 源仅对 uv 有效。使用其他工具时只会读取标准项目表中的定义；若用其他工具开发，需按其格式重新指定源表中提供的任何元数据。

这些源都可以选择通过命令行直接安装，或者修改pyproject.toml

#code/python
### 索引（Index）
?
要从特定索引添加包，使用 `--index` 选项：

```bash
uv add torch --index pytorch=https://download.pytorch.org/whl/cpu
```

uv 会将索引存入 `[[tool.uv.index]]`，并添加一个 `[tool.uv.sources]` 条目：

```toml
[project]
dependencies = ["torch"]

[tool.uv.sources]
torch = { index = "pytorch" }

[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cpu"
```

- 使用 `index` 源会把包**固定**到给定索引，不会从其他索引下载。
- `explicit = true` 表示该索引**仅**应用于在 `tool.uv.sources` 中显式指定它的包；若未设置，其他包在别处找不到时可能从该索引解析。

> 上例仅适用于 x86-64 Linux（PyTorch 索引的特殊性）。本文档开头 `[tool.uv.sources]` 中的 torch/torchvision 配置即属于「索引源 + 特定平台」的组合用法。
<!--SR:!2026-11-09,3,250-->

---

### Git

在兼容 Git 的 URL（即 `git clone` 所用 URL）前加上 `git+` 前缀即可：

```bash
uv add git+https://github.com/encode/httpx
```

```toml
[project]
dependencies = ["httpx"]

[tool.uv.sources]
httpx = { git = "https://github.com/encode/httpx" }
```

可指定具体引用：

- 标签：`--tag 0.27.0` → `tag = "0.27.0"`
- 分支：`--branch main` → `branch = "main"`
- 版本（提交）：`--rev <commit>` → `rev = "<commit>"`
- 子目录（包不在仓库根目录）：`#subdirectory=libs/langchain` → `subdirectory = "libs/langchain"`

### URL

提供指向 wheel 文件（`.whl`）或源分发包（通常 `.tar.gz` 或 `.zip`）的 `https://` URL：

```toml
[project]
dependencies = ["httpx"]

[tool.uv.sources]
httpx = { url = "https://files.pythonhosted.org/packages/.../httpx-0.27.0.tar.gz" }
```

### 路径（Path）

提供 wheel、源分发包，或包含 `pyproject.toml` 的目录的路径（可为相对路径）：

```bash
uv add ./foo-0.1.0-py3-none-any.whl
uv add ~/projects/bar/
```

```toml
[tool.uv.sources]
foo = { path = "/example/foo-0.1.0-py3-none-any.whl" }
```

> 默认路径依赖**不使用**可编辑安装。对项目目录可请求可编辑安装：`uv add --editable ../projects/bar/`（生成 `editable = true`）。若项目被标记为**非包**却希望作为包安装，则设置 `package = true`。同一仓库中的多个包，工作区（Workspace）可能更合适。

### 特定平台的源

通过为源提供与依赖说明符兼容的**环境标记**（Environment Marker），可将源限制在特定平台或 Python 版本：

```toml
[tool.uv.sources]
httpx = { git = "https://github.com/encode/httpx", tag = "0.27.2", marker = "sys_platform == 'darwin'" }
```

> 上例中 uv 仍会在所有平台包含 `httpx`，但只在 macOS 上从 GitHub 下载源，其他平台回退到 PyPI。


## 附加
*Ruff* 是由 Astral 团队开发的一个现代 **Python 代码静态分析工具**（Linter）和**格式化工具**（Formatter）。**用rust编写**

*mypy* 是一个针对 Python 的开源**静态类型检查器**（Static Type Checker）。它通过分析 Python 代码中的类型提示（Type Hints），在不实际运行代码的情况下，帮助开发者发现潜在的类型错误和逻辑漏洞。

*pytest* 是 Python 生态中最主流的**单元测试框架**（Testing Framework）。它按约定自动发现并运行测试（文件 `test_*.py`、函数 `test_*`），直接用 `assert` 断言并智能打印失败上下文；通过 `@pytest.fixture` 管理测试前后置资源，用 `@pytest.mark.parametrize` 做参数化批量测试，并拥有 `pytest-cov`（覆盖率）、`pytest-xdist`（并行）、`pytest-asyncio`（异步）等丰富插件。相比标准库 `unittest` 更简洁，是社区事实上的标准。
