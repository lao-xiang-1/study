---
sr-due: 2026-11-06
sr-interval: 48
sr-ease: 250
---
#code 

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

## 附加
*Ruff* 是由 Astral 团队开发的一个现代 **Python 代码静态分析工具**（Linter）和**格式化工具**（Formatter）。**用rust编写**

*mypy* 是一个针对 Python 的开源**静态类型检查器**（Static Type Checker）。它通过分析 Python 代码中的类型提示（Type Hints），在不实际运行代码的情况下，帮助开发者发现潜在的类型错误和逻辑漏洞。