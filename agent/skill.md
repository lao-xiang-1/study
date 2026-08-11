# AI Skill（技能）

> Skill（技能）是为大语言模型（LLM）或 Agent 编写的结构化指令文档，用于在特定场景下复用经验、规范行为、降低重复提示成本。Claude Code 中的 Skill 常被称作 **Superpowers**。

---

## 1. 什么是 Skill

### 1.1 核心定义

- **Skill** 是一段**可复用的提示词（Prompt）模板**。
- 它告诉模型：在什么场景下、以什么流程、用什么标准去完成任务。
-  Skill 通常以 Markdown 文件形式存在，带有 YAML 元数据（Frontmatter）。

### 1.2 解决的问题

- 避免每次都要重复输入完整指令。
- 让复杂任务有**固定、可审计**的执行流程。
- 方便团队协作：把最佳实践沉淀为可共享文件。

---

## 2. Skill 的典型结构

一个 Skill 文件通常由两部分组成：

### 2.1 YAML Frontmatter（元数据）

```yaml
---
name: quiz
description: 根据笔记内容生成交互式问答测验，逐个提问并评判回答
disable-model-invocation: true
args:
  - name: note_path
    description: 笔记文件的路径（支持相对路径和绝对路径）
    required: true
---
```

常见字段说明：

| 字段 | 含义 |
| --- | --- |
| `name` | Skill 的唯一标识名。 |
| `description` | 一句话说明 Skill 的用途，用于匹配用户请求。 |
| `disable-model-invocation` | 是否禁止该 Skill 内部再调用模型。 |
| `args` | Skill 接受的参数列表。 |

### 2.2 Markdown 正文（指令体）

- 用 `#` 标题说明 Skill 是什么。
- 用 `##` 划分执行阶段或模块。
- 用列表、表格、代码块清晰表达规则。
- 用 `>` 引用块强调关键约束。

---

## 3. Skill 的两种风格

根据约束强度，Skill 可分为两类：

### 3.1 刚性 Skill（Rigid）

- **特点**：必须严格按步骤执行，不能跳过或自由发挥。
- **适用**：TDD、调试、代码审查、安全检查等需要纪律的场景。
- **示例**：`superpowers:test-driven-development`、`superpowers:systematic-debugging`。

### 3.2 柔性 Skill（Flexible）

- **特点**：提供原则和框架，允许根据上下文调整。
- **适用**：设计、写作、头脑风暴等需要创造性的任务。
- **示例**：`frontend-design:frontend-design`、`superpowers:brainstorming`。

---

## 4. Skill 如何被触发

### 4.1 自动触发

- 平台根据用户请求与 Skill `description` 的匹配度自动选择。
- 例如用户说“帮我写个前端页面”，可能触发 `frontend-design` Skill。

### 4.2 显式调用

- 用户直接输入 Skill 名，如 `/quiz` 或 `/translate-paper`。
- 平台加载对应 Skill 文件，将其内容作为系统上下文附加到当前对话。

### 4.3 强制前置

某些 Skill 要求“在回应前先调用”，例如 `superpowers:using-superpowers` 会在每次会话开始时提醒模型检查可用 Skill。

---

## 5. Skill 与系统提示词的关系

| 对比项 | 系统提示词（System Prompt） | Skill |
| --- | --- | --- |
| 作用范围 | 整个会话全局生效 | 按需加载，局部生效 |
| 更新方式 | 通常硬编码在平台/应用里 | 以文件形式存在，可独立维护 |
| 可复用性 | 低，每个应用单独配置 | 高，一个 Skill 可被多个项目引用 |
| 粒度 | 宏观行为约束 | 针对特定任务的专业流程 |

> Skill 可以看作系统提示词的**模块化、文件化、可插拔**版本。

---

## 6. 写好 Skill 的原则

### 6.1 触发条件清晰

- `description` 要准确描述适用场景，避免误触发。
- 使用明确的关键词，例如“TDD”、“debug”、“review”。

### 6.2 流程可执行

- 步骤要具体到模型能直接照做。
- 每一步说明输入、输出、判断标准。

### 6.3 约束明确

- 哪些必须做，哪些不能做，用列表或引用块标出。
- 例如：“**每次只翻译一段**”、“**不要跳过测试**”。

### 6.4 示例具体

- 提供输入/输出示例，减少模型理解偏差。
- 示例要贴近真实使用场景。

### 6.5 避免过度冗长

- Skill 太长会增加上下文负担。
- 把通用知识交给模型，Skill 只放**差异化指令**。

---

## 7. 本仓库中的 Skill 示例

### 7.1 Quiz Skill

- 路径：`.claude/skills/quiz/SKILL.md`
- 用途：根据笔记内容生成测验题目并逐题提问。
- 特点：
  - 接受 `note_path` 参数。
  - 分“阅读分析 → 开场 → 逐题提问 → 总结”四步。
  - 强调**每次只问一题**、**灵活评判**。

### 7.2 Translate Paper Skill

- 路径：`.claude/skills/translate-paper/skill.md`
- 用途：将 OCR 提取的英文学术论文逐段翻译为中文。
- 特点：
  - 先清理无断空格（NBSP）避免 Edit 失败。
  - **每次只翻译一段**，原文不动，译文追加。
  - 保留引用标记和数学公式。

---

## 8. 小结

- **Skill** 是 LLM/Agent 的复用型指令模块。
- 结构为 **YAML Frontmatter + Markdown 指令体**。
- 分为**刚性**（严格执行）和**柔性**（原则指导）两类。
- 触发方式包括自动匹配、显式调用、强制前置。
- 写好 Skill 的关键是：**触发条件清晰、流程可执行、约束明确、示例具体、避免冗长**。
