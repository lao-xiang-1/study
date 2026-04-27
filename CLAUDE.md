# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

An Obsidian-based personal study vault containing university course notes in Chinese. Pure Markdown documentation — no code, no build system, no test runner.

Remote: `git@github.com:lao-xiang-1/study.git`

## Directory layout

Each top-level folder is a separate course:

- `自动控制原理/` — Automatic Control Theory
- `微机原理与接口技术/` — Microcomputer Principles & Interface Technology
- `信号与系统/` — Signals and Systems
- `电路与电子学/` — Circuit and Electronics
- `机器学习/` — Machine Learning

Image assets live in `<course>/assets/` with Obsidian-style names (`Pasted image YYYYMMDDHHMMSS.png`).

## Note formatting conventions

- **Top-level heading**: `# 第X章 主题名` or `# 第X讲 主题名`
- **Sections**: `## 2.X 节名`, separated by `---`
- **Subsections**: `### 2.X.X 小节名`
- **Sub-sub items**: `**（1）标题**` — Chinese parenthesized bold, not `####`
- **Lists**: `-` with 2-space indent for nesting
- **Key terms**: `**加粗**`
- **Code/instructions**: backtick (e.g. `MOV A, #35H`)
- **Math**: LaTeX blocks (`$$...$$`) and inline (`$...$`)
- **Notes/caveats**: `>` blockquotes
- **Images**: relative paths, e.g. `![](微机原理与接口技术/assets/xxx.png)`
- **Tables**: standard Markdown with `:---` or `---` alignment
- No stray slide numbers, no duplicate headings

## Note layering pattern

Courses follow a three-tier structure — preserve it when adding content:

1. **Roadmap/overview** — macro map of the course (e.g. `课程学习路线.md`)
2. **Quick-reference** — compact formula/definition recall sheet (e.g. `易忘知识点.md`, `常见指令速查.md`)
3. **Deep-dive topics** — detailed single-topic expansions (e.g. `二阶系统的时域分析法.md`, `IO接口扩展及应用.md`)

## Language

Content is in Chinese with English term annotations in parentheses where useful (e.g. "锁存（Latch）", "中断（Interrupt）"). Canonical symbols and terminology must be consistent across notes within a course.

## Git

- Commit messages: short Chinese or English descriptions
- `.gitignore` excludes `.obsidian/`, `.github/`, `.trash/`, `copilot/`, `temp.md`, `整理笔记.md`
