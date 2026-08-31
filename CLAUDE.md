# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

An Obsidian-based personal study vault containing university course notes in Chinese. Pure Markdown documentation — no code, no build system, no test runner.

Remote: `git@github.com:lao-xiang-1/study.git`

## Directory layout

Each top-level folder is a separate course:

- `自动控制原理/` — Automatic Control Theory
- `信号与系统/` — Signals and Systems
- `机器学习/` — Machine Learning
- `复变函数基础/` — Complex Analysis Basics
- `高等数学/` — Advanced Mathematics
- `数据结构与算法分析/` — Data Structures & Algorithm Analysis

Image assets live in `<course>/assets/` with Obsidian-style names (`Pasted image YYYYMMDDHHMMSS.png`).
Example notebooks live in `<course>/例题/` subfolders.

## Note formatting conventions

- **Lists**: `-` with 2-space indent for nesting
- **Key terms**: `**加粗**`
- **Code/instructions**: backtick (e.g. `MOV A, #35H`)
- **Math**: LaTeX blocks (`$$...$$`) and inline (`$...$`)
- **Notes/caveats**: `>` blockquotes
- **Images**: relative paths, e.g. `![](微机原理与接口技术/assets/xxx.png)`
- **Tables**: standard Markdown with `:---` or `---` alignment
- No stray slide numbers, no duplicate headings

## Symbol consistency

Reuse canonical symbols and terminology consistently within each course so that notes stay aligned. Examples:

- 自动控制原理: `G(s)`, `C(s)`, `R(s)`, `\omega_n`, `\zeta`, `\Phi(s)`
- 信号与系统: `\omega_0`, `f(t)`, `F(j\omega)`, `a_n`, `b_n`

## Language

Content is in Chinese with English term annotations in parentheses where useful (e.g. "锁存（Latch）", "中断（Interrupt）", "阻尼比（Damping Ratio）"). Canonical symbols and terminology must be consistent across notes within a course.

## Git

- Commit messages: short Chinese or English descriptions
- `.gitignore` excludes `.obsidian/`, `.github/`, `.trash/`, `copilot/`, `temp.md`
