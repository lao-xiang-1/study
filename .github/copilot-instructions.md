# Repository Instructions for Copilot

## Build, test, and lint

This repository does not define build, test, or lint tooling at the root or in subdirectories (no `package.json`, `pyproject.toml`, `requirements.txt`, `Cargo.toml`, `go.mod`, `.sln`, `.csproj`, `pom.xml`, `build.gradle`, or `Makefile` were found).

Use Markdown/Obsidian validation workflows if added later, but do not assume a test runner exists.

## High-level architecture

This repository is an Obsidian-style study vault rather than an application codebase.

- Top-level `.obsidian/` stores workspace/editor configuration (for example, `app.json` and `core-plugins.json`).
- Domain content is organized under `自动控制原理/` as focused topic notes.
- The knowledge structure is layered:
  - `课程学习路线.md` is the macro map of the course (modeling, analysis methods, design, and extensions).
  - `易忘知识点.md` is a compact formula/definition recall sheet.
  - `二阶系统的时域分析法.md` and `复变函数基础知识.md` are deep-dive topic notes that expand specific areas from the roadmap.

When adding new notes, preserve this layering: roadmap-level summary, concise quick-reference notes, and detailed single-topic expansions.

## Key conventions in this repository

- Language and style: content is primarily in Chinese, with standard control-theory terminology and common English term annotations in parentheses when useful.
- Math representation: formulas are written in LaTeX math blocks/inline math (`$$...$$`, `$...$`).
- Structure pattern for topic notes:
  - Start with a clear H1 title.
  - Use numbered sections (`## 1.`, `## 2.`) for major concepts.
  - Use bullet lists for definitions/comparisons and include explicit symbols/variables.
- Cross-note consistency: reuse canonical symbols and names consistently (`G(s)`, `C(s)`, `R(s)`, `\omega_n`, `\zeta`, etc.) so quick-reference and deep-dive notes stay aligned.
