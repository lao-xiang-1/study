# 第7章负反馈放大电路笔记整理 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将当前 OCR 生成的《第7章 负反馈放大电路》Markdown 笔记整理成层级清晰、便于阅读和复习的 Obsidian 笔记。

**Architecture:** 这是纯 Markdown 笔记整理任务，不涉及代码、构建系统或测试框架。实施方式是完整重写单个当前笔记文件的正文结构，保留原知识内容和远程图片 HTML 链接，只调整标题层级、段落组织、列表、公式、例题、小结和习题格式。

**Tech Stack:** Obsidian Markdown、LaTeX 数学公式、HTML 图片标签、Git diff 人工核对。

---

## File Structure

**Modify:** `第7章 负反馈放大电路.pdf_by_PaddleOCR-VL-1.6.md`

Responsibilities:
- 承载第 7 章“负反馈放大电路”的完整学习内容。
- 使用 vault 约定的标题层级：`#` 章标题，`##` 一级节，`###` 小节，`**（1）...**` 表示小项。
- 保留 OCR 生成的远程图片 HTML 链接，避免因图片下载或链接替换导致内容丢失。
- 不添加教材外扩写，只修正明显格式问题和 OCR 造成的阅读障碍。

**No new content files required.**

---

### Task 1: Normalize Top-Level Structure

**Files:**
- Modify: `第7章 负反馈放大电路.pdf_by_PaddleOCR-VL-1.6.md:1-13`

- [ ] **Step 1: Read the current note**

Use the filesystem read tool on:

```text
D:\docs\study\第7章 负反馈放大电路.pdf_by_PaddleOCR-VL-1.6.md
```

Expected: file begins with `### 第7章 负反馈放大电路` and contains OCR-style mixed heading levels.

- [ ] **Step 2: Change the title and requirements section**

Replace the opening structure with:

```markdown
# 第7章 负反馈放大电路

## 本章基本要求

- **会判**：判断电路中有无反馈及反馈的性质。
- **会算**：估算深度负反馈条件下的放大倍数。
- **会引**：根据需求引入合适的反馈。

---

## 7.1 反馈的基本概念与分类

### 7.1.1 反馈的基本概念
```

- [ ] **Step 3: Verify heading hierarchy after this section**

Check that the first headings are exactly:

```markdown
# 第7章 负反馈放大电路
## 本章基本要求
## 7.1 反馈的基本概念与分类
### 7.1.1 反馈的基本概念
```

Expected: no `### 第7章` or `#### 本章基本要求` remains.

---

### Task 2: Clean Up Basic Concepts Section

**Files:**
- Modify: `第7章 负反馈放大电路.pdf_by_PaddleOCR-VL-1.6.md:15-36`

- [ ] **Step 1: Convert definitions to bold key terms**

Rewrite the section content as:

```markdown
**（1）反馈的定义**

**反馈**：放大电路输出量的一部分或全部通过一定的方式引回到输入回路，影响输入，称为反馈。

**（2）反馈放大电路的组成**

反馈放大电路分为：

- **基本放大电路**
- **反馈网**

[keep original image HTML for feedback amplifier composition]

**（3）开环与闭环**

- **反馈通路**：信号反向传输的通道。
- **开环**：无反馈通路，信号只从输入端传输到输出端。
- **闭环**：有反馈通路，信号传输形成闭环。

**（4）有无反馈的判断**

**判别方法**：找输出回路与输入回路之间的联系；若输出量能通过某种通路影响输入回路，则存在反馈。

[keep original two image HTML blocks for feedback existence examples]
```

- [ ] **Step 2: Preserve all image HTML blocks in this section**

Keep the three original `<div style="text-align: center;"><img ... /></div>` blocks from the original basic concepts section.

Expected: image URLs are unchanged byte-for-byte unless surrounding blank lines are adjusted.

- [ ] **Step 3: Verify no knowledge content was removed**

Confirm these phrases still exist:

```text
放大电路输出量的一部分或全部
反馈通路
开环
闭环
找输出回路与输入回路
```

---

### Task 3: Clean Up Feedback Polarity Section

**Files:**
- Modify: `第7章 负反馈放大电路.pdf_by_PaddleOCR-VL-1.6.md:38-102`

- [ ] **Step 1: Normalize heading and definitions**

Use this structure:

```markdown
### 7.1.2 反馈的分类

**（1）按反馈极性分类：正反馈和负反馈**

- **正反馈**：使放大电路净输入量增大的反馈。
- **负反馈**：使放大电路净输入量减小的反馈。

[keep original image HTML for positive/negative feedback]
```

- [ ] **Step 2: Format instantaneous polarity method**

Replace the loose explanation with:

```markdown
**判别方法：瞬时极性法**

核心是“看反馈的结果”：判断净输入量是被增大还是被减小。

步骤：

1. 假设某一瞬时，在放大电路输入端加入一个正极性的输入信号。
2. 按信号传输方向依次判断各处电流、电位的瞬时极性。
3. 判断反馈信号的瞬时极性。
4. 若反馈信号使净输入信号减小，则为**负反馈**；反之，为**正反馈**。
```

- [ ] **Step 3: Normalize formulas**

Format the original formulas as:

```markdown
$$
\dot{X}_{\mathrm{o}} \rightarrow \dot{X}_{\mathrm{f}} \rightarrow \dot{X}_{\mathrm{i}},\ \dot{X}_{\mathrm{f}},\ \dot{X}_{\mathrm{id}}
$$

若：

$$
\dot{U}_{\mathrm{id}}=\dot{U}_{\mathrm{i}}-\dot{U}_{\mathrm{f}}
$$

或：

$$
\dot{I}_{\mathrm{id}}=\dot{I}_{\mathrm{i}}-\dot{I}_{\mathrm{f}}
$$

则为**负反馈**。

若：

$$
\dot{U}_{\mathrm{id}}=\dot{U}_{\mathrm{i}}+\dot{U}_{\mathrm{f}}
$$

或：

$$
\dot{I}_{\mathrm{id}}=\dot{I}_{\mathrm{i}}+\dot{I}_{\mathrm{f}}
$$

则为**正反馈**。
```

- [ ] **Step 4: Convert polarity examples to consistent captions**

For each image/result pair in the original polarity examples, keep the image and format the result as:

```markdown
> 结论：负反馈。
```

or:

```markdown
> 结论：正反馈。
```

Expected: the original conclusions `负反馈` and `正反馈` are preserved, but no nested `<div><div>结论</div></div>` wrappers remain.

---

### Task 4: Clean Up AC/DC Feedback Section

**Files:**
- Modify: `第7章 负反馈放大电路.pdf_by_PaddleOCR-VL-1.6.md:105-150`

- [ ] **Step 1: Normalize heading and definitions**

Use this structure:

```markdown
**（2）按反馈的交直流性质分类：直流反馈、交流反馈和交直流反馈**

- **直流反馈**：反馈信号中只含直流成分。直流负反馈用于稳定静态工作点，对放大电路的动态性能没有影响。
- **交流反馈**：反馈信号中只含交流成分。交流负反馈用于改善放大电路的各项动态性能指标。
- **交直流反馈**：反馈信号中同时包含交、直流成分。

**判别方法：电容观察法**。
```

- [ ] **Step 2: Normalize examples**

Keep every original image HTML block in the same order and change the text results to blockquotes:

```markdown
> 结论：直流反馈。
> 结论：交流反馈。
> 结论：交直流反馈。
> 结论：仅有直流反馈。
> 结论：交、直流反馈共存。
> 结论：仅有直流反馈。
> 结论：仅有交流反馈。
```

- [ ] **Step 3: Preserve the capacitor assumption sentence**

Keep this sentence near the second group of examples:

```markdown
例：设以下电路中所有电容对交流信号均可视为短路。
```

Expected: the assumption is not moved away from the associated example images.

---

### Task 5: Clean Up Series/Parallel and Voltage/Current Feedback Sections

**Files:**
- Modify: `第7章 负反馈放大电路.pdf_by_PaddleOCR-VL-1.6.md:152-200`

- [ ] **Step 1: Convert series/parallel section from accidental H2 to subsection item**

Replace the original long heading:

```markdown
## 根据反馈信号与输入信号在放大电路输入回路中求和形式的不同，可分为：串联反馈和并联反馈
```

with:

```markdown
**（3）按输入端求和方式分类：串联反馈和并联反馈**
```

- [ ] **Step 2: Format series/parallel definitions**

Use:

```markdown
串联反馈和并联反馈描述放大电路和反馈网络在输入端的连接方式，即输入量和反馈量之间的叠加关系。

- 若 $\dot{X}_{\mathrm{f}}$ 为电压信号，则形成**串联反馈**。
- 若 $\dot{X}_{\mathrm{f}}$ 为电流信号，则形成**并联反馈**。

**判别方法**：看反馈信号和输入信号是否在同一端；若二者在同一端，则为**并联反馈**，否则为**串联反馈**。
```

- [ ] **Step 3: Preserve series/parallel example images**

Keep the original three image HTML blocks from the series/parallel example area in their original order.

- [ ] **Step 4: Format voltage/current feedback section**

Use:

```markdown
**（4）按输出端取样对象分类：电压反馈和电流反馈**

电流反馈和电压反馈描述放大电路和反馈网络在输出端的连接方式，即反馈网络的取样对象。

- **电流反馈**：将输出电流的一部分或全部引回到输入回路来影响净输入量，即 $\dot{X}_{\mathrm{o}}=\dot{I}_{\mathrm{o}}$。
- **电压反馈**：将输出电压的一部分或全部引回到输入回路来影响净输入量，即 $\dot{X}_{\mathrm{o}}=\dot{U}_{\mathrm{o}}$。

**判别方法：输出端短路法**。将输出端短路，即令 $u_{\mathrm{o}}=0$：

- 若反馈信号消失，则为**电压反馈**。
- 若反馈信号仍然存在，则为**电流反馈**。
```

Expected: OCR symbols `X_0` and `I_0` are normalized to output subscript `\mathrm{o}` for consistency.

---

### Task 6: Clean Up Local/Interstage Feedback and Summary

**Files:**
- Modify: `第7章 负反馈放大电路.pdf_by_PaddleOCR-VL-1.6.md:203-230`

- [ ] **Step 1: Format local/interstage section**

Use:

```markdown
**（5）按作用范围分类：局部反馈和级间反馈**

- **局部反馈**：只对多级放大电路中某一级起反馈作用。
- **级间反馈**：将多级放大电路的输出量引回到其输入级的输入回路。

[keep original image HTML]

图中：

- 通过 $R_{3}$ 引入的是**局部反馈**。
- 通过 $R_{4}$ 引入的是**级间反馈**。

通常重点研究级间反馈，或称**总体反馈**。
```

- [ ] **Step 2: Convert summary heading and bullets**

Use:

```markdown
---

## 小结

- **判别反馈的有无**：看是否形成闭环，以及反馈量能否影响净输入量。
- **正、负反馈判别**：瞬时极性法。
- **交、直流反馈判别**：电容观察法。
- **串、并联反馈判别**：看输入信号和反馈信号是否在同一端。
- **电压、电流反馈判别**：输出端短路法。
- **多级放大电路的反馈**：一般重点判断级间反馈。
```

Expected: original diamond bullets `♦` are replaced with standard Markdown bullets.

---

### Task 7: Clean Up Exercises Section

**Files:**
- Modify: `第7章 负反馈放大电路.pdf_by_PaddleOCR-VL-1.6.md:232-284`

- [ ] **Step 1: Add exercises heading**

Before the first question, add:

```markdown
---

## 习题
```

- [ ] **Step 2: Convert each question to numbered subsection**

Use this exact format for the existing questions:

```markdown
### 1. 单选题（2 分）

对于放大电路，所谓开环是指：

A. 无信号源  
B. 无反馈通路  
C. 无电源  
D. 无负载

### 2. 单选题（2 分）

对于放大电路，所谓闭环是指：

A. 考虑信号源内阻  
B. 接入电源  
C. 接入负载  
D. 存在反馈通路

### 3. 填空题（4 分）

1. 反馈放大电路是一个由基本放大电路和 [填空1] 构成的闭合回路。
2. 在放大电路中，为了稳定静态工作点，可以引入 [填空2] 负反馈。

### 4. 单选题（2 分）

构成反馈通路的元器件：

A. 只能是三极管、集成运放等有源器件  
B. 只能是电阻  
C. 只能是无源器件  
D. 可以是无源器件也可以是有源器件

### 5. 单选题（2 分）

直流负反馈是指：

A. 直接耦合放大电路中所引入的负反馈  
B. 放大直流信号时才有的负反馈  
C. 在直流通路中的负反馈  
D. 只存在于阻容耦合电路中的反馈
```

- [ ] **Step 3: Do not add answers**

Expected: the exercises remain questions only; no answer key is introduced.

---

### Task 8: Final Markdown Verification

**Files:**
- Verify: `第7章 负反馈放大电路.pdf_by_PaddleOCR-VL-1.6.md`

- [ ] **Step 1: Check heading outline**

Expected outline:

```markdown
# 第7章 负反馈放大电路
## 本章基本要求
## 7.1 反馈的基本概念与分类
### 7.1.1 反馈的基本概念
### 7.1.2 反馈的分类
## 小结
## 习题
### 1. 单选题（2 分）
### 2. 单选题（2 分）
### 3. 填空题（4 分）
### 4. 单选题（2 分）
### 5. 单选题（2 分）
```

- [ ] **Step 2: Check image preservation**

Confirm the number of `<img src="https://pplines-online...` references after editing matches the original note count.

Expected: no image block is deleted.

- [ ] **Step 3: Check OCR artifact cleanup**

Confirm the file no longer contains:

```text
<div style="text-align: center;"><div style="text-align: center;">
##### 单选题 2分
##### 填空题 4分
## 根据反馈信号与输入信号在放大电路输入回路中求和形式的不同
### 第7章 负反馈放大电路
```

- [ ] **Step 4: Review diff before reporting completion**

Run:

```bash
git diff -- "第7章 负反馈放大电路.pdf_by_PaddleOCR-VL-1.6.md"
```

Expected:
- Diff shows Markdown restructuring only.
- No unrelated files are modified.
- No answer key or new textbook explanations are added.

---

## Self-Review

**Spec coverage:**
- Deep整理: covered by Tasks 1-7.
- Chapter hierarchy: covered by Tasks 1, 3, 5, 6, 8.
- Lists/formulas/examples/summary/exercises: covered by Tasks 2-7.
- Preserve image links: covered by Tasks 2-6 and Task 8 Step 2.
- No extra expansion: covered by File Structure and Task 7 Step 3.

**Placeholder scan:** No unresolved placeholders are present. Bracketed phrases such as `[填空1]` are original exercise blanks and must remain.

**Consistency check:** Heading levels and terminology follow the vault conventions in `CLAUDE.md`: `#` chapter title, `##` sections, `###` subsections, bold Chinese parenthesized sub-items, standard Markdown lists, LaTeX math notation.
