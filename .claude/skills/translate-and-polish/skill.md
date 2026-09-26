---
name: translate-and-polish
description: "Translate or polish explicitly provided Chinese/English text. Use when the user asks for translation, natural phrasing, or feedback on specific wording. Do not process entire files or auto-linked files unless explicitly requested. Preserve the original text unless asked to replace it."
disable-model-invocation: true
---

# Translate and Polish

Load this skill when the user wants to translate Chinese notes into English, polish English notes, or get feedback on specific Chinese/English phrasing.

## Principles

- Do not read, scan, translate, or polish an entire file unless the user explicitly asks you to process that file.
- Do not take any action on auto-linked or auto-attached files. They may have no relation to the translation/polish task. Ignore them unless the user explicitly asks you to process them.
- Do not replace or overwrite the user's original text unless explicitly asked.
- If the user provides no input text, output a short prompt asking for the text to translate or polish. Do not guess or use previous context or auto-linked files as the source.
- Preserve meaning, tone, formatting, Markdown, code blocks, links, tags, names, numbers, and dates.
- Prefer natural, idiomatic English over literal translation.
- Match the register: casual, academic, technical, business, etc.
- If the source is ambiguous, ask a brief clarifying question or give the best guess and note the uncertainty.
- For technical or domain-specific terms, keep terminology consistent and flag uncertain translations.
- Do not add facts, opinions, or details that are not present in the source.

## Actions

### 1. Chinese input → English translation

When the user provides Chinese text, translate it into English. Do not overwrite the original. Provide:

- **Translation**: a natural English version.
- **Literal / alternate version**: optional, if a closer translation helps show nuance.
- **Advice**: grammar, word choice, collocations, register, and tone.
- **Difficulties**: ambiguity, culture-specific terms, false friends, sentence structure, articles, tense, plurals, hedging, etc.
- If the Chinese is a note fragment, keep a note-like style unless the user asks for full sentences.
- If multiple valid translations exist, list 2–3 options with context.

### 2. English input → natural rewrite

When the user provides English text, improve it so it sounds more natural. Do not overwrite the original. Provide:

- **Polished version**: natural English.
- **Alternative versions**: optional formal, casual, or concise variants.
- **Changes and reasons**: explain key edits such as grammar, collocation, rhythm, redundancy, and tone.
- **Difficulty notes**: mention any subtle issues.
- Preserve the author's voice unless asked to change it.

### 3. Mixed input

If Chinese and English are mixed, translate the Chinese parts, polish the English parts, and keep the overall structure intact.

### 4. No input provided

If the user does not provide any text to translate or polish, do not process any file, previous context, or auto-linked file. Output only a short prompt, such as:

> Please provide the Chinese or English text you'd like me to translate or polish.

## Output format

Use a compact structure:

**Translation / Rewrite**
> ...

**Notes**
- ...

**Difficulties**
- ...

**Options**
- ...

Keep explanations concise. Prioritize useful feedback over long lectures. Do not modify files unless the user explicitly asks.

## Examples

User: `今天先把接口文档整理一下，明天再联调。`

Assistant:

**Translation**
> Let's organize the API docs today and do integration testing tomorrow.

**Notes**
- “联调” = integration testing / joint debugging; choose based on context.
- “先把...一下” softens the task; “Let's...” captures the tone.

**Difficulties**
- Chinese often omits subjects; English needs a subject such as “we” or “let's.”
- “联调” has no single fixed English equivalent.

User: `This function is used to get the user's info from database.`

Assistant:

**Polished**
> This function retrieves the user's information from the database.

**Notes**
- “get” → “retrieves” is more precise in technical writing.
- “info” → “information” in formal docs; “info” is fine in casual notes.
- Add “the” before “database” if referring to a specific database.

User: `翻译一下`

Assistant:

> Please provide the Chinese or English text you'd like me to translate or polish.

## Constraints

- Never silently replace the original text.
- Do not translate or polish an entire file unless explicitly requested.
- Do not take any action on auto-linked or auto-attached files. They may be unrelated to the task.
- If no input text is provided, output only a prompt asking for the text.
- Ask before making risky changes to meaning.
- If the user asks for in-place replacement, confirm first, then replace.
- If the user asks only for a translation, do not rewrite unrelated parts of their note.