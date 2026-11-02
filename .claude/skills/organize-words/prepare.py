#!/usr/bin/env python3
"""为 organize-words skill 准备输入数据。

读取 words.txt 的最后一个 ===日期=== 批次、words.csv 的释义、lexical-chunk.md
的已整理单词集合、现有分组结构与最大组号，输出 JSON 供后续 LLM 分组与排版。

路径分三处（脚本已从项目根搬进 skill 目录，故不能再用 cwd 或单一 root）：
  - words.txt / words.csv / 草稿 与脚本同在 skill 目录（SKILL_DIR）；
  - 词块笔记的位置由同目录下 config.json 的 notes_path 指定（相对项目根）；
  - 项目根由 find_project_root() 从 SKILL_DIR 向上定位。

释义经 condense_definition 精简成「词性 + 主要义项」短形式，JSON 以紧凑格式单行输出——
这两处都是为了压住每次运行都要重读的 payload 体积（约 90KB → 37KB）。

用法（从任意目录运行均可）：
    uv run python .claude/skills/organize-words/prepare.py
"""
import csv
import json
import re
import sys
from pathlib import Path

# Windows 默认 stdout 可能是 GBK，强制 UTF-8 保证中文释义不乱码
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

CN_DIGITS = "零一二三四五六七八九"


SKILL_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SKILL_DIR / "config.json"
DEFAULT_CONFIG = {
    "notes_path": "lexical-chunk.md",
    "draft_name": "lexical-chunk-draft.md",
}


def load_config() -> dict:
    """读取 skill 目录下的 config.json，缺省或字段缺失时回落到 DEFAULT_CONFIG。

    两个键的基准不同，别混：
      - notes_path：相对**项目根**，换项目时笔记位置不一样，改这里即可；
      - draft_name：只是**文件名**，草稿固定放在 skill 目录内（用完即删，
                放在这里才能被 skill 自带的 .gitignore 覆盖）。
    只接受 DEFAULT_CONFIG 里已知的键，多余的忽略，避免旧配置残留字段。
    """
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cfg.update({k: v for k, v in raw.items() if k in DEFAULT_CONFIG})
    return cfg


def _rel_to(path: Path, base: Path) -> str:
    """尽量返回 path 相对 base 的 POSIX 字符串；不在 base 之下时退回绝对路径。

    这样 payload 里的路径都能直接从项目根使用（git 命令、Read/Edit 都认）。
    """
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def find_project_root(notes_rel: str) -> Path:
    """从本脚本位置向上找项目根目录。

    判据是 pyproject.toml / .git / 词块笔记自身（笔记未必在项目根，所以要用
    config 里的相对路径去拼）。**不能用 words.txt 当判据**：它就在 skill 目录里，
    向上查找第一步就会命中，会误把 skill 目录当成项目根，导致读不到现有分组、
    新增组号从「一」重来。
    """
    p = SKILL_DIR
    for _ in range(8):
        if (
            (p / "pyproject.toml").exists()
            or (p / ".git").exists()
            or (p / notes_rel).exists()
        ):
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path.cwd()


def num_to_cn(n: int) -> str:
    """1-999 转中文数字（37 -> 三十七，100 -> 一百，105 -> 一百零五）。"""
    if n <= 0 or n > 999:
        return str(n)
    if n < 10:
        return CN_DIGITS[n]
    if n < 20:
        return "十" + (CN_DIGITS[n - 10] if n > 10 else "")
    if n < 100:
        t, o = divmod(n, 10)
        return CN_DIGITS[t] + "十" + (CN_DIGITS[o] if o else "")
    h, rem = divmod(n, 100)
    out = CN_DIGITS[h] + "百"
    if rem == 0:
        return out
    if rem < 10:
        return out + "零" + CN_DIGITS[rem]
    if rem < 20:
        return out + "一十" + (CN_DIGITS[rem - 10] if rem > 10 else "")
    t, o = divmod(rem, 10)
    return out + CN_DIGITS[t] + "十" + (CN_DIGITS[o] if o else "")


def cn_to_num(s: str):
    """中文数字转 int（三十六 -> 36）。无法解析返回 None。"""
    s = s.strip()
    if not s or any(c not in CN_DIGITS + "十百零" for c in s):
        return None
    total = 0
    if "百" in s:
        i = s.index("百")
        total += (CN_DIGITS.index(s[i - 1]) if i > 0 and s[i - 1] in CN_DIGITS else 1) * 100
        s = s[i + 1:]
    if "十" in s:
        i = s.index("十")
        total += (CN_DIGITS.index(s[i - 1]) if i > 0 and s[i - 1] in CN_DIGITS else 1) * 10
        s = s[i + 1:]
    s = s.replace("零", "")
    if s:
        if s in CN_DIGITS:
            total += CN_DIGITS.index(s)
        else:
            return None
    return total or None


# 有道释义里的噪音：<英>/<旧>/<非正式> 等语域标签、（…）补充说明、【名】…（人名）整行
_TAG_RE = re.compile(r"<[^>]{1,12}>")
_PAREN_RE = re.compile(r"[（(][^）)]{0,60}[）)]")
_POS_RE = re.compile(r"^([a-z]{1,5}\.)\s*(.*)$")
_NOISE_RE = re.compile(r"^【[^】]*】")
_TRIM = "；;，,、。.：: \t"


def condense_definition(raw: str, max_pos: int = 3, max_senses: int = 2, cap: int = 80) -> str:
    """把有道原始长释义精简成「词性 + 主要义项」短形式。

    去掉人名等噪音行、语域标签与括号补充说明，每个词性只保留前 max_senses 个义项，
    最多保留 max_pos 个词性，总长不超过 cap 个字符。
    """
    if not raw or not raw.strip():
        return ""

    blocks: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or _NOISE_RE.match(line):
            continue
        line = _TAG_RE.sub("", line)
        line = _PAREN_RE.sub("", line)
        line = re.sub(r"\s+", " ", line).strip(_TRIM)
        if len(line) < 2:
            continue

        m = _POS_RE.match(line)
        pos, body = (m.group(1), m.group(2)) if m else ("", line)
        body = re.split(r"[：:]", body, maxsplit=1)[0].strip(_TRIM)
        senses = [s.strip(_TRIM) for s in re.split(r"[；;]", body)]
        senses = [s for s in senses if s][:max_senses]
        if not senses:
            continue
        if pos:
            blocks.append(f"{pos} " + "，".join(senses))
        else:
            blocks.append("；".join(senses))

    out = "；".join(blocks[:max_pos])
    if len(out) <= cap:
        return out
    cut = max(out.rfind("；", 0, cap + 1), out.rfind("，", 0, cap + 1))
    return out[:cut] if cut > 0 else out[:cap]


def main():
    config = load_config()
    root = find_project_root(config["notes_path"])
    print(json.dumps(
        build_payload(root, config),
        ensure_ascii=False,
        separators=(",", ":"),
    ))


def build_payload(project_root: Path, config: dict) -> dict:
    notes_rel = config["notes_path"]
    # 草稿固定落在 skill 目录内；回传成相对项目根的路径，方便 SKILL.md 直接使用
    draft_rel = _rel_to(SKILL_DIR / config["draft_name"], project_root)
    words_txt = SKILL_DIR / "words.txt"
    words_csv = SKILL_DIR / "words.csv"
    chunk_md = project_root / notes_rel

    # 1. words.txt 中最后一个 ===日期=== 批次
    date_str = None
    batch: list[str] = []
    if words_txt.exists():
        lines = words_txt.read_text(encoding="utf-8").splitlines()
        last = None
        for i, ln in enumerate(lines):
            st = ln.strip()
            if st.startswith("===") and st.endswith("==="):
                last = i
                date_str = st.strip("= ")
        if last is not None:
            batch = [ln.strip() for ln in lines[last + 1:] if ln.strip()]

    # 2. lexical-chunk.md：已整理单词、现有分组结构、最大组号
    organized: set[str] = set()
    existing_groups: list[dict] = []
    max_num = 0
    if chunk_md.exists():
        text = chunk_md.read_text(encoding="utf-8")
        for m in re.finditer(r"^##\s+([^、\s]+)、\s*(.+)$", text, re.M):
            cn = m.group(1)
            n = cn_to_num(cn)
            if n and n > max_num:
                max_num = n
            members = [p.strip() for p in m.group(2).split("/") if p.strip()]
            existing_groups.append({"num": n, "num_cn": cn, "members": members})
            for p in members:
                if not (p.startswith("(") and p.endswith(")")):
                    organized.add(p)
        for m in re.finditer(r"^###\s+\d+\.\s+(.+)$", text, re.M):
            w = m.group(1).strip()
            if w and not (w.startswith("(") and w.endswith(")")):
                organized.add(w)

    # 待整理 = 本批中尚未出现在 lexical-chunk.md 的词
    to_do = [w for w in batch if w not in organized]

    # 3. 从 words.csv 取释义
    defs: dict[str, str] = {}
    if words_csv.exists():
        with open(words_csv, encoding="utf-8-sig", newline="") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                first = row[0].strip()
                if not first or first == "word" or first.startswith("==="):
                    continue
                defs[first] = (row[1].strip() if len(row) > 1 else "")

    return {
        "date": date_str,
        "notes_path": notes_rel,
        "draft_path": draft_rel,
        "latest_batch_count": len(batch),
        "already_organized_count": len(organized),
        "to_organize_count": len(to_do),
        "next_group_number": max_num + 1,
        "next_group_number_cn": num_to_cn(max_num + 1),
        "existing_groups": existing_groups,
        "words": [
            {"word": w, "definition": condense_definition(defs.get(w, ""))}
            for w in to_do
        ],
    }


if __name__ == "__main__":
    main()
