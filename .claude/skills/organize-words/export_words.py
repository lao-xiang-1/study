import argparse
import csv
import time
from datetime import date
from pathlib import Path


def _format_date(d):
    """格式化日期为 2026-8-12 形式（不补零）。"""
    return f"{d.year}-{d.month}-{d.day}"


def _read_existing_words_txt(path):
    """读取 words.txt 中已有的单词。

    返回 (单词集合, 最后一个日期标记)。日期标记行形如 ===2026-8-12===，
    会被跳过不计入单词。最后一个标记用于判断同一天是否已有日期段，
    避免重复写标记。文件不存在时返回 (空集合, None)。
    """
    words = set()
    last_marker = None
    if not Path(path).exists():
        return words, last_marker
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("===") and stripped.endswith("==="):
                last_marker = stripped
                continue
            if stripped:
                words.add(stripped)
    return words, last_marker


def _read_existing_words_csv(path):
    """读取 words.csv 中已有的单词。

    返回 (单词集合, 最后一个日期标记)。跳过表头行（首列为 word）和
    ===日期=== 分隔行。csv.reader 正确处理释义中的多行/逗号/引号。
    文件不存在时返回 (空集合, None)。
    """
    words = set()
    last_marker = None
    if not Path(path).exists():
        return words, last_marker
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            first = row[0].strip()
            if first == "word":  # 表头
                continue
            if first.startswith("===") and first.endswith("==="):
                last_marker = first
                continue
            words.add(first)
    return words, last_marker


def save_words(words, path, today=None):
    """增量追加新单词到 words.txt，新增单词前标注 ===日期===。

    读取已有文件中的单词，只追加未出现过的单词。若有新单词，插入空行 +
    ===YYYY-M-D=== 日期分隔行 + 空行（标记前后各空一行作视觉分隔，标记为
    文件首行时不加前导空行）；同一天已有日期段则直接追加到该段下，不重复
    写标记。若无新单词，文件保持不变。today 默认为当天，可传参以便测试。
    返回新增单词数量。
    """
    if today is None:
        today = date.today()
    marker = f"==={_format_date(today)}==="
    existing, last_marker = _read_existing_words_txt(path)
    seen = set(existing)
    new_words = []
    for w in words:
        if w not in seen:
            seen.add(w)
            new_words.append(w)
    if not new_words:
        return 0
    has_content = Path(path).exists() and Path(path).stat().st_size > 0
    with open(path, "a", encoding="utf-8") as f:
        if last_marker != marker:
            if has_content:
                f.write("\n")  # 标记前空一行
            f.write(f"{marker}\n\n")  # 标记行 + 标记后空一行
        for w in new_words:
            f.write(f"{w}\n")
    return len(new_words)


def save_words_csv(pairs, path, today=None):
    """增量追加新单词到 words.csv（word,definition 两列），新增单词前标注 ===日期===。

    读取已有文件中的单词，只追加未出现过的 (word, definition)。若有新单词，
    插入空行 + ===YYYY-M-D=== 分隔行（单列）+ 空行（标记前后各空一行，标记为
    文件首行时不加前导空行）；同一天已有日期段则直接追加，不重复写标记。
    若无新单词，文件保持不变。首次创建文件用 UTF-8 with BOM 并写表头；追加时
    用普通 UTF-8 避免中部插入 BOM。用 csv 模块处理逗号/引号/换行转义。today
    默认为当天。返回新增单词数量。
    """
    if today is None:
        today = date.today()
    marker = f"==={_format_date(today)}==="
    existing, last_marker = _read_existing_words_csv(path)
    seen = set(existing)
    new_pairs = []
    for word, definition in pairs:
        if word not in seen:
            seen.add(word)
            new_pairs.append((word, definition))
    if not new_pairs:
        return 0
    exists = Path(path).exists()
    has_content = exists and Path(path).stat().st_size > 0
    # 首次创建用 utf-8-sig 写 BOM；追加用普通 utf-8，避免在文件中部插入 BOM
    encoding = "utf-8" if exists else "utf-8-sig"
    with open(path, "a", encoding=encoding, newline="") as f:
        writer = csv.writer(f)
        preceding = has_content  # 此前文件已有内容
        if not exists:
            writer.writerow(["word", "definition"])
            preceding = True  # 刚写入表头，标记前需空一行
        if last_marker != marker:
            if preceding:
                writer.writerow([])  # 标记前空一行
            writer.writerow([marker])
            writer.writerow([])  # 标记后空一行
        for word, definition in new_pairs:
            writer.writerow([word, definition])
    return len(new_pairs)


def wait_for_login(page, timeout=300):
    """等待用户登录完成。timeout 单位为秒，默认 5 分钟。

    判定逻辑：URL 必须仍在 webwordbook 页面，且页面有 .word-name 元素
    （真正的单词内容）。未登录时页面会显示 0 个单词，不会通过检测。
    每隔几秒尝试重新导航到单词本页面，用户登录后无需手动跳回。
    """
    wordbook_url = "https://dict.youdao.com/webwordbook/wordlist"
    start = time.time()
    logged_in_hint = False
    while time.time() - start < timeout:
        url = page.url
        if "webwordbook" in url:
            word_count = page.locator(".word-name").count()
            if word_count > 0:
                return True
            if not logged_in_hint:
                print("未检测到单词，可能需要登录。请在浏览器中登录有道账号。")
                logged_in_hint = True
        else:
            page.wait_for_timeout(3000)
            try:
                page.goto(wordbook_url, wait_until="domcontentloaded")
            except Exception:
                pass
        page.wait_for_timeout(2000)
    raise TimeoutError(f"登录超时（{timeout} 秒）")


def debug_dump_page(page, path="page_dump.html"):
    """把当前页面 HTML 保存到文件，用于分析 DOM 结构。"""
    html = page.content()
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"页面 HTML 已保存到 {path}")


def extract_words_from_page(page):
    """从当前页提取 (word, definition) pairs。

    有道单词本每个单词在一个 .word-cord 容器内，含 .word-name（单词）和
    .word-detail（释义）。.word-name 内嵌的发音图标 <a> 无文本，text_content()
    直接返回单词。缺 .word-detail 时 definition 存空字符串。页内按 word 去重。
    """
    pairs = []
    for cord in page.locator(".word-cord").all():
        word_el = cord.locator(".word-name").first
        if not word_el.count():
            continue
        word = (word_el.text_content() or "").strip()
        if not word:
            continue
        detail_el = cord.locator(".word-detail").first
        definition = ""
        if detail_el.count():
            definition = (detail_el.text_content() or "").strip()
        if not any(w == word for w, _ in pairs):
            pairs.append((word, definition))
    if pairs:
        return pairs
    # 没匹配到，保存页面 HTML 供分析
    debug_dump_page(page, str(SKILL_DIR / "page_dump.html"))
    print("警告：未提取到单词，已保存页面 HTML 到 page_dump.html")
    return []


def iterate_all_pages(page, max_pages=200, existing_words=None, early_stop=True):
    """遍历所有分页，聚合 (word, definition) pairs。

    max_pages 是安全上限，防止无限循环。
    existing_words 是已导出单词集合，用于早停判断。
    early_stop=True 时，若当前页所有单词均已在 existing_words 中，
    认为后续页面也已处理，提前停止。
    """
    existing_words = existing_words or set()
    all_pairs = []
    for page_num in range(1, max_pages + 1):
        pairs = extract_words_from_page(page)
        if not pairs:
            break
        all_pairs.extend(pairs)
        known_on_page = sum(1 for w, _ in pairs if w in existing_words)
        new_on_page = len(pairs) - known_on_page
        print(
            f"第 {page_num} 页：提取到 {len(pairs)} 个单词"
            f"（新增 {new_on_page}，已有 {known_on_page}），累计 {len(all_pairs)}"
        )
        if early_stop and known_on_page == len(pairs):
            print(f"当前页 {known_on_page} 个单词全部已导出，早停")
            break
        # 尝试点击"下一页"（有道单词本的禁用 class 是 disable 不是 disabled）
        next_selectors = [
            "span.pagination-button.next:not(.disable)",
            "a:has-text('下一页')",
            "a:has-text('Next')",
            ".next:not(.disable)",
            "a[rel='next']",
        ]
        clicked = False
        for sel in next_selectors:
            btn = page.locator(sel)
            if btn.count() > 0 and btn.first.is_enabled():
                btn.first.click()
                page.wait_for_load_state("networkidle")
                page.wait_for_timeout(1000)
                clicked = True
                break
        if not clicked:
            print("没有下一页，结束")
            break
    return all_pairs


from playwright.sync_api import sync_playwright

# 脚本与数据文件同在 skill 目录内，所有路径都相对本文件解析，不依赖运行时 cwd，
# 这样从任何目录调用结果都一致。
SKILL_DIR = Path(__file__).resolve().parent
BROWSER_DATA_DIR = SKILL_DIR / "browser_data"


def main():
    parser = argparse.ArgumentParser(description="从有道单词本导出单词")
    parser.add_argument(
        "--full", action="store_true", help="禁用早停，遍历全部页面"
    )
    args = parser.parse_args()

    url = "https://dict.youdao.com/webwordbook/wordlist"
    txt_path = SKILL_DIR / "words.txt"
    csv_path = SKILL_DIR / "words.csv"

    # 读取已导出单词，用于早停判断（--full 时跳过读取）
    if not args.full:
        existing_txt, _ = _read_existing_words_txt(txt_path)
        existing_csv, _ = _read_existing_words_csv(csv_path)
        existing = existing_txt | existing_csv
    else:
        existing = set()

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            str(BROWSER_DATA_DIR), headless=False
        )
        page = context.pages[0] if context.pages else context.new_page()
        page.goto(url)

        print("如需登录，请在浏览器中完成登录。登录后脚本会自动继续。")
        wait_for_login(page)

        print("登录成功，开始抓取单词...")
        pairs = iterate_all_pages(page, existing_words=existing, early_stop=not args.full)

        today = date.today()
        new_txt = save_words([w for w, _ in pairs], txt_path, today)
        new_csv = save_words_csv(pairs, csv_path, today)
        date_str = _format_date(today)
        if new_txt:
            print(
                f"新增 {new_txt} 个单词，已标注 ==={date_str}=== 到 {txt_path} 和 {csv_path}"
            )
        else:
            print(f"没有新单词，{txt_path} 和 {csv_path} 未改动（共 {len(pairs)} 个单词）")

        context.close()


if __name__ == "__main__":
    main()
