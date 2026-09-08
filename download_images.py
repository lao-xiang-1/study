# -*- coding: utf-8 -*-
"""扫描 git 跟踪的 Markdown 文件，下载网络图片到同目录 assets/ 并改写链接。

用法: python download_images.py [--dry-run]
"""
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse, unquote

REPO = Path(__file__).resolve().parent
DRY_RUN = "--dry-run" in sys.argv
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

# <img src="https://..."> （双引号或单引号）
IMG_TAG_RE = re.compile(r'<img\s[^>]*?src=(["\'])(https?://[^"\']+)\1', re.IGNORECASE)
# ![alt](https://...)
MD_IMG_RE = re.compile(r'!\[([^\]]*)\]\((https?://[^)\s]+)\)')

url_map: dict[str, str] = {}  # url -> assets 内相对文件名（相对 md 所在目录）


def git_tracked_md_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=REPO, capture_output=True, check=True,
    ).stdout.decode("utf-8")
    return [
        REPO / line for line in out.splitlines()
        if line.strip()
        # 文档/工具目录，非笔记内容（含行内代码示例，避免误匹配）
        and not line.startswith(("docs/", ".claude/"))
    ]


def safe_name(url: str) -> str:
    """从 URL 生成安全文件名；无扩展名时按 Content-Type 补全。"""
    base = unquote(Path(urlparse(url).path).name) or "image"
    base = re.sub(r'[\\/:*?"<>|#%]', "_", base)
    return base


def download(url: str, dest_dir: Path) -> tuple[str | None, str]:
    """下载 url 到 dest_dir，返回 (文件名, 状态信息)。失败返回 (None, 原因)。"""
    if url in url_map:
        return url_map[url], "cached"

    name = safe_name(url)
    candidate = dest_dir / name
    # 同名但不同 URL：追加序号
    n = 1
    while candidate.exists() and candidate not in downloaded_targets:
        # 已存在但不是本次下载的文件，改名避免覆盖
        stem, suffix = candidate.stem, candidate.suffix
        n += 1
        candidate = dest_dir / f"{stem}_{n}{suffix}"
        name = candidate.name

    headers = {"User-Agent": UA}
    if "zhimg" in url:
        headers["Referer"] = "https://www.zhihu.com/"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            ctype = resp.headers.get("Content-Type", "")
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    if resp_status_bad(ctype, data):
        return None, f"bad response ({ctype}, {len(data)} bytes)"

    # 无扩展名时补全
    if not candidate.suffix:
        ext = {"image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif",
               "image/webp": ".webp", "image/svg+xml": ".svg"}.get(ctype.split(";")[0].strip(), ".jpg")
        candidate = candidate.with_suffix(ext)
        name = candidate.name

    if not DRY_RUN:
        dest_dir.mkdir(parents=True, exist_ok=True)
        candidate.write_bytes(data)
    downloaded_targets.add(candidate)
    url_map[url] = name
    return name, "ok"


def resp_status_bad(ctype: str, data: bytes) -> bool:
    if len(data) < 100:
        return True
    if ctype.startswith("text/") or "html" in ctype:
        return True
    return False


downloaded_targets: set[Path] = set()

stats = {"files": 0, "links": 0, "ok": 0, "fail": 0}
failures: list[tuple[Path, str, str]] = []

for md in git_tracked_md_files():
    if not md.exists():
        continue
    text = md.read_text(encoding="utf-8")
    orig = text
    dest_dir = md.parent / "assets"

    def sub_img_tag(m: re.Match) -> str:
        quote, url = m.group(1), m.group(2)
        stats["links"] += 1
        name, status = download(url, dest_dir)
        if name is None:
            failures.append((md, url, status))
            stats["fail"] += 1
            return m.group(0)
        stats["ok"] += 1
        return m.group(0).replace(url, f"assets/{name}")

    def sub_md_img(m: re.Match) -> str:
        alt, url = m.group(1), m.group(2)
        stats["links"] += 1
        name, status = download(url, dest_dir)
        if name is None:
            failures.append((md, url, status))
            stats["fail"] += 1
            return m.group(0)
        stats["ok"] += 1
        return f"![{alt}](assets/{name})"

    text = IMG_TAG_RE.sub(sub_img_tag, text)
    text = MD_IMG_RE.sub(sub_md_img, text)

    if text != orig:
        stats["files"] += 1
        if not DRY_RUN:
            md.write_text(text, encoding="utf-8", newline="")
        print(f"[updated] {md.relative_to(REPO)}")

print(f"\n文件 {stats['files']} 个更新，链接 {stats['links']} 个：成功 {stats['ok']}，失败 {stats['fail']}")
if failures:
    print("\n失败列表:")
    for md, url, why in failures[:20]:
        print(f"  {md.relative_to(REPO)}\n    {why}\n    {url[:120]}")
    if len(failures) > 20:
        print(f"  ...共 {len(failures)} 条")
