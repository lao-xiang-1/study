import argparse
import re
import os


def split_chapter(source_file, target_dir, level=2):
    """按指定级别标题拆分 Markdown 文件。

    Args:
        source_file: 源 Markdown 文件路径
        target_dir: 拆分后文件保存目录
        level: 按第几级标题拆分（1-6）
    """
    if not 1 <= level <= 6:
        raise ValueError("level 必须为 1-6")

    os.makedirs(target_dir, exist_ok=True)

    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 按指定级别标题作为分割点，保留分割符
    pattern = rf'(?=\n{{{{1,6}}}} {"#" * level} )'
    # 修正：匹配指定级别的标题行（行首 N 个 # 加空格）
    heading_re = re.compile(rf'^#{{{level}}}\s+(.+)$', re.MULTILINE)
    split_re = re.compile(rf'(?m)(?=^#{{{level}}}\s+)')
    sections = split_re.split(content)

    # 统一计数（仅对实际写入的文件编号）
    file_idx = 0
    created = 0
    for section in sections:
        section = section.strip()
        if not section:
            continue

        title_match = heading_re.search(section)
        filename_base = title_match.group(1).strip() if title_match else "章节概述"

        # 清理文件名中的非法字符
        safe_name = re.sub(r'[\\/*?:"<>|]', "", filename_base)
        # 加上两位序号前缀，便于排序
        prefix = f"{file_idx:02d}_"
        safe_filename = prefix + safe_name + ".md"
        full_path = os.path.join(target_dir, safe_filename)

        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(section + "\n")

        print(f"已创建: {safe_filename}")
        file_idx += 1
        created += 1

    print(f"\n拆分完成！共生成 {created} 个文件，保存到目录: {target_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="按指定级别标题拆分 Markdown 文件，每个文件名加两位序号前缀便于排序。"
    )
    parser.add_argument(
        "source_file",
        nargs="?",
        default="5.4 放大电路的分析.pdf_by_PaddleOCR-VL-1.6.md",
        help="源 Markdown 文件路径（默认：5.4 放大电路的分析.pdf_by_PaddleOCR-VL-1.6.md）",
    )
    parser.add_argument(
        "target_dir",
        nargs="?",
        default="5.4 放大电路的分析/",
        help="拆分后文件保存目录（默认：5.4 放大电路的分析/）",
    )
    parser.add_argument(
        "-l", "--level",
        type=int,
        default=2,
        choices=range(1, 7),
        help="按第几级标题拆分（1-6，默认 2）",
    )
    args = parser.parse_args()
    split_chapter(args.source_file, args.target_dir, args.level)
