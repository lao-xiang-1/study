import re
import os

# 原文件路径
source_file = "微机原理与接口技术/第三章：指令系统与汇编语言程序设计.md"
# 目标目录（与原文件同目录）
target_dir = "微机原理与接口技术/第三章：指令系统与汇编语言程序设计/"

# 确保目标目录存在
os.makedirs(target_dir, exist_ok=True)

# 读取原文件内容
with open(source_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 分割内容，按二级标题（## 开头的行）作为分割点，保留分割符
sections = re.split(r'(?=\n## )', content)

# 处理每个部分
for idx, section in enumerate(sections):
    section = section.strip()
    if not section:
        continue

    # 尝试提取二级标题作为文件名
    title_match = re.search(r'##\s+(.+)', section)
    if title_match:
        # 有二级标题的情况
        filename_base = title_match.group(1)
    else:
        # 开头的章节概述部分
        filename_base = "00_章节概述"

    # 清理文件名中的非法字符
    safe_filename = re.sub(r'[\\/*?:"<>|]', "", filename_base) + ".md"
    # 完整文件路径
    full_path = os.path.join(target_dir, safe_filename)

    # 写入文件
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(section + "\n")

    print(f"已创建: {safe_filename}")

print(f"\n拆分完成！共生成{len(sections)}个文件，保存到目录: {target_dir}")