# Obsidian SVG 画图经验

在本 Obsidian vault 中写内联 SVG 时，为保证 Obsidian/Markdown 预览稳定渲染：

- 不要把 SVG 放在 ```svg 代码块中；代码块会显示源码，不会渲染成图。
- 优先直接内联 `<svg>...</svg>`。
- 白底不要用 `<rect width="100%" height="100%" fill="#fff"/>` 铺底，可能出现遮挡或渲染异常；优先在 `<svg>` 标签上写：
  - `style="background-color:#fff"`
- 不要依赖 `<style>` 和 `class="..."` 设置线条/文字样式；Obsidian 内联 SVG 中可能不稳定，导致线条看不到。
- 线条、文字、节点等都使用元素内联属性，例如：
  - `<line stroke="#111" stroke-width="2.4" fill="none" stroke-linecap="round" stroke-linejoin="round" .../>`
  - `<polyline stroke="#111" stroke-width="2.4" fill="none" stroke-linecap="round" stroke-linejoin="round" .../>`
  - `<circle fill="#111" .../>` 用于节点圆点
  - `<circle stroke="#111" stroke-width="2.4" fill="none" .../>` 用于空心圆
  - `<text font-family="Times New Roman, Microsoft YaHei, sans-serif" font-size="18" fill="#111" ...>`
- 修改后可检查是否仍残留 `<style>`、`class="..."`、背景 `<rect>`，这些都可能造成预览兼容问题。
