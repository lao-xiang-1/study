#code 

## css设置指定元素为透明
- 似乎只支持https格式的图片，不支持本地图片
- 可以将图片经过base64编码，直接放在css文件里面（会导致css文件体积变得很大）
```css
/* 让工作区面板变透明 */
.workspace, 
.workspace-split, 
.workspace-tabs, 
.workspace-leaf, 
.workspace-leaf-content,
.view-content, 
.markdown-source-view, 
.markdown-preview-view {
  background-color: transparent !important;
}
```

## 示例笔记库
[TIPS for blue topaz](D:\all_download\Downloads\TIPS%20for%20Blue%20Topaz)
- 路径编码机制：%20表示空格，具体是%加上ASCII码。如果是中文会更加复杂

## ctrl+shift+I打开dev tool