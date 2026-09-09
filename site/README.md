# cs-net 展示网站（site/）

`https://gary2005.github.io/cs-net/` 的静态站点源码，深色科技风单页，**支持中英文切换**（右上角 EN / 中文；默认跟随浏览器语言，选择会被记住）。

## 目录结构

```
site/
  index.html            # 单页站点（英文为静态内容，中文在 data-zh 属性里）
  css/style.css         # 深色科技风样式
  js/main.js            # i18n 切换 / 滚动动画 / 代码复制（无外部依赖）
  assets/logo.svg       # 项目 logo
  assets/screenshots/   # 截图目录（目前为空，占位图待替换）
```

## 新增 / 修改文案（双语）

每个可翻译元素挂两个属性，静态内容保持英文（无 JS 也能看）：

```html
<p data-en="English text" data-zh="中文文本">English text</p>
```

- `data-en` = 英文，`data-zh` = 中文，属性值允许内嵌 HTML（用单引号包裹内层属性，如 `<span class='grad'>`）
- 不需要翻译的元素（文件名、代码本身、图标等）保持原样即可

## 替换截图

1. 把截图（建议 16:9，PNG/WebP）放进 `site/assets/screenshots/`
2. 改 `index.html` 里 Gallery 段的 `<div class="shot-ph">…</div>` 为：

   ```html
   <img src="assets/screenshots/你的截图.png" alt="…" style="width:100%;display:block">
   ```

## 部署

推送到 GitHub 后由 `.github/workflows/pages.yml` 自动部署（推 `site/**` 时触发）。
仓库 Settings → Pages → Source 选 **GitHub Actions** 即可启用，无需其他配置。

## 本地预览

```bash
cd site && python3 -m http.server 8000   # 打开 http://127.0.0.1:8000
```
