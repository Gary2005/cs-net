# cs-net 展示网站（site/）

`https://gary2005.github.io/cs-net/` 的静态站点源码，深色科技风单页。

## 目录结构

```
site/
  index.html            # 单页站点（英文；需要中文版可加）
  css/style.css         # 深色科技风样式
  js/main.js            # 滚动动画 / 代码复制按钮（无外部依赖）
  assets/logo.svg       # 项目 logo
  assets/screenshots/   # 截图目录（目前为空，占位图待替换）
```

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
