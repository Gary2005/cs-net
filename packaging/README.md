# 打包成 Windows 双击即用的 zip

给玩家的是两个 zip，解压后双击 `cs-net.exe`，不需要 Python、conda、命令行：

| 包 | 体积 | 权重 | 适合谁 |
|---|---|---|---|
| `cs-net-5.0.0-win64-lite.zip` | ~610 MB | 不随包，首次开 AI 时点横幅下载（665 MB） | 默认选择 |
| `cs-net-5.0.0-win64-full.zip` | ~1.2 GB | 随包，解压即离线可用 | 网络差 / 离线机器 |

地图几何**两个包里都是解压好的**：943 MB 的文本 OBJ 压进 zip 只占约 300 MB，
不值得换玩家一次"点横幅 + 等解压 + 额外占 943 MB"。

## 解压后的形状

```
cs-net-5.0.0-win64\
├─ cs-net.exe                 双击这个
├─ _internal\                 Python 运行时 + 依赖 + 模板/静态资源（不能删）
├─ maps\optimized_obj_files\*.obj
├─ models\...                 仅 full 版
├─ build-info.json            版本、commit、torch 版本、权重 sha256
└─ data\                      首次运行自动生成：config.json、library.db、csz、thumbs、logs
```

`data\` 在 exe 旁边（绿色版）：换电脑整目录拷走，不写注册表。如果 exe 所在目录
不可写（比如被塞进 `Program Files`），会自动退到 `%LOCALAPPDATA%\cs-net` —— 规则
在 [`app/paths.py`](../app/paths.py)，`CSNET_DATA_DIR` 环境变量可以强制指定。

## 本地怎么打

```bash
# 1) 一个干净的 3.12 环境（原生轮子全是 cp312，别用 3.10/3.11）
python -m venv .build\packenv
.build\packenv\Scripts\pip install -r packaging\requirements-build.txt -r packaging\requirements-app.txt

# 2) 火/烟辅助程序（CI 里由 go build 提供）
bash tools/demo_extract/build_sidechannel.sh

# 3) 地图先解压，权重先就位（full 版要用）
python scripts/extract_maps.py
python scripts/download_models.py

# 4) 打 onedir，再压成两个 zip
.build\packenv\Scripts\python -m PyInstaller packaging\cs-net.spec --noconfirm
pwsh -File packaging\package.ps1 -Flavor both
dist\cs-net\cs-net.exe --packaging-selftest      # 先自检，再发
```

产物在 `packaging\out\`。

## CI 怎么跑

`.github/workflows/release-win.yml`：

* **push 一个 `v*` tag** → 构建 + 自检 + 组包 + 建 GitHub Release（`lite` / `full` /
  `sha256sums.txt` 三个资产）；
* **手动 `workflow_dispatch`** → 只跑构建，产物进 artifact（14 天），不建 Release。
  改 `cs-net.spec` 之后用这个验证，不用先想版本号。

流水线里最要紧的一步是自检，它跑的是**打出来的 exe**，而且把 `PATH` 清成只剩
`C:\Windows\system32;C:\Windows`。不清的话，漏收的 DLL 会从 runner 的 Python 环境
里被找到 —— 那就是"CI 全绿、玩家那边 `DLL load failed`"的经典来源。

## 几个刻意的选择

**onedir，不是 onefile。** `--onefile` 等于把 1.5 GB 打成自解压包，每次双击都要
解到 `%TEMP%`，冷启动半分钟，还容易被杀软误报。

**CPU 版 torch。** 默认 Windows 轮子是 `+cu128`（4.4 GB 的 CUDA DLL）。推理是
10 个人、几百毫秒的事，GPU 一点忙帮不上，却要多背 3.5 GB，还要赌玩家机器上
没有 CUDA 运行时的 `DLL load failed`。`app/ai/models.py` 里 `torch.cuda.is_available()`
在 CPU 轮子上恒为 False，自动落回 CPU，代码零改动。

**地图在 CI 里解压。** 见开头：外层 zip 从 207 MB（已是压缩格式）变成约 250 MB，
多 40 MB 下载，换掉玩家一次解压步骤和 943 MB 临时空间。

**权重不随 lite 包。** 公开分发时 1.1 GB 比 1.6 GB 的下完率高得多；权重还能在
HF 上独立更新，不用为了换模型重发整个包。full 版是兜底。

**`--packaging-selftest` 写进 exe 里。** 不是单测，是给产物做的体检：import
`torch` / `numba` / `demoparser2`、走一遍 `demo_extract` 和 `app.ai` 两条最容易漏
的链、起真的 HTTP 服务打 `/healthz` `/library` `/replay` `/api/maps` 等路由，
最后打印一行 `SELFTEST_JSON`。缺隐藏 import 时，玩家看到的是"库页面和回放都
正常、只有 AI 面板 500" —— 那种 bug 只能靠它挡。

## 体积预算（实测 5.0.0）

| 组成 | 压缩后 |
|---|---|
| Python 运行时 + 依赖（含 CPU torch 的 114 MB 轮子） | ~300 MB |
| Valve 静态资源（武器模型 / 贴图 / 雷达图，295 MB） | ~260 MB |
| 地图几何（解压后 943 MB 的 OBJ） | ~300 MB |
| 权重 665 MB | full 才有 |

所以 lite **607 MB**、full **1225 MB**。都远在 GitHub Release 单文件 2 GB 之下，
不需要分卷。

## 常见问题

**"Windows 已保护你的电脑"** —— 包没做代码签名，zip 又带 Mark-of-the-Web。
点"更多信息 → 仍要运行"，或者右键 exe → 属性 → 解除锁定。买代码签名证书能消掉它，
这是 v2 的事。

**解压很慢** —— 3 万个小文件，Windows 自带解压要好几分钟，还容易被杀软实时扫描
拖住。建议 7-Zip / Bandizip。

**`_internal` 不能删也不能改名** —— 那是程序本体。`README.txt` 里第一行就该写这个。

**AI 面板一直转/报错** —— 先看 `data\logs\desktop.log`，再跑一次
`cs-net.exe --packaging-selftest`，把 `SELFTEST_JSON` 那行发出来。

**想知道玩家装的是哪个版本** —— `cs-net.exe --version`，或者看包里的
`build-info.json`（版本、commit、torch 版本、两份权重的 sha256）。
