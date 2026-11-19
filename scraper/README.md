# scraper — HLTV → `.dem` → `.csz`（v5）

从 HLTV 抓比赛 → 下载 `.rar` → 解出 `.dem` → 转成 v5 的 `.csz`。

**爬虫本体是从 `cs-net-dev/scraper/` 原样搬过来的**，只有「数据转换」那一步换了：
dev 用 `demo_parser.parse_demo` + `save_demo_json` 出 `.json.gz`，
v5 用 `tools/demo_extract.extract` + `codec.save` 出 `.csz`。

对照 diff 只涉及这些行（其余 900 多行逐字相同）：

| 位置 | dev | v5 |
|---|---|---|
| import | `from demo_parser import parse_demo, save_demo_json` | `from demo_extract import codec, extract as extract_demo` |
| 输出后缀 | `*.json` | `*.csz` |
| 输出目录 | `data/demos/json/` | `data/demos/csz/` |
| 解析 | `parse_demo(dem, interval=0.25)` | `extract_demo(dem, interval=..., stride=...)` |
| 写盘 | `save_demo_json(data, path, compact=True, compress=True)` | `codec.save(data, path)` |
| 采样默认 | 0.25 秒（4 Hz） | **每个 tick**（≈64 Hz） |

---

## 1. 输出长什么样

```
data/demos/
├─ csz/                  ← 成品：<match_id>-<team1>-vs-<team2>+<demo>.csz
├─ tmp/                  ← 下载的 .rar 与解出的 .dem（解析完即删）
├─ done.json             ← 已完成的 match_id 列表（断点续跑用）
└─ candidates.json       ← 待下载队列 + 翻页 offset（断点续跑用）
```

`data/` 已在 `.gitignore` 里。

一场 110 MB 的 de_ancient demo 实测体积（`demo_extract/README.md`）：

| 采样 | 采样点 | `.csz` 大小 |
|---|---|---|
| 每 tick（默认） | 74,003 | ~7.0 MB |
| `--interval 0.25` | 4,632 | ~0.9 MB |
| `--interval 0.5` | 2,322 | ~0.6 MB |

---

## 2. Linux 上的一次性准备

```bash
# 1) Python 依赖
pip install playwright rarfile demoparser2 pandas numpy zstandard
python -m playwright install chromium
sudo python -m playwright install-deps chromium     # 缺系统库时

# 2) unrar —— rarfile 解 .rar 要靠它
sudo apt install unrar          # Debian/Ubuntu
# sudo dnf install unrar        # Fedora（RPM Fusion）

# 3) ★ 重新编译火/烟辅助程序（见下，最容易漏）
bash tools/demo_extract/build_sidechannel.sh
```

### ★ 必须在 Linux 上重编 sidechannel

`tools/demo_extract/sidechannel_go/cs2sidechannel` **是 macOS arm64 的二进制**
（仓库里那份 `file` 出来是 `Mach-O 64-bit executable arm64`）。
Linux 上直接用会 `Exec format error`，**每一场解析都会失败**。

```bash
bash tools/demo_extract/build_sidechannel.sh
# 需要 Go 工具链；脚本会把 GOPATH/GOCACHE 指到 .build/ 下，不污染系统
```

编译产物会覆盖 `sidechannel_go/cs2sidechannel`。**不要把 macOS 那份提交回去**，
否则 Mac 上又跑不了。要同时支持两边就各自本地编译、不提交二进制。

> 不想要火/烟（文件更小）也可以跳过这步，但要在 `pipeline.py` 的 `extract_demo(...)`
> 里加 `include_fire_smoke=False` —— 默认是开着的，缺二进制会解析失败。

---

## 3. 跑

```bash
# 一次性：打开 Chrome 登录 HLTV / 过 Cloudflare，然后回车
python -m scraper --setup

# 抓 10 场，默认每个 tick
python -m scraper --count 10

# 省空间：4 Hz
python -m scraper --count 10 --interval 0.25

# 每 8 个 tick 采一次
python -m scraper --count 10 --stride 8

# 只要 3 星以上的比赛
python -m scraper --count 20 --min-stars 3

# Chrome 没开时自动拉起
python -m scraper --count 10 --auto-launch
```

`python -m scraper --help` 看全部参数。

### 采样率怎么选

`extract_demo()` 的语义（`extract.py:_pick_ticks`）：

* `interval=None`（**默认**）或 `interval=0` → **每个 tick**，≈64 Hz；
* `interval=0.25` → 4 Hz，取离目标秒数最近的真实 tick；
* `stride=N` → 每 N 个 tick 采一次，**优先级高于 `--interval`**，`stride=1` = 每 tick。

> ⚠️ 下蹲过渡只有约 200 ms（13 tick）。`--interval 0.25` 会把这条斜坡吃掉 94%，
> 想要 `crouch` 的完整过程就别降采样（见 `tools/demo_extract/API.md` §4.12）。

### 断点续跑

`done.json` / `candidates.json` 每次下载完都会落盘。中断后重跑同一条命令即可，
已经完成的 `match_id` 会被跳过，翻页 offset 也会接着上次继续。

---

## 4. 跑起来之后先确认这几件事

爬虫本身在你本地测不了，但下面这些都能在 Linux 上第一次运行时立刻看出来：

1. **第一场解析成功**，日志里出现 `✅ <team1> vs <team2>`；
   失败是 `❌` 或 `💥`，`--verbose` 会多打一行 `📊 ... (xxxx KB)`。
2. **`data/demos/csz/` 里出现了非空的 `.csz`**（每 tick 约 7 MB）。
3. **没有 `Exec format error` / `unable to find existing entity`** ——
   前者是 sidechannel 没在 Linux 重编，后者是 demoinfocs 版本过旧（需要 `/v5`）。
   还有一种**不报错**的：`.csz` 里 `fire` / `smoke` 是 0（2026-11-11 之后的 demo 用
   v5.2.0 就是这个结果），那是 demoinfocs 需要 **v5.3.0**。
4. 随手验一场：

   ```bash
   PYTHONPATH=tools python -m demo_extract --inspect data/demos/csz/<某个>.csz
   ```

   会打印 format / map / rounds / fire / smoke 数量；`format` 应是 `cs2.demo.v7`。

`--workers` 只并行「解析」，下载是主线程串行的（Cloudflare 友好），别调太大，默认 2、上限 4。
