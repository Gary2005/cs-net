# examples

示例数据目录（git 忽略，不入库）：

- `examples/demo/` — `.dem` 录像文件（可用 `python -m demo_parser file.dem -o out.json` 解析）
- `examples/json/` — 已解析的回合 JSON / `.json.gz`（`cs2.demo.v2` 格式，见 `docs/demo-json-format.md`）
- `examples/dataset/` — round 级 WebDataset shards（`train/`、`test/` 子目录）

可视化工具启动后，`/api/examples` 会列出 `examples/demo` 与 `examples/json`
下的文件供页面一键加载；把录像放进对应目录即可使用该功能。
