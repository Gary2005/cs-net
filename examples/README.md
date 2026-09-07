# examples

示例数据目录：

- `examples/json/test.json.gz` — **随仓库发布**的回合测试数据（`cs2.demo.v2`
  格式，de_mirage，24 回合）。用于：
  - `python scripts/test_checkpoints.py` 的完整 pipeline 测试
    （json.gz → filter → process_round → 模型推理，与训练/可视化同链路）
  - 可视化工具页面一键加载（会出现在 `/api/examples` 列表里）
- `examples/demo/` — `.dem` 录像文件（git 忽略，可自行放入，
  用 `python -m demo_parser file.dem -o out.json` 解析）
- `examples/json/` — 其它已解析的回合 JSON / `.json.gz`（git 忽略，
  `test.json.gz` 除外）
- `examples/dataset/` — round 级 WebDataset shards（git 忽略，
  训练 / 微调用 `--data-dir` 指向它）
