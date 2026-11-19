"""模型预测接进回放页。

    from app.ai import service
    payload = service.predict_round(data, round_idx, csz_path=path)

    # 或者走 HTTP（页面用的就是这几条）
    GET /api/ai/status                  # 模型在不在（不加载权重）
    GET /api/ai/round/<idx>             # 整回合预测（阻塞）
    GET /api/ai/round/<idx>?progress=1  # 同上，边算边推 NDJSON 进度

模块分工
--------
* :mod:`app.ai.models`   写死的两份权重（`PINNED_WEIGHTS`：`<run>.pt` + `<run>.yaml`）并装上 GPU
* :mod:`app.ai.features` 把会话里已经加载好的 demo dict 变成模型输入（**复用**
  `scripts/training_data` 的特征代码，不另抄一份）
* :mod:`app.ai.service`  推理（spatial 5 个头 + aim 对枪）、对枪窗口、三层缓存、进度
* :mod:`app.ai.api`      三个 Flask 路由

这个 `__init__` **不 import 上面任何一个**：torch / numba / demo_extract 都是
一两秒级的 import，而 `app.server` 启动时不该为它们等。用的时候
`from app.ai import service`（或在路由里 import）。
"""
