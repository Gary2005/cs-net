"""aim 数据集的可视化 web app。

    python -m scripts.training_data_visualizer.aim            # 默认 127.0.0.1:5117
    python -m scripts.training_data_visualizer.aim --port 5118 --open

模块划分：

    data.py     npz → 浏览器能直接用的 JSON（把 [0,1] 的仿射值**逆变换**回物理量）
    views.py    `atk_view` / `tgt_view`（192×256×5，单位 HU）→ PNG
    server.py   标准库 HTTP server（不依赖 Flask）：扫描样本、发载荷、发图
    static/     前端（原生 JS + canvas，没有构建步骤、没有 three.js）

样本本身**不在**这个目录里：默认读 `data/demos/prepared/aim/{train,valid,test}`。
"""
