"""spatial_prediction 数据集的可视化 web app。

    python -m scripts.training_data_visualizer.spatial_prediction        # 默认 127.0.0.1:5118
    python -m scripts.training_data_visualizer.spatial_prediction --port 5119 --open

模块划分：

    data.py     npz → 浏览器能直接用的 JSON（把 [0,1] 的仿射值**逆变换**回物理量，
                并对同一文件的标签序列做不变式体检）
    server.py   标准库 HTTP server（不依赖 Flask）：扫描文件、发载荷、发雷达图
    static/     前端（原生 JS + canvas，没有构建步骤、没有 three.js）

样本本身**不在**这个目录里：默认读
`data/demos/prepared/spatial_prediction/{train,valid,test}`，一个 npz 装一个回合被随机
抽到的 K 帧，所以 URL 里除了文件名还要带行号（`?k=`）。
"""

from . import data

__all__ = ["data"]
