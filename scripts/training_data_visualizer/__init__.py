"""训练数据的人工检查工具（web app），按任务分目录：

    aim/                 瞄准数据集（`data/demos/prepared/aim/**/*.npz`）的逐帧可视化
                         一条样本 = 击杀前 4 秒的窗口（257 帧）
    spatial_prediction/  空间预测数据集（`prepared/spatial_prediction/**/*.npz`）
                         一条样本 = **一帧** + 5 个未来标签（CT 胜 / 终局存活 /
                         还能否击杀 / 下一个死亡 / 下一个击杀）

每个子目录都是**独立可跑的 web app**：`python -m scripts.training_data_visualizer.<任务>`。
它们只读 `prepared` 里已经写好的样本，不碰 `.csz`、不 import `demo_extract`
（那会连带 `demoparser2` / `numba`），所以在一台只有 numpy + Pillow 的机器上也能开。
"""
