# optimize_obj.py — OBJ 地图网格优化

> 文件: `scripts/geometry/optimize_obj.py`
>
> 功能: 使用 Open3D 的 quadric edge-collapse decimation 算法压缩 CS2 地图 OBJ 模型，用于 AI 训练

---

## 概述

该脚本使用 **Garland-Heckbert 二次误差度量边折叠算法** (`simplify_quadric_decimation`) 来减少 OBJ 网格的面数。相比简单的顶点量化，该算法能更好地保留锐利边缘和边界。

### 适用场景

- AI 模型训练需要轻量地图几何数据
- 降低显存占用，加快数据加载速度
- 在保持视觉结构的前提下压缩文件大小

---

## 环境

```bash
conda activate cs2demo
```

### 依赖

|包|用途|
|------|------|
|`open3d`|网格读取、清理、quadric 简化、写入|
|`argparse`, `os`, `sys`, `time`, `pathlib`|标准库，CLI 与文件处理|

---

## 使用方式

### 三种精简模式（互斥，必选其一）

```bash
# 模式 1: 目标文件大小（MB）
python scripts/geometry/optimize_obj.py --target-size 20

# 模式 2: 保留原始三角形比例
python scripts/geometry/optimize_obj.py --ratio 0.3

# 模式 3: 精确目标三角形数
python scripts/geometry/optimize_obj.py --target-faces 500000
```

### 可选参数

|参数|类型|默认值|说明|
|------|------|------|------|
|`--input-dir`|`str`|`maps/obj_files`|源 OBJ 文件目录|
|`--output-dir`|`str`|`maps/optimized_obj_files`|输出目录|
|`--files`|`str[]`|全部 `.obj`|指定要处理的文件列表|
|`--dry-run`|`flag`|`False`|预览而不实际写入文件|

### 示例

```bash
# 预览 de_dust2.obj 压缩到 ~15MB 的效果（不写入）
python scripts/geometry/optimize_obj.py --target-size 15 --files de_dust2.obj --dry-run

# 将所有 .obj 文件压缩到原始面数的 30%
python scripts/geometry/optimize_obj.py --ratio 0.3

# 精确压缩特定文件到 50 万面
python scripts/geometry/optimize_obj.py --target-faces 500000 --files de_nuke.obj de_inferno.obj
```

---

## 处理流水线

```text
读取 OBJ 文件
    │
    ▼
┌─────────────────────────────────┐
│ 1. 预清洗 (Pre-clean)            │
│   · remove_duplicated_vertices   │  合并重复顶点
│   · remove_degenerate_triangles  │  删除退化面
│   · remove_unreferenced_vertices │  删除孤立顶点
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ 2. 计算目标面数                  │
│   --ratio       → clean_t × ratio│
│   --target-faces → 直接使用      │
│   --target-size  → 文件大小反推  │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ 3. Quadric 简化 (Pass 1)         │
│   simplify_quadric_decimation()  │
│   若目标 ≥ 原始面数 → 跳过       │
└─────────────────────────────────┘
    │
    ▼  (仅 --target-size 模式)
┌─────────────────────────────────┐
│ 4. 校准修正 (Pass 2)             │
│   若输出大小偏差 > 5%:           │
│     · 计算实际 字节/三角形 比率  │
│     · 修正目标三角形数            │
│     · 重新执行 quadric 简化       │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ 5. 后清洗 (Post-clean)           │
│   再次清理退化面和孤立顶点        │
└─────────────────────────────────┘
    │
    ▼
写入 OBJ (ASCII, 无法线/颜色/UV)
```

---

## 文件大小估算模型

`--target-size` 模式使用经验公式反推目标三角形数:

|常量|值|说明|
|------|------|------|
|`_BYTES_PER_VERTEX_LINE`|28|每行顶点数据的典型字节数|
|`_BYTES_PER_FACE_LINE`|22|每行面数据的典型字节数|
|`_FILE_OVERHEAD`|200|文件头固定开销（字节）|
|`_BYTES_PER_TRI`|~40.8|`0.67 × 28 + 22`（考虑简化后顶点/面比）|

### 估算公式

```python
est_bytes = 200 + n_vertices × 28 + n_triangles × 22
target_triangles = max((target_bytes - 200) / 40.8, 4)
```

Pass 1 输出若偏差 > 5%，Pass 2 会使用该网格实际的 bytes-per-triangle 重新校准，确保最终文件大小尽可能接近目标。

---

## 核心函数

### `_clean_mesh(mesh, label) → dict`

对网格执行三步清理（均就地修改）:

1. `remove_duplicated_vertices()` — 合并坐标完全相同的顶点
2. `remove_degenerate_triangles()` — 删除零面积/退化面
3. `remove_unreferenced_vertices()` — 删除无面引用的孤立顶点

返回清理前后顶点/面数差值的字典。

### `_estimate_output_size(n_vertices, n_triangles) → int`

根据顶点和三角形数量估算 OBJ 文件的磁盘大小（字节）。

### `_triangles_for_target_bytes(target_bytes) → int`

从目标文件大小反推所需三角形数，使用 `_BYTES_PER_TRI` 常量。

### `process_file(input_path, output_path, *, target_triangles, ratio, target_size_mb) → dict`

完整的单文件处理流水线，返回详细的统计字典:

|字段|说明|
|------|------|
|`raw_vertices` / `raw_triangles`|原始网格数据|
|`clean_vertices` / `clean_triangles`|预清洗后数据|
|`simp_vertices` / `simp_triangles`|简化后数据|
|`final_vertices` / `final_triangles`|后清洗后最终数据|
|`v_reduction` / `t_reduction`|顶点/面缩减百分比|
|`pre_cleaned` / `post_cleaned`|是否实际清理了数据|
|`passes`|quadric 简化执行次数 (1 或 2)|
|`t_read` / `t_preclean` / `t_simplify` / `t_postclean` / `t_write`|各阶段耗时 (秒)|

### `main()`

CLI 入口:

- 解析参数并校验
- 收集 OBJ 文件列表
- 逐文件调用 `process_file()`
- 打印单文件及汇总统计

---

## 输出格式

每个文件处理后输出:

```text
─ de_nuke.obj
  pre-clean: 1,234,567v/4,100,000t → 1,230,000v/4,095,000t
  triangles:  4,095,000 →  1,228,500  (70.0%)
  vertices :  1,230,000 →    830,000  (32.5%)
  size     :    156.3 MB →     20.1 MB  (2 pass(es))
  timing   : read 2.3s  preclean 0.5s  simplify 4.1s  postclean 0.2s  write 1.1s
```

最后打印汇总:

```text
==========================================================
TOTAL
  triangles: 12,345,678 →  3,703,703  (70.0%)
  vertices :  3,700,000 →  2,490,000  (32.7%)
  wall time: 18.2s
```

---

## 注意事项

- **算法限制**: `simplify_quadric_decimation` 要求目标三角形数 ≥ 4
- **写入格式**: 固定输出 ASCII OBJ，不带法线/颜色/UV — 最小化文件体积
- **输出目录**: 自动创建（如不存在）
- **错误容忍**: 单文件失败不中断整体流程，打印错误后继续处理下一个
- **虚拟环境**: 使用前必须 `conda activate cs2demo` 以确保 Open3D 可用
