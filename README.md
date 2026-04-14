# Medical CT to VoxDet Data Pipeline

将医学CT数据（CT扫描 + 器官分割）转换为VoxDet兼容的3D体素训练数据。

## 核心设计

- **世界坐标系（毫米）** — 不做归一化，直接使用CT affine矩阵提供的物理坐标
- **固定体素尺寸，动态网格大小** — 指定物理体素大小（默认4×4×4mm），网格维度由点云边界自动计算
- **123类器官标签** — 覆盖TotalSegmentator v2全集，通过优先级列表解决重叠冲突
- **自动裁剪** — 基于HU阈值检测人体区域，裁剪多余空间

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 处理单个案例

```bash
python generate_voxdet_data.py \
    --input_dir /path/to/ct_data \
    --output_dir ./output \
    --single_case BDMAP_00000001
```

### 批量处理

```bash
python generate_voxdet_data.py \
    --input_dir /path/to/ct_data \
    --output_dir ./output \
    --train_ratio 0.7 \
    --val_ratio 0.15
```

### 验证输出

```bash
python generate_voxdet_data.py \
    --output_dir ./output \
    --verify
```

### 3D可视化

```bash
python scripts/visualize_voxel_3d.py --input path/to/case.npz
```

支持交互式Plotly渲染，包含预设视图（全器官、骨骼、胸腔、血管等）和透明度控制。

## CLI参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input_dir` | — | 输入数据目录 |
| `--output_dir` | — | 输出目录（必填） |
| `--single_case` | — | 仅处理指定案例 |
| `--train_ratio` | 0.7 | 训练集比例 |
| `--val_ratio` | 0.15 | 验证集比例 |
| `--voxel_size` | 4 4 4 | 体素物理尺寸（mm） |
| `--max_occ_size` | 256 | 网格上限 |
| `--min_occ_size` | 64 | 网格下限 |
| `--skip_existing` | false | 跳过已处理案例 |
| `--verify` | false | 仅验证模式 |
| `--seed` | 42 | 随机种子 |

## 数据格式

### 输入

```
input_dir/
├── BDMAP_00000001/
│   ├── ct.nii.gz
│   └── segmentations/
│       ├── liver.nii.gz
│       ├── kidney_left.nii.gz
│       └── ...
└── ...
```

### 输出

每个案例输出为单个压缩 `.npz` 文件：

```
output_dir/
├── train/
│   └── BDMAP_00000001.npz
├── val/
├── test/
└── dataset_info.json
```

`.npz` 内容：

| Key | 类型 | 说明 |
|-----|------|------|
| `sensor_pc` | float32, (N, 3) | 皮肤表面点云（世界坐标） |
| `voxel_labels` | uint8, (X, Y, Z) | 器官体素标签 |
| `grid_world_min` | float32, (3,) | 网格起始坐标 |
| `grid_world_max` | float32, (3,) | 网格终止坐标 |
| `grid_voxel_size` | float32, (3,) | 体素物理尺寸 |
| `grid_occ_size` | int, (3,) | 网格维度 |

## 项目结构

```
medical_voxdet_data_pipeline/
├── config/
│   ├── data_config.py            # DataConfig / CameraConfig / VoxelConfig
│   └── organ_mapping.py          # 123类器官映射与优先级
├── pipeline/
│   └── ct_to_voxdet.py           # CTToVoxDetConverter 主转换流程
├── utils/
│   ├── mesh_generation.py        # CT加载、皮肤mesh生成（marching cubes）
│   ├── camera_system.py          # 虚拟相机 + 深度渲染 → 点云
│   ├── voxelization.py           # 体素网格创建、器官标签、裁剪
│   └── format_converter.py       # NPZ保存与验证
├── scripts/
│   └── visualize_voxel_3d.py     # 交互式3D可视化（Plotly）
├── generate_voxdet_data.py       # 主入口
├── requirements.txt
└── README.md
```

## 处理流程

```
CT + Segmentations
    │
    ├─ 1. 加载CT → 提取affine矩阵
    ├─ 2. HU阈值 → 皮肤mesh → 深度渲染 → 传感器点云
    ├─ 3. 点云边界 → 创建体素网格（固定体素尺寸）
    ├─ 4. 逐器官查询分割掩码 → 按优先级合并标签
    ├─ 5. HU阈值检测人体区域 → 裁剪至人体边界
    └─ 6. 保存为 .npz
```

## 依赖

- numpy, nibabel, scikit-image, trimesh, pyrender, PyOpenGL, tqdm
