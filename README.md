# 医学CT到VoxDet数据转换管道

将医学CT数据（CT扫描 + 器官分割）转换为VoxDet兼容的训练数据格式。

## 关键设计决策

根据 `ANALYSIS_REUSE.md` 的分析：

1. **不需要归一化** - 直接使用世界坐标（毫米）
2. **皮肤和器官数据天然对齐** - 都使用相同的 `ct_affine` 矩阵
3. **体素网格范围基于皮肤点云边界确定** - 保证覆盖一致

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 处理单个案例

```bash
python generate_voxdet_data.py \
    --input_dir /path/to/ct_data \
    --output_dir ./output/voxdet_medical \
    --single_case BDMAP_00000001
```

### 批量处理

```bash
python generate_voxdet_data.py \
    --input_dir /path/to/ct_data \
    --output_dir ./output/voxdet_medical \
    --train_ratio 0.7 \
    --val_ratio 0.15
```

### 验证输出

```bash
python generate_voxdet_data.py \
    --output_dir ./output/voxdet_medical \
    --verify
```

## 输入数据结构

```
input_dir/
├── BDMAP_00000001/
│   ├── ct.nii.gz                    # CT扫描
│   └── segmentations/
│       ├── liver.nii.gz             # 器官分割
│       ├── spleen.nii.gz
│       └── ...
├── BDMAP_00000002/
│   └── ...
└── ...
```

## 输出数据结构

```
output/voxdet_medical/
├── sequences/
│   ├── train/velodyne/
│   │   ├── BDMAP_00000001.bin       # 点云 [N, 4] float32
│   │   └── ...
│   ├── val/velodyne/
│   └── test/velodyne/
├── labels/
│   ├── train/
│   │   ├── BDMAP_00000001_1_1.npy   # 体素标签 [128,128,128] uint8
│   │   ├── BDMAP_00000001_1_2.npy   # 下采样 [64,64,64] uint8
│   │   └── ...
│   ├── val/
│   └── test/
└── metadata/
    ├── class_names.json
    ├── train_split.txt
    ├── val_split.txt
    └── test_split.txt
```

## 项目结构

```
medical_voxdet_data_pipeline/
├── config/
│   ├── data_config.py               # 数据配置
│   ├── organ_mapping.py             # 器官标签映射
│   └── voxdet_config_template.py    # VoxDet配置模板
├── utils/
│   ├── mesh_generation.py           # Mesh生成（复用自CT2PointCloud）
│   ├── camera_system.py             # 相机系统（复用）
│   ├── voxelization.py              # 体素化
│   └── format_converter.py          # 格式转换
├── pipeline/
│   └── ct_to_voxdet.py              # 主转换管道
├── scripts/
│   ├── verify_output.py             # 验证输出
│   └── visualize_sample.py          # 可视化样本
├── tests/
│   └── test_all_modules.py          # 全面测试
├── generate_voxdet_data.py          # 主入口
└── requirements.txt
```

## 器官类别

| ID | 器官 | ID | 器官 |
|----|------|----|----|
| 0 | background | 10 | colon |
| 1 | liver | 11 | lung_left |
| 2 | spleen | 12 | lung_right |
| 3 | kidney_left | 13 | aorta |
| 4 | kidney_right | 14 | inferior_vena_cava |
| 5 | stomach | 15 | adrenal_gland_left |
| 6 | pancreas | 16 | adrenal_gland_right |
| 7 | gallbladder | 17 | esophagus |
| 8 | urinary_bladder | 18 | small_bowel |
| 9 | heart | 19 | duodenum |

## VoxDet配置

生成数据后，需要计算 `point_cloud_range`：

```python
from config.voxdet_config_template import compute_dataset_range

# 计算点云范围
range = compute_dataset_range('output/voxdet_medical/sequences/train/velodyne')
# 输出: [-500.0, -500.0, -200.0, 100.0, 100.0, 400.0]

# 在VoxDet配置中使用
point_cloud_range = range
occ_size = [128, 128, 128]
```

## 运行测试

```bash
cd medical_voxdet_data_pipeline
python tests/test_all_modules.py
```

## 技术细节

### 坐标系统

- 使用世界坐标（毫米），不需要归一化
- 皮肤点云和体素标签在同一坐标系中对齐
- VoxDet模型通过 `point_cloud_range` 配置处理坐标映射

### 体素化流程

1. 加载CT数据和affine矩阵
2. 生成皮肤mesh → 深度渲染 → 皮肤点云
3. 根据皮肤点云边界创建体素网格
4. 对每个器官分割：查询体素标签
5. 按优先级合并多器官标签
6. 保存为VoxDet格式

### 复用代码

- `mesh_generation.py`: 100%复用自CT2PointCloud
- `camera_system.py`: 大部分复用（去除归一化）
- `voxelization.py`: 新实现
- `format_converter.py`: 新实现
