"""
格式转换模块 - 保存为 VoxDet 简化 .npz 格式

输出格式:
  .npz 单文件包含:
  - sensor_pc: [N, 3] float32 皮肤点云坐标
  - voxel_labels: [X, Y, Z] uint8 体素标签
  - grid_*: 网格元数据
"""

import numpy as np
from pathlib import Path
from typing import Union, Dict, Any
import json


def save_case_npz(
    sensor_pc: np.ndarray,
    voxel_labels: np.ndarray,
    grid_info: Dict[str, Any],
    output_path: Union[str, Path]
) -> None:
    """
    保存单个案例数据为 .npz 格式

    参数:
        sensor_pc: (N, 3) 世界坐标点云（毫米）
        voxel_labels: (X, Y, Z) 体素标签
        grid_info: 网格信息字典，包含 world_min, world_max, voxel_size, occ_size
        output_path: 输出路径 (.npz)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    world_min = np.asarray(grid_info.get('world_min', [0, 0, 0]), dtype=np.float32)
    world_max = np.asarray(grid_info.get('world_max', [0, 0, 0]), dtype=np.float32)
    voxel_size = np.asarray(grid_info.get('voxel_size', [1, 1, 1]), dtype=np.float32)
    occ_size = np.asarray(grid_info.get('occ_size', voxel_labels.shape), dtype=np.int32)

    np.savez_compressed(
        str(output_path),
        sensor_pc=sensor_pc.astype(np.float32),
        voxel_labels=voxel_labels.astype(np.uint8),
        grid_world_min=world_min,
        grid_world_max=world_max,
        grid_voxel_size=voxel_size,
        grid_occ_size=occ_size
    )


def save_dataset_info(
    output_dir: Union[str, Path],
    class_names: list,
    num_classes: int,
    voxel_size: list = None,
    max_occ_size: int = 256,
    min_occ_size: int = 64
) -> None:
    """
    保存数据集全局配置信息

    参数:
        output_dir: 输出目录
        class_names: 类别名称列表
        num_classes: 类别数量
        voxel_size: 固定体素大小（毫米），如 [4.0, 4.0, 4.0]
        max_occ_size: 单轴最大体素数
        min_occ_size: 单轴最小体素数
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    info = {
        'num_classes': num_classes,
        'class_names': class_names,
        'voxel_size_mm': voxel_size or [4.0, 4.0, 4.0],
        'max_occ_size': max_occ_size,
        'min_occ_size': min_occ_size,
        'coordinate_system': 'world_mm',
        'label_dtype': 'uint8',
        'pointcloud_dtype': 'float32',
        'format_version': '3.0',
        'note': '网格大小根据数据动态计算，体素大小固定'
    }

    info_path = output_dir / 'dataset_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def verify_npz_format(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    验证 .npz 文件格式

    参数:
        file_path: 文件路径

    返回:
        验证结果字典
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return {'valid': False, 'error': 'File not found'}

    try:
        data = np.load(str(file_path))

        required_keys = ['sensor_pc', 'voxel_labels', 'grid_world_min',
                         'grid_world_max', 'grid_voxel_size', 'grid_occ_size']
        missing_keys = [k for k in required_keys if k not in data.files]
        if missing_keys:
            return {'valid': False, 'error': f'Missing keys: {missing_keys}'}

        sensor_pc = data['sensor_pc']
        voxel_labels = data['voxel_labels']

        return {
            'valid': True,
            'num_points': len(sensor_pc),
            'pointcloud_shape': sensor_pc.shape,
            'labels_shape': voxel_labels.shape,
            'xyz_range': {
                'min': sensor_pc.min(axis=0).tolist(),
                'max': sensor_pc.max(axis=0).tolist(),
            },
            'unique_labels': np.unique(voxel_labels).tolist(),
            'non_zero_voxels': int((voxel_labels > 0).sum()),
            'grid_info': {
                'world_min': data['grid_world_min'].tolist(),
                'world_max': data['grid_world_max'].tolist(),
                'voxel_size': data['grid_voxel_size'].tolist(),
                'occ_size': data['grid_occ_size'].tolist()
            }
        }
    except Exception as e:
        return {'valid': False, 'error': str(e)}
