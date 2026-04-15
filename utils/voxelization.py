"""
体素化模块 - 将器官分割转换为体素标签

核心功能:
1. 根据皮肤点云边界创建体素网格的世界坐标
2. 查询每个体素在器官分割中的标签
3. 合并多器官标签（按优先级）
4. 创建体内掩码（基于CT HU阈值）
5. 裁剪到人体边界框
"""

import numpy as np
from typing import Tuple, Dict, List
import warnings


def create_voxel_grid_world_coords(
    sensor_pc: np.ndarray,
    padding: float = 0.05,
    voxel_size: List[float] = None,
    max_occ_size: int = 256,
    min_occ_size: int = 64
) -> Tuple[np.ndarray, Dict]:
    """
    根据皮肤点云边界创建体素网格的世界坐标

    固定体素大小模式: 指定 voxel_size，网格大小动态计算

    参数:
        sensor_pc: (N, 3) 皮肤点云，世界坐标（毫米）
        padding: 边界扩展比例（避免边缘裁剪）
        voxel_size: 固定体素大小（毫米），如 [4.0, 4.0, 4.0]
        max_occ_size: 单轴最大体素数（防止内存溢出）
        min_occ_size: 单轴最小体素数（保证最小分辨率）

    返回:
        voxel_world_coords: (X, Y, Z, 3) 每个体素中心的世界坐标
        grid_info: 网格信息字典
    """
    if len(sensor_pc) == 0:
        raise ValueError("皮肤点云为空")

    if voxel_size is None:
        voxel_size = [4.0, 4.0, 4.0]

    voxel_size = np.array(voxel_size, dtype=np.float64)

    # 计算皮肤点云边界
    min_bound = sensor_pc.min(axis=0)
    max_bound = sensor_pc.max(axis=0)

    # 添加padding
    extent = max_bound - min_bound
    min_bound = min_bound - extent * padding
    max_bound = max_bound + extent * padding

    # 使用各向同性网格（最大范围，确保立方体）
    max_extent = (max_bound - min_bound).max()
    center = (min_bound + max_bound) / 2

    # 计算需要的网格大小（向上取整）
    computed_occ_size = np.ceil(max_extent / voxel_size).astype(int)

    # 应用网格大小限制
    computed_occ_size = np.clip(computed_occ_size, min_occ_size, max_occ_size)

    # 由于网格大小被限制，重新计算实际的世界范围
    actual_extent = computed_occ_size * voxel_size
    world_min = center - actual_extent / 2
    world_max = center + actual_extent / 2
    max_extent = actual_extent.max()

    final_occ_size = computed_occ_size.tolist()

    # 创建体素中心坐标
    x = np.linspace(
        world_min[0] + voxel_size[0] / 2,
        world_min[0] + voxel_size[0] * (final_occ_size[0] - 0.5),
        final_occ_size[0]
    )
    y = np.linspace(
        world_min[1] + voxel_size[1] / 2,
        world_min[1] + voxel_size[1] * (final_occ_size[1] - 0.5),
        final_occ_size[1]
    )
    z = np.linspace(
        world_min[2] + voxel_size[2] / 2,
        world_min[2] + voxel_size[2] * (final_occ_size[2] - 0.5),
        final_occ_size[2]
    )

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    voxel_world_coords = np.stack([xx, yy, zz], axis=-1)

    grid_info = {
        'world_min': world_min,
        'world_max': world_max,
        'voxel_size': voxel_size,
        'occ_size': final_occ_size,
        'center': center,
        'max_extent': max_extent,
        'mode': 'fixed_voxel_size',
    }

    return voxel_world_coords, grid_info


def label_voxels_from_segmentation(
    voxel_world_coords: np.ndarray,
    seg_data: np.ndarray,
    ct_affine: np.ndarray,
    organ_label: int
) -> np.ndarray:
    """
    查询每个体素在器官分割中的标签

    参数:
        voxel_world_coords: (X, Y, Z, 3) 体素中心世界坐标
        seg_data: 器官分割数据（3D数组）
        ct_affine: CT的仿射矩阵
        organ_label: 该器官的标签ID

    返回:
        labels: (X, Y, Z) 体素标签
    """
    shape = voxel_world_coords.shape[:3]
    flat_coords = voxel_world_coords.reshape(-1, 3)

    # 世界坐标 -> CT体素坐标
    affine_inv = np.linalg.inv(ct_affine)
    coords_homo = np.hstack([flat_coords, np.ones((len(flat_coords), 1))])
    ct_coords = (affine_inv @ coords_homo.T).T[:, :3]

    # 四舍五入到最近的体素
    indices = np.round(ct_coords).astype(int)

    # 边界检查
    valid = ((indices >= 0) & (indices < np.array(seg_data.shape))).all(axis=1)

    # 查询标签
    labels = np.zeros(len(flat_coords), dtype=np.uint8)
    valid_indices = indices[valid]

    is_inside = seg_data[valid_indices[:, 0],
                         valid_indices[:, 1],
                         valid_indices[:, 2]] > 0
    labels[valid] = np.where(is_inside, organ_label, 0)

    return labels.reshape(shape)


def combine_organ_labels(
    label_arrays: List[np.ndarray],
) -> np.ndarray:
    """
    合并多个器官标签（按优先级）

    后处理的器官会覆盖先处理的（用于处理重叠区域）

    参数:
        label_arrays: 标签数组列表，每个shape为(X, Y, Z)

    返回:
        combined: (X, Y, Z) 合并后的标签
    """
    if len(label_arrays) == 0:
        raise ValueError("标签数组列表为空")

    shape = label_arrays[0].shape
    combined = np.zeros(shape, dtype=np.uint8)

    for labels in label_arrays:
        mask = labels > 0
        combined[mask] = labels[mask]

    return combined


def create_body_mask_from_ct(
    voxel_world_coords: np.ndarray,
    ct_data: np.ndarray,
    ct_affine: np.ndarray,
    hu_threshold: float = -500.0
) -> np.ndarray:
    """
    根据CT数据的HU值创建体内掩码

    参数:
        voxel_world_coords: (X, Y, Z, 3) 体素中心世界坐标
        ct_data: CT数据（3D数组，HU值）
        ct_affine: CT的仿射矩阵
        hu_threshold: HU阈值

    返回:
        body_mask: (X, Y, Z) 布尔数组，True表示体内
    """
    shape = voxel_world_coords.shape[:3]
    flat_coords = voxel_world_coords.reshape(-1, 3)

    # 世界坐标 -> CT体素坐标
    affine_inv = np.linalg.inv(ct_affine)
    coords_homo = np.hstack([flat_coords, np.ones((len(flat_coords), 1))])
    ct_coords = (affine_inv @ coords_homo.T).T[:, :3]

    indices = np.round(ct_coords).astype(int)
    valid = ((indices >= 0) & (indices < np.array(ct_data.shape))).all(axis=1)

    body_mask = np.zeros(len(flat_coords), dtype=bool)
    valid_indices = indices[valid]

    hu_values = ct_data[valid_indices[:, 0],
                        valid_indices[:, 1],
                        valid_indices[:, 2]]
    body_mask[valid] = hu_values > hu_threshold

    return body_mask.reshape(shape)


def crop_to_body_bbox(
    voxel_labels: np.ndarray,
    body_mask: np.ndarray,
    grid_info: Dict,
    padding_voxels: int = 2
) -> Tuple[np.ndarray, Dict]:
    """
    将体素网格裁剪到body mask的边界框

    参数:
        voxel_labels: (X, Y, Z) 体素标签
        body_mask: (X, Y, Z) 体内掩码
        grid_info: 网格信息字典
        padding_voxels: 边界外保留的体素数

    返回:
        cropped_labels, cropped_grid_info
    """
    body_indices = np.where(body_mask)

    if len(body_indices[0]) == 0:
        warnings.warn("body_mask全为False，无法裁剪，返回原始数据")
        return voxel_labels, grid_info

    min_idx = [int(idx.min()) for idx in body_indices]
    max_idx = [int(idx.max()) for idx in body_indices]

    shape = voxel_labels.shape
    min_idx_padded = [max(0, m - padding_voxels) for m in min_idx]
    max_idx_padded = [min(s - 1, m + padding_voxels) for m, s in zip(max_idx, shape)]

    cropped_labels = voxel_labels[
        min_idx_padded[0]:max_idx_padded[0] + 1,
        min_idx_padded[1]:max_idx_padded[1] + 1,
        min_idx_padded[2]:max_idx_padded[2] + 1
    ].copy()

    voxel_size = np.array(grid_info['voxel_size'])
    old_world_min = np.array(grid_info['world_min'])

    new_world_min = old_world_min + np.array(min_idx_padded) * voxel_size
    new_world_max = old_world_min + np.array([m + 1 for m in max_idx_padded]) * voxel_size
    new_occ_size = list(cropped_labels.shape)

    cropped_grid_info = {
        'world_min': new_world_min,
        'world_max': new_world_max,
        'voxel_size': grid_info['voxel_size'],
        'occ_size': new_occ_size,
        'center': (new_world_min + new_world_max) / 2,
        'max_extent': (new_world_max - new_world_min).max(),
        'mode': grid_info.get('mode', 'fixed_voxel_size'),
        'crop_offset': min_idx_padded,
        'original_occ_size': list(shape),
    }

    return cropped_labels, cropped_grid_info


def apply_body_region_labels(
    voxel_labels: np.ndarray,
    body_mask: np.ndarray,
    inside_body_empty_label: int = 0,
    outside_body_background_label: int = 255
) -> np.ndarray:
    """
    对体素应用体内区域标签

    逻辑:
    - 体内且无器官标签: inside_body_empty_label (0)
    - 有器官标签: 保持原标签（>= 1）
    - 体外无器官: outside_body_background_label (255)
    """
    modified_labels = voxel_labels.copy()

    no_organ = (voxel_labels == 0)

    # 体内无器官 -> inside_body_empty_label (0)
    modified_labels[body_mask & no_organ] = inside_body_empty_label
    # 体外无器官 -> outside_body_background_label (255)
    modified_labels[~body_mask & no_organ] = outside_body_background_label

    return modified_labels
