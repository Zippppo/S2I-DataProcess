"""
内存Mesh生成模块 - 从CT/分割数据生成mesh（不保存到磁盘）

核心功能：
1. 从CT数据生成皮肤mesh（基于HU阈值）
2. 从分割文件生成器官mesh
3. 全部在内存中处理，返回trimesh对象
4. 支持affine矩阵坐标转换

复用自CT2PointCloud项目

作者：rongkun
日期：2025-12
"""

import numpy as np
import nibabel as nib
import trimesh
from pathlib import Path
from typing import Union, Optional, Tuple
import warnings

try:
    from skimage import measure
except ImportError:
    raise ImportError("需要安装scikit-image: pip install scikit-image")


def marching_cubes_memory(
    volume: np.ndarray,
    level: float = 0.5,
    spacing: tuple = (1.0, 1.0, 1.0)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Marching Cubes算法（内存版本）

    Args:
        volume: np.ndarray - 3D体数据，shape (D, H, W)
        level: float - 等值面阈值
        spacing: tuple - 体素间距（用于缩放）

    Returns:
        vertices: np.ndarray - 顶点坐标，shape (N, 3)
        faces: np.ndarray - 三角形索引，shape (M, 3)
    """
    try:
        vertices, faces, normals, values = measure.marching_cubes(
            volume,
            level=level,
            spacing=spacing,
            allow_degenerate=False
        )
        return vertices, faces
    except Exception as e:
        warnings.warn(f"Marching Cubes失败: {e}")
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)


def apply_affine_to_vertices(
    vertices: np.ndarray,
    affine: np.ndarray
) -> np.ndarray:
    """
    对顶点应用affine变换（体素坐标 → 世界坐标）

    Args:
        vertices: np.ndarray - 体素坐标顶点，shape (N, 3)
        affine: np.ndarray - 4x4 affine矩阵

    Returns:
        vertices_world: np.ndarray - 世界坐标顶点，shape (N, 3)
    """
    # 构造齐次坐标 (N, 4)
    ones = np.ones((vertices.shape[0], 1))
    vertices_homogeneous = np.hstack([vertices, ones])

    # 应用affine变换: vertices_world = affine @ vertices_voxel^T
    vertices_world = (affine @ vertices_homogeneous.T).T[:, :3]

    return vertices_world


def generate_mesh_from_ct_memory(
    ct_data: np.ndarray,
    ct_affine: np.ndarray,
    hu_threshold: float,
    level: float = 0.5
) -> Optional[trimesh.Trimesh]:
    """
    从CT数据生成mesh（通用函数）

    Args:
        ct_data: np.ndarray - CT数据，shape (D, H, W)
        ct_affine: np.ndarray - 4x4 affine矩阵
        hu_threshold: float - HU阈值
        level: float - Marching Cubes阈值

    Returns:
        mesh: trimesh.Trimesh - 生成的mesh，失败返回None
    """
    # 1. 阈值化
    mask = (ct_data > hu_threshold).astype(np.uint8)

    # 2. 检查是否有数据
    if np.sum(mask) == 0:
        warnings.warn(f"CT数据中没有HU值 > {hu_threshold} 的体素")
        return None

    # 3. Marching Cubes（体素空间）
    vertices, faces = marching_cubes_memory(mask, level=level, spacing=(1, 1, 1))

    if len(vertices) == 0:
        warnings.warn("Marching Cubes未生成任何顶点")
        return None

    # 4. 应用affine变换（体素 → 世界坐标）
    vertices_world = apply_affine_to_vertices(vertices, ct_affine)

    # 5. 创建trimesh对象
    mesh = trimesh.Trimesh(vertices=vertices_world, faces=faces)

    return mesh


def generate_skin_mesh(
    ct_data: np.ndarray,
    ct_affine: np.ndarray,
    hu_threshold: float = -500.0
) -> Optional[trimesh.Trimesh]:
    """
    从CT数据生成皮肤mesh

    Args:
        ct_data: np.ndarray - CT数据，shape (D, H, W)
        ct_affine: np.ndarray - 4x4 affine矩阵
        hu_threshold: float - HU阈值（默认-500）

    Returns:
        mesh: trimesh.Trimesh - 皮肤mesh
    """
    return generate_mesh_from_ct_memory(ct_data, ct_affine, hu_threshold)


def load_ct_for_meshing(
    ct_path: Union[str, Path]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载CT数据用于mesh生成

    Args:
        ct_path: str or Path - CT文件路径（.nii.gz）

    Returns:
        ct_data: np.ndarray - CT数据，shape (D, H, W)
        ct_affine: np.ndarray - 4x4 affine矩阵
    """
    ct_path = Path(ct_path)

    if not ct_path.exists():
        raise FileNotFoundError(f"CT文件不存在: {ct_path}")

    # 加载NIfTI文件
    nii_img = nib.load(str(ct_path))
    ct_data = nii_img.get_fdata()
    ct_affine = nii_img.affine

    return ct_data, ct_affine


def load_segmentation(
    seg_path: Union[str, Path]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载分割数据

    Args:
        seg_path: str or Path - 分割文件路径（.nii.gz）

    Returns:
        seg_data: np.ndarray - 分割数据，shape (D, H, W)
        seg_affine: np.ndarray - 4x4 affine矩阵
    """
    seg_path = Path(seg_path)

    if not seg_path.exists():
        raise FileNotFoundError(f"分割文件不存在: {seg_path}")

    nii_img = nib.load(str(seg_path))
    seg_data = nii_img.get_fdata()
    seg_affine = nii_img.affine

    return seg_data, seg_affine
