"""
数据配置模块 - VoxDet医学数据管道配置

设计决策:
1. 不需要归一化，直接使用世界坐标（毫米）
2. 体素网格范围基于sensor_pc边界确定
3. 固定体素物理大小，网格尺寸动态计算
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class CameraConfig:
    """相机配置参数"""
    FOV: float = 60.0
    DEPTH_RESOLUTION: Tuple[int, int] = (512, 512)
    Z_NEAR: float = 100.0
    Z_FAR: float = 2000.0


@dataclass
class VoxelConfig:
    """体素化配置

    固定体素物理尺寸，网格大小根据数据动态计算:
    - VOXEL_SIZE: 固定体素大小（毫米），默认 [4, 4, 4]
    - MAX_OCC_SIZE: 网格大小上限，防止内存溢出
    - MIN_OCC_SIZE: 网格大小下限，保证最小分辨率
    """
    VOXEL_SIZE: List[float] = field(default_factory=lambda: [4.0, 4.0, 4.0])
    MAX_OCC_SIZE: int = 256
    MIN_OCC_SIZE: int = 64
    PADDING: float = 0.05
    BODY_CROP_PADDING: int = 2


@dataclass
class DataConfig:
    """完整数据配置"""
    camera: CameraConfig = field(default_factory=CameraConfig)
    voxel: VoxelConfig = field(default_factory=VoxelConfig)
    SKIN_HU_THRESHOLD: float = -500.0
    CAMERA_DISTANCE: float = 800.0
