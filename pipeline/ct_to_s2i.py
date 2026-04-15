"""
CT到S2I数据转换管道

流程:
1. 加载CT数据和affine矩阵
2. 生成皮肤mesh -> 深度渲染 -> sensor_pc（世界坐标）
3. 根据sensor_pc边界创建体素网格世界坐标
4. 对每个器官分割：查询体素标签
5. 合并多器官标签（按优先级）+ 区分体内/体外空白
6. 裁剪到人体边界框
7. 保存为 .npz 格式
"""

import numpy as np
from pathlib import Path
from typing import Union, Dict, Optional, Tuple
from dataclasses import dataclass
import warnings

from utils.mesh_generation import load_ct_for_meshing, generate_skin_mesh, load_segmentation
from utils.camera_system import generate_sensor_pointcloud
from utils.voxelization import (
    create_voxel_grid_world_coords,
    label_voxels_from_segmentation,
    combine_organ_labels,
    create_body_mask_from_ct,
    apply_body_region_labels,
    crop_to_body_bbox,
)
from utils.format_converter import save_case_npz, verify_npz_format
from config.data_config import DataConfig
from config.organ_mapping import (
    ORGAN_MAPPING,
    ORGAN_PRIORITY,
    INSIDE_BODY_EMPTY_LABEL,
    get_organ_label,
)


@dataclass
class ConversionResult:
    """转换结果"""
    case_id: str
    success: bool
    sensor_pc_shape: Tuple[int, ...]
    voxel_labels_shape: Tuple[int, ...]
    num_organs: int
    grid_info: Dict
    error: Optional[str] = None


class CTToS2IConverter:
    """CT到S2I数据转换器"""

    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()

    def convert_case(
        self,
        ct_path: Union[str, Path],
        seg_dir: Union[str, Path],
        output_dir: Union[str, Path],
        case_id: str,
        split: str = 'train',
    ) -> ConversionResult:
        """
        转换单个案例

        参数:
            ct_path: CT文件路径
            seg_dir: 分割文件目录
            output_dir: 输出目录
            case_id: 案例ID
            split: 数据集划分 ('train', 'val', 'test')

        返回:
            ConversionResult
        """
        ct_path = Path(ct_path)
        seg_dir = Path(seg_dir)
        output_dir = Path(output_dir)

        try:
            # Step 1: 加载CT数据
            print(f"  [1/6] 加载CT数据...")
            ct_data, ct_affine = load_ct_for_meshing(ct_path)

            # Step 2: 生成皮肤点云
            print(f"  [2/6] 生成皮肤点云...")
            sensor_pc = self._generate_sensor_pointcloud(ct_data, ct_affine)
            print(f"        点云大小: {len(sensor_pc)} 点")

            # Step 3: 创建体素网格
            print(f"  [3/6] 创建体素网格...")
            voxel_coords, grid_info = create_voxel_grid_world_coords(
                sensor_pc,
                padding=self.config.voxel.PADDING,
                voxel_size=self.config.voxel.VOXEL_SIZE,
                max_occ_size=self.config.voxel.MAX_OCC_SIZE,
                min_occ_size=self.config.voxel.MIN_OCC_SIZE
            )
            print(f"        网格大小: {grid_info['occ_size']}")
            print(f"        体素大小: {grid_info['voxel_size']} mm")
            print(f"        模式: {grid_info['mode']}")

            # Step 4: 生成体素标签
            print(f"  [4/6] 生成体素标签...")
            voxel_labels, body_mask, num_organs = self._generate_voxel_labels(
                voxel_coords, seg_dir, ct_data, ct_affine
            )
            print(f"        处理器官数: {num_organs}")
            print(f"        有器官体素: {(voxel_labels >= 1).sum()}")
            print(f"        体内空白: {(voxel_labels == INSIDE_BODY_EMPTY_LABEL).sum()}")

            # Step 5: 裁剪到人体边界框
            print(f"  [5/6] 裁剪到人体边界框...")
            original_shape = voxel_labels.shape
            voxel_labels, grid_info = crop_to_body_bbox(
                voxel_labels, body_mask, grid_info,
                padding_voxels=self.config.voxel.BODY_CROP_PADDING
            )
            print(f"        裁剪前: {original_shape} -> 裁剪后: {voxel_labels.shape}")
            print(f"        体素数减少: {np.prod(original_shape)} -> {np.prod(voxel_labels.shape)} "
                  f"({100 * np.prod(voxel_labels.shape) / np.prod(original_shape):.1f}%)")
            print(f"        体内空白占比: {100 * (voxel_labels == INSIDE_BODY_EMPTY_LABEL).sum() / voxel_labels.size:.1f}%")

            # Step 6: 保存输出
            print(f"  [6/6] 保存输出 (简化格式)...")
            split_dir = output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            save_case_npz(sensor_pc, voxel_labels, grid_info, split_dir / f'{case_id}.npz')

            return ConversionResult(
                case_id=case_id,
                success=True,
                sensor_pc_shape=sensor_pc.shape,
                voxel_labels_shape=voxel_labels.shape,
                num_organs=num_organs,
                grid_info=grid_info
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return ConversionResult(
                case_id=case_id,
                success=False,
                sensor_pc_shape=(0,),
                voxel_labels_shape=(0,),
                num_organs=0,
                grid_info={},
                error=str(e)
            )

    def _generate_sensor_pointcloud(
        self,
        ct_data: np.ndarray,
        ct_affine: np.ndarray
    ) -> np.ndarray:
        """生成传感器点云（皮肤表面）"""
        skin_mesh = generate_skin_mesh(
            ct_data, ct_affine,
            hu_threshold=self.config.SKIN_HU_THRESHOLD
        )
        if skin_mesh is None:
            raise ValueError("无法生成皮肤mesh")

        return generate_sensor_pointcloud(
            skin_mesh,
            camera_distance=self.config.CAMERA_DISTANCE,
            fov=self.config.camera.FOV,
            resolution=self.config.camera.DEPTH_RESOLUTION,
            z_near=self.config.camera.Z_NEAR,
            z_far=self.config.camera.Z_FAR
        )

    def _generate_voxel_labels(
        self,
        voxel_coords: np.ndarray,
        seg_dir: Path,
        ct_data: np.ndarray,
        ct_affine: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        生成体素标签

        标签语义:
        - 0: 体内空白 (INSIDE_BODY_EMPTY_LABEL)
        - 1+: 器官标签

        返回:
            final_labels, body_mask, num_organs
        """
        label_arrays = []
        organ_names = []
        processed_files = set()

        # 按优先级顺序处理器官
        for organ_name in ORGAN_PRIORITY:
            seg_path = seg_dir / f"{organ_name}.nii.gz"
            if not seg_path.exists():
                continue

            organ_label = get_organ_label(organ_name)
            if organ_label < 1:
                continue

            try:
                seg_data, _ = load_segmentation(seg_path)
                labels = label_voxels_from_segmentation(
                    voxel_coords, seg_data, ct_affine, organ_label
                )
                label_arrays.append(labels)
                organ_names.append(organ_name)
                processed_files.add(seg_path.name)
            except Exception as e:
                warnings.warn(f"处理器官 {organ_name} 失败: {e}")

        # 处理不在优先级列表中的器官（动态发现）
        for seg_file in seg_dir.glob('*.nii.gz'):
            if seg_file.name in processed_files:
                continue

            organ_name = seg_file.stem.replace('.nii', '')

            if organ_name in organ_names:
                continue

            organ_label = get_organ_label(organ_name)
            if organ_label < 1:
                if organ_label == -1:
                    warnings.warn(f"未知器官 {organ_name}，请添加到ORGAN_MAPPING")
                continue

            try:
                seg_data, _ = load_segmentation(seg_file)
                labels = label_voxels_from_segmentation(
                    voxel_coords, seg_data, ct_affine, organ_label
                )
                label_arrays.append(labels)
                organ_names.append(organ_name)
                processed_files.add(seg_file.name)
            except Exception as e:
                warnings.warn(f"处理器官 {organ_name} 失败: {e}")

        if len(label_arrays) == 0:
            shape = voxel_coords.shape[:3]
            combined = np.zeros(shape, dtype=np.uint8)
        else:
            combined = combine_organ_labels(label_arrays)

        # 创建体内掩码并应用区域标签
        body_mask = create_body_mask_from_ct(
            voxel_coords, ct_data, ct_affine,
            hu_threshold=self.config.SKIN_HU_THRESHOLD
        )
        final_labels = apply_body_region_labels(
            combined, body_mask,
            inside_body_empty_label=INSIDE_BODY_EMPTY_LABEL
        )

        return final_labels, body_mask, len(organ_names)

    def verify_output(
        self,
        output_dir: Union[str, Path],
        case_id: str,
        split: str = 'train',
    ) -> Dict:
        """验证输出文件"""
        output_dir = Path(output_dir)

        result = {
            'case_id': case_id,
            'valid': True,
            'errors': []
        }

        npz_path = output_dir / split / f'{case_id}.npz'
        if not npz_path.exists():
            result['valid'] = False
            result['errors'].append(f"数据文件不存在: {npz_path}")
            return result

        verify_result = verify_npz_format(npz_path)
        if not verify_result['valid']:
            result['valid'] = False
            result['errors'].append(verify_result.get('error', 'Unknown error'))
        else:
            result['num_points'] = verify_result['num_points']
            result['labels_shape'] = verify_result['labels_shape']
            result['xyz_range'] = verify_result['xyz_range']

        return result
