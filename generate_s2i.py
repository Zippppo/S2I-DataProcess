r"""
S2I医学数据生成主入口

用法:
    # 处理所有数据（默认4mm体素）
    python generate_s2i.py \
        --input_dir /path/to/ct_data \
        --output_dir ./output/s2i_medical \
        --train_ratio 0.7 \
        --val_ratio 0.15

    # 自定义体素大小（如3mm，更精细）
    python generate_s2i.py \
        --input_dir /path/to/ct_data \
        --output_dir ./output/s2i_medical \
        --voxel_size 3 3 3

    # 处理单个案例（调试用）
    python generate_s2i.py \
        --input_dir /path/to/ct_data \
        --output_dir ./output/test \
        --single_case BDMAP_00000001

    # 验证输出
    python generate_s2i.py \
        --output_dir ./output/s2i_medical \
        --verify

    # 增量处理（跳过已存在的案例）
    python generate_s2i.py \
        --input_dir /path/to/ct_data \
        --output_dir ./output/s2i_medical \
        --skip_existing
"""

import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

from config.data_config import DataConfig
from config.organ_mapping import get_class_names, NUM_CLASSES
from pipeline.ct_to_s2i import CTToS2IConverter
from utils.format_converter import save_dataset_info


def find_cases(input_dir: Path) -> List[Dict]:
    """
    查找所有可处理的案例

    预期目录结构:
        input_dir/
        ├── BDMAP_00000001/
        │   ├── ct.nii.gz
        │   └── segmentations/
        │       ├── liver.nii.gz
        │       └── ...
        └── ...
    """
    cases = []

    for case_dir in sorted(input_dir.iterdir()):
        if not case_dir.is_dir():
            continue

        ct_path = case_dir / 'ct.nii.gz'
        seg_dir = case_dir / 'segmentations'

        if not ct_path.exists():
            print(f"  Warning: {case_dir.name} is missing ct.nii.gz, skipping")
            continue

        if not seg_dir.exists():
            print(f"  Warning: {case_dir.name} is missing the segmentations directory, skipping")
            continue

        seg_files = list(seg_dir.glob('*.nii.gz'))
        if len(seg_files) == 0:
            print(f"  Warning: {case_dir.name} has no segmentation files, skipping")
            continue

        cases.append({
            'case_id': case_dir.name,
            'ct_path': ct_path,
            'seg_dir': seg_dir,
            'num_organs': len(seg_files)
        })

    return cases


def split_dataset(
    cases: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """划分数据集"""
    random.seed(seed)
    cases_shuffled = cases.copy()
    random.shuffle(cases_shuffled)

    n = len(cases_shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return (
        cases_shuffled[:n_train],
        cases_shuffled[n_train:n_train + n_val],
        cases_shuffled[n_train + n_val:]
    )


def process_cases(
    converter: CTToS2IConverter,
    cases: List[Dict],
    output_dir: Path,
    split: str,
    skip_existing: bool = False
) -> List[Dict]:
    """处理一组案例"""
    results = []

    for case in tqdm(cases, desc=f"Processing {split} split"):
        if skip_existing and (output_dir / split / f"{case['case_id']}.npz").exists():
            print(f"\nSkipping existing case: {case['case_id']}")
            results.append({
                'case_id': case['case_id'],
                'success': True,
                'skipped': True,
                'sensor_pc_shape': None,
                'voxel_labels_shape': None,
                'num_organs': None,
                'error': None
            })
            continue

        print(f"\nProcessing case: {case['case_id']}")
        result = converter.convert_case(
            ct_path=case['ct_path'],
            seg_dir=case['seg_dir'],
            output_dir=output_dir,
            case_id=case['case_id'],
            split=split,
        )

        results.append({
            'case_id': result.case_id,
            'success': result.success,
            'skipped': False,
            'sensor_pc_shape': result.sensor_pc_shape,
            'voxel_labels_shape': result.voxel_labels_shape,
            'num_organs': result.num_organs,
            'error': result.error
        })

        if not result.success:
            print(f"  Failed: {result.error}")

    return results


def verify_outputs(output_dir: Path, config: DataConfig) -> Dict:
    """验证所有输出"""
    converter = CTToS2IConverter(config=config)
    results = {'valid': 0, 'invalid': 0, 'errors': []}

    for split in ['train', 'val', 'test']:
        split_dir = output_dir / split
        if not split_dir.exists():
            continue

        for npz_file in tqdm(sorted(split_dir.glob('*.npz')), desc=f"Verifying {split} split"):
            case_id = npz_file.stem
            result = converter.verify_output(output_dir, case_id, split)
            if result['valid']:
                results['valid'] += 1
            else:
                results['invalid'] += 1
                results['errors'].append({
                    'case_id': case_id,
                    'split': split,
                    'errors': result['errors']
                })

    return results


def main():
    parser = argparse.ArgumentParser(description='Generate S2I medical training data')

    parser.add_argument('--input_dir', type=str, help='Input data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--single_case', type=str, help='Process only one case')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training split ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--voxel_size', type=float, nargs=3, default=[4.0, 4.0, 4.0],
                        help='Fixed voxel size in millimeters, default [4, 4, 4]')
    parser.add_argument('--max_occ_size', type=int, default=256,
                        help='Maximum voxel count per axis, default 256')
    parser.add_argument('--min_occ_size', type=int, default=64,
                        help='Minimum voxel count per axis, default 64')
    parser.add_argument('--verify', action='store_true', help='Verify outputs instead of generating them')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip cases whose outputs already exist')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # 创建配置
    config = DataConfig()
    config.voxel.VOXEL_SIZE = args.voxel_size
    config.voxel.MAX_OCC_SIZE = args.max_occ_size
    config.voxel.MIN_OCC_SIZE = args.min_occ_size

    print("Voxel configuration:")
    print(f"  Fixed voxel size: {config.voxel.VOXEL_SIZE} mm")
    print(f"  Grid size range: [{config.voxel.MIN_OCC_SIZE}, {config.voxel.MAX_OCC_SIZE}]")

    if args.verify:
        print("Verifying output data...")
        results = verify_outputs(output_dir, config)
        print("\nVerification results:")
        print(f"  Valid: {results['valid']}")
        print(f"  Invalid: {results['invalid']}")
        if results['errors']:
            print("\nError details:")
            for err in results['errors'][:10]:
                print(f"  {err['case_id']} ({err['split']}): {err['errors']}")
        return

    # 生成模式
    if args.input_dir is None:
        parser.error("Generation mode requires --input_dir")

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: input directory does not exist: {input_dir}")
        return

    converter = CTToS2IConverter(config=config)

    if args.single_case:
        # 处理单个案例
        case_dir = input_dir / args.single_case
        if not case_dir.exists():
            print(f"Error: case directory does not exist: {case_dir}")
            return

        print(f"Processing single case: {args.single_case}")
        result = converter.convert_case(
            ct_path=case_dir / 'ct.nii.gz',
            seg_dir=case_dir / 'segmentations',
            output_dir=output_dir,
            case_id=args.single_case,
            split='train',
        )

        if result.success:
            print("\nSuccess!")
            print(f"  Sensor point cloud shape: {result.sensor_pc_shape}")
            print(f"  Voxel label shape: {result.voxel_labels_shape}")
            print(f"  Number of organs: {result.num_organs}")

            class_names = get_class_names()
            save_dataset_info(
                output_dir, class_names, NUM_CLASSES,
                voxel_size=config.voxel.VOXEL_SIZE,
                max_occ_size=config.voxel.MAX_OCC_SIZE,
                min_occ_size=config.voxel.MIN_OCC_SIZE
            )
            print("  Dataset info saved")
        else:
            print(f"\nFailed: {result.error}")

        # 验证输出
        print("\nVerifying output...")
        verify_result = converter.verify_output(output_dir, args.single_case, 'train')
        if verify_result['valid']:
            print("  Verification passed!")
        else:
            print(f"  Verification failed: {verify_result['errors']}")

    else:
        # 批量处理
        print("Finding cases...")
        cases = find_cases(input_dir)
        print(f"Found {len(cases)} cases")

        if len(cases) == 0:
            print("No processable cases found")
            return

        train_cases, val_cases, test_cases = split_dataset(
            cases,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed
        )
        print("\nDataset split:")
        print(f"  Train: {len(train_cases)}")
        print(f"  Validation: {len(val_cases)}")
        print(f"  Test: {len(test_cases)}")

        # 保存数据集信息
        class_names = get_class_names()
        save_dataset_info(
            output_dir, class_names, NUM_CLASSES,
            voxel_size=config.voxel.VOXEL_SIZE,
            max_occ_size=config.voxel.MAX_OCC_SIZE,
            min_occ_size=config.voxel.MIN_OCC_SIZE
        )
        print(f"Dataset info saved to: {output_dir}/dataset_info.json")

        # 处理每个划分
        all_results = []
        for split_name, split_cases in [('train', train_cases), ('val', val_cases), ('test', test_cases)]:
            if split_cases:
                print(f"\nProcessing {split_name} split...")
                results = process_cases(converter, split_cases, output_dir, split_name, args.skip_existing)
                all_results.extend(results)

        # 统计结果
        success_count = sum(1 for r in all_results if r['success'] and not r.get('skipped'))
        skipped_count = sum(1 for r in all_results if r.get('skipped'))
        fail_count = sum(1 for r in all_results if not r['success'])

        print("\nProcessing complete:")
        print(f"  Succeeded: {success_count}")
        print(f"  Skipped: {skipped_count}")
        print(f"  Failed: {fail_count}")


if __name__ == '__main__':
    main()
