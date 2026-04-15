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
            print(f"  警告: {case_dir.name} 没有ct.nii.gz，跳过")
            continue

        if not seg_dir.exists():
            print(f"  警告: {case_dir.name} 没有segmentations目录，跳过")
            continue

        seg_files = list(seg_dir.glob('*.nii.gz'))
        if len(seg_files) == 0:
            print(f"  警告: {case_dir.name} 没有分割文件，跳过")
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

    for case in tqdm(cases, desc=f"处理{split}集"):
        if skip_existing and (output_dir / split / f"{case['case_id']}.npz").exists():
            print(f"\n跳过已存在: {case['case_id']}")
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

        print(f"\n处理: {case['case_id']}")
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
            print(f"  失败: {result.error}")

    return results


def verify_outputs(output_dir: Path, config: DataConfig) -> Dict:
    """验证所有输出"""
    converter = CTToS2IConverter(config=config)
    results = {'valid': 0, 'invalid': 0, 'errors': []}

    for split in ['train', 'val', 'test']:
        split_dir = output_dir / split
        if not split_dir.exists():
            continue

        for npz_file in tqdm(sorted(split_dir.glob('*.npz')), desc=f"验证{split}集"):
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
    parser = argparse.ArgumentParser(description='生成S2I医学训练数据')

    parser.add_argument('--input_dir', type=str, help='输入数据目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--single_case', type=str, help='只处理单个案例')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='验证集比例')
    parser.add_argument('--voxel_size', type=float, nargs=3, default=[4.0, 4.0, 4.0],
                        help='固定体素大小（毫米），默认 [4, 4, 4]')
    parser.add_argument('--max_occ_size', type=int, default=256,
                        help='单轴最大体素数，默认 256')
    parser.add_argument('--min_occ_size', type=int, default=64,
                        help='单轴最小体素数，默认 64')
    parser.add_argument('--verify', action='store_true', help='验证输出而不是生成')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--skip_existing', action='store_true',
                        help='跳过已存在输出的案例')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # 创建配置
    config = DataConfig()
    config.voxel.VOXEL_SIZE = args.voxel_size
    config.voxel.MAX_OCC_SIZE = args.max_occ_size
    config.voxel.MIN_OCC_SIZE = args.min_occ_size

    print(f"体素配置:")
    print(f"  固定体素大小: {config.voxel.VOXEL_SIZE} mm")
    print(f"  网格大小范围: [{config.voxel.MIN_OCC_SIZE}, {config.voxel.MAX_OCC_SIZE}]")

    if args.verify:
        print(f"验证输出数据...")
        results = verify_outputs(output_dir, config)
        print(f"\n验证结果:")
        print(f"  有效: {results['valid']}")
        print(f"  无效: {results['invalid']}")
        if results['errors']:
            print(f"\n错误详情:")
            for err in results['errors'][:10]:
                print(f"  {err['case_id']} ({err['split']}): {err['errors']}")
        return

    # 生成模式
    if args.input_dir is None:
        parser.error("生成模式需要指定 --input_dir")

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        return

    converter = CTToS2IConverter(config=config)

    if args.single_case:
        # 处理单个案例
        case_dir = input_dir / args.single_case
        if not case_dir.exists():
            print(f"错误: 案例目录不存在: {case_dir}")
            return

        print(f"处理单个案例: {args.single_case}")
        result = converter.convert_case(
            ct_path=case_dir / 'ct.nii.gz',
            seg_dir=case_dir / 'segmentations',
            output_dir=output_dir,
            case_id=args.single_case,
            split='train',
        )

        if result.success:
            print(f"\n成功!")
            print(f"  点云形状: {result.sensor_pc_shape}")
            print(f"  标签形状: {result.voxel_labels_shape}")
            print(f"  器官数量: {result.num_organs}")

            class_names = get_class_names()
            save_dataset_info(
                output_dir, class_names, NUM_CLASSES,
                voxel_size=config.voxel.VOXEL_SIZE,
                max_occ_size=config.voxel.MAX_OCC_SIZE,
                min_occ_size=config.voxel.MIN_OCC_SIZE
            )
            print(f"  数据集信息已保存")
        else:
            print(f"\n失败: {result.error}")

        # 验证输出
        print(f"\n验证输出...")
        verify_result = converter.verify_output(output_dir, args.single_case, 'train')
        if verify_result['valid']:
            print("  验证通过!")
        else:
            print(f"  验证失败: {verify_result['errors']}")

    else:
        # 批量处理
        print(f"查找案例...")
        cases = find_cases(input_dir)
        print(f"找到 {len(cases)} 个案例")

        if len(cases) == 0:
            print("没有找到可处理的案例")
            return

        train_cases, val_cases, test_cases = split_dataset(
            cases,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed
        )
        print(f"\n数据集划分:")
        print(f"  训练集: {len(train_cases)}")
        print(f"  验证集: {len(val_cases)}")
        print(f"  测试集: {len(test_cases)}")

        # 保存数据集信息
        class_names = get_class_names()
        save_dataset_info(
            output_dir, class_names, NUM_CLASSES,
            voxel_size=config.voxel.VOXEL_SIZE,
            max_occ_size=config.voxel.MAX_OCC_SIZE,
            min_occ_size=config.voxel.MIN_OCC_SIZE
        )
        print(f"数据集信息已保存到: {output_dir}/dataset_info.json")

        # 处理每个划分
        all_results = []
        for split_name, split_cases in [('train', train_cases), ('val', val_cases), ('test', test_cases)]:
            if split_cases:
                print(f"\n处理{split_name}集...")
                results = process_cases(converter, split_cases, output_dir, split_name, args.skip_existing)
                all_results.extend(results)

        # 统计结果
        success_count = sum(1 for r in all_results if r['success'] and not r.get('skipped'))
        skipped_count = sum(1 for r in all_results if r.get('skipped'))
        fail_count = sum(1 for r in all_results if not r['success'])

        print(f"\n处理完成:")
        print(f"  成功: {success_count}")
        print(f"  跳过: {skipped_count}")
        print(f"  失败: {fail_count}")


if __name__ == '__main__':
    main()
