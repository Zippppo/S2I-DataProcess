"""
器官标签映射模块 - S2I医学数据

定义器官类别ID和处理优先级。
基于TotalSegmentator v2，包含所有类别，不做任何融合或忽略。

标签约定:
- 0: inside_body_empty，体内但无器官标签
- 1-122: 器官标签
- 255: outside_body_background，体外背景特殊值

训练类别数仍为 123（0-122）。
255 是输出格式中的特殊背景值，不计入训练类别。

作者：rongkun
日期：2026-4
"""

from typing import Dict, List


# 器官类别映射
# 0-122: 训练类别
# 255: 体外背景特殊值
ORGAN_MAPPING: Dict[str, int] = {
    # ==================== 特殊标签 ====================
    'inside_body_empty': 0, # 体内空白（无器官标注但在体内）
    'outside_body_background': 255, # 体外背景（不计入训练类别）

    # ==================== 实质器官 (1-13) ====================
    'liver': 1,
    'spleen': 2,
    'kidney_left': 3,
    'kidney_right': 4,
    'stomach': 5,
    'pancreas': 6,
    'gallbladder': 7,
    'urinary_bladder': 8,
    'prostate': 9,
    'heart': 10,
    'brain': 11,
    'thyroid_gland': 12,
    'spinal_cord': 13,

    # ==================== 肺叶 (14-18) ====================
    'lung_upper_lobe_left': 14,
    'lung_lower_lobe_left': 15,
    'lung_upper_lobe_right': 16,
    'lung_middle_lobe_right': 17,
    'lung_lower_lobe_right': 18,

    # ==================== 消化道 (19-23) ====================
    'esophagus': 19,
    'trachea': 20,
    'small_bowel': 21,
    'duodenum': 22,
    'colon': 23,

    # ==================== 肾上腺 (24-25) ====================
    'adrenal_gland_left': 24,
    'adrenal_gland_right': 25,

    # ==================== 脊椎 (26-51) ====================
    # 颈椎 C1-C7
    'vertebrae_C1': 26,
    'vertebrae_C2': 27,
    'vertebrae_C3': 28,
    'vertebrae_C4': 29,
    'vertebrae_C5': 30,
    'vertebrae_C6': 31,
    'vertebrae_C7': 32,
    # 胸椎 T1-T12
    'vertebrae_T1': 33,
    'vertebrae_T2': 34,
    'vertebrae_T3': 35,
    'vertebrae_T4': 36,
    'vertebrae_T5': 37,
    'vertebrae_T6': 38,
    'vertebrae_T7': 39,
    'vertebrae_T8': 40,
    'vertebrae_T9': 41,
    'vertebrae_T10': 42,
    'vertebrae_T11': 43,
    'vertebrae_T12': 44,
    # 腰椎 L1-L5
    'vertebrae_L1': 45,
    'vertebrae_L2': 46,
    'vertebrae_L3': 47,
    'vertebrae_L4': 48,
    'vertebrae_L5': 49,
    # 骶椎
    'vertebrae_S1': 50,
    'sacrum': 51,

    # ==================== 肋骨 (52-75) ====================
    # 左侧肋骨 1-12
    'rib_left_1': 52,
    'rib_left_2': 53,
    'rib_left_3': 54,
    'rib_left_4': 55,
    'rib_left_5': 56,
    'rib_left_6': 57,
    'rib_left_7': 58,
    'rib_left_8': 59,
    'rib_left_9': 60,
    'rib_left_10': 61,
    'rib_left_11': 62,
    'rib_left_12': 63,
    # 右侧肋骨 1-12
    'rib_right_1': 64,
    'rib_right_2': 65,
    'rib_right_3': 66,
    'rib_right_4': 67,
    'rib_right_5': 68,
    'rib_right_6': 69,
    'rib_right_7': 70,
    'rib_right_8': 71,
    'rib_right_9': 72,
    'rib_right_10': 73,
    'rib_right_11': 74,
    'rib_right_12': 75,

    # ==================== 其他骨骼 (76-88) ====================
    'skull': 76,
    'sternum': 77,
    'costal_cartilages': 78,
    'scapula_left': 79,
    'scapula_right': 80,
    'clavicula_left': 81,
    'clavicula_right': 82,
    'humerus_left': 83,
    'humerus_right': 84,
    'hip_left': 85,
    'hip_right': 86,
    'femur_left': 87,
    'femur_right': 88,

    # ==================== 肌肉 (89-98) ====================
    'gluteus_maximus_left': 89,
    'gluteus_maximus_right': 90,
    'gluteus_medius_left': 91,
    'gluteus_medius_right': 92,
    'gluteus_minimus_left': 93,
    'gluteus_minimus_right': 94,
    'autochthon_left': 95,
    'autochthon_right': 96,
    'iliopsoas_left': 97,
    'iliopsoas_right': 98,

    # ==================== 血管-动脉 (99-107) ====================
    'aorta': 99,
    'brachiocephalic_trunk': 100,
    'coronary_arteries': 101,
    'subclavian_artery_left': 102,
    'subclavian_artery_right': 103,
    'common_carotid_artery_left': 104,
    'common_carotid_artery_right': 105,
    'iliac_artery_left': 106,
    'iliac_artery_right': 107,

    # ==================== 血管-静脉及其他 (108-116) ====================
    'pulmonary_vein': 108,
    'atrial_appendage_left': 109,
    'inferior_vena_cava': 110,
    'superior_vena_cava': 111,
    'portal_vein_and_splenic_vein': 112,
    'brachiocephalic_vein_left': 113,
    'brachiocephalic_vein_right': 114,
    'iliac_vena_left': 115,
    'iliac_vena_right': 116,

    # ==================== 病变 (117-118) ====================
    'kidney_cyst_left': 117,
    'kidney_cyst_right': 118,

    # ==================== 体脂/体成分 (119-122) ====================
    'subcutaneous_fat': 119,
    'torso_fat': 120,
    'intermuscular_fat': 121,
    'skeletal_muscle': 122,
}

# 反向映射：ID -> 器官名称
LABEL_TO_ORGAN: Dict[int, str] = {v: k for k, v in ORGAN_MAPPING.items()}

# 类别数（仅统计训练类别，不含 255 特殊背景值）
NUM_CLASSES: int = 123  # 0=inside_body_empty, 1-122=organs

# 特殊标签常量
INSIDE_BODY_EMPTY_LABEL: int = 0  # 体内空白（无器官标注）
OUTSIDE_BODY_BACKGROUND_LABEL: int = 255  # 体外背景特殊值

# 器官处理优先级（后处理的会覆盖先处理的）
# 设计原则：大/弥散区域先处理（可被小器官覆盖），肋骨最后处理（保护肋骨不被覆盖）
ORGAN_PRIORITY: List[str] = [
    # ===== 第一优先级：体脂/体成分（最低，弥散区域）=====
    'subcutaneous_fat', 'torso_fat', 'intermuscular_fat',
    'skeletal_muscle',

    # ===== 第二优先级：大型骨骼（先处理，可被覆盖）=====
    'skull', 'sternum', 'costal_cartilages',
    'hip_left', 'hip_right', 'femur_left', 'femur_right',
    'humerus_left', 'humerus_right',
    'scapula_left', 'scapula_right',
    'clavicula_left', 'clavicula_right',

    # 脊椎（各椎骨独立类别）
    'vertebrae_C1', 'vertebrae_C2', 'vertebrae_C3', 'vertebrae_C4',
    'vertebrae_C5', 'vertebrae_C6', 'vertebrae_C7',
    'vertebrae_T1', 'vertebrae_T2', 'vertebrae_T3', 'vertebrae_T4',
    'vertebrae_T5', 'vertebrae_T6', 'vertebrae_T7', 'vertebrae_T8',
    'vertebrae_T9', 'vertebrae_T10', 'vertebrae_T11', 'vertebrae_T12',
    'vertebrae_L1', 'vertebrae_L2', 'vertebrae_L3', 'vertebrae_L4',
    'vertebrae_L5',
    'vertebrae_S1', 'sacrum',

    # 肌肉
    'gluteus_maximus_left', 'gluteus_maximus_right',
    'gluteus_medius_left', 'gluteus_medius_right',
    'gluteus_minimus_left', 'gluteus_minimus_right',
    'autochthon_left', 'autochthon_right',
    'iliopsoas_left', 'iliopsoas_right',

    # ===== 第三优先级：大型实质器官 =====
    'lung_upper_lobe_left', 'lung_lower_lobe_left',
    'lung_upper_lobe_right', 'lung_middle_lobe_right', 'lung_lower_lobe_right',
    'liver', 'stomach', 'colon', 'small_bowel',
    'brain',

    # ===== 第四优先级：血管（细长结构，需要保护）=====
    'aorta', 'brachiocephalic_trunk', 'coronary_arteries',
    'subclavian_artery_left', 'subclavian_artery_right',
    'common_carotid_artery_left', 'common_carotid_artery_right',
    'iliac_artery_left', 'iliac_artery_right',
    'pulmonary_vein', 'atrial_appendage_left',
    'inferior_vena_cava', 'superior_vena_cava',
    'portal_vein_and_splenic_vein',
    'brachiocephalic_vein_left', 'brachiocephalic_vein_right',
    'iliac_vena_left', 'iliac_vena_right',

    # ===== 第五优先级：中型器官 =====
    'spleen', 'kidney_left', 'kidney_right',
    'pancreas', 'urinary_bladder', 'heart', 'prostate',

    # ===== 第六优先级：小型器官和病变 =====
    'gallbladder', 'duodenum', 'esophagus', 'trachea',
    'adrenal_gland_left', 'adrenal_gland_right',
    'thyroid_gland', 'spinal_cord',
    'kidney_cyst_left', 'kidney_cyst_right',

    # ===== 第七优先级：肋骨（最高优先级，最后处理，不会被覆盖）=====
    'rib_left_1', 'rib_left_2', 'rib_left_3', 'rib_left_4',
    'rib_left_5', 'rib_left_6', 'rib_left_7', 'rib_left_8',
    'rib_left_9', 'rib_left_10', 'rib_left_11', 'rib_left_12',
    'rib_right_1', 'rib_right_2', 'rib_right_3', 'rib_right_4',
    'rib_right_5', 'rib_right_6', 'rib_right_7', 'rib_right_8',
    'rib_right_9', 'rib_right_10', 'rib_right_11', 'rib_right_12',
]


def get_organ_label(organ_name: str) -> int:
    """
    获取器官标签

    Args:
        organ_name: 器官名称

    Returns:
        标签ID，如果不存在返回-1
    """
    return ORGAN_MAPPING.get(organ_name, -1)


def get_organ_name(label: int) -> str:
    """
    获取器官名称

    Args:
        label: 标签ID

    Returns:
        器官名称，如果不存在返回'unknown'
    """
    return LABEL_TO_ORGAN.get(label, 'unknown')


def get_class_names() -> List[str]:
    """获取训练类别名称列表（按ID排序，范围 0-122）"""
    return [get_organ_name(i) for i in range(NUM_CLASSES)]
