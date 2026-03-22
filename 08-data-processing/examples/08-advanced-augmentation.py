# examples/08-advanced-augmentation.py
"""
高级数据增强 pipeline

针对缺陷检测的专用增强
"""
import albumentations as A

def create_advanced_augment():
    """创建高级数据增强 pipeline"""
    return A.Compose([
        A.CoarseDropout(
            max_holes=8,
            max_height=50,
            max_width=50,
            p=0.5
        ),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
        ], p=0.3),
        A.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1),
            num_shadows_lower=1,
            num_shadows_upper=2,
            p=0.2
        ),
        A.Perspective(
            scale=(0.05, 0.1),
            p=0.2
        ),
        A.ElasticTransform(
            alpha=1,
            sigma=50,
            alpha_affine=50,
            p=0.1
        ),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels']
    ))

def create_defect_specific_augment():
    """
    针对缺陷检测的专用增强

    缺陷检测特点:
    - 缺陷通常很小
    - 缺陷方向不重要
    - 颜色可能很重要
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.3
        ),
        A.Blur(blur_limit=3, p=0.15),
        A.Sharpen(alpha=(0.2, 0.5), p=0.3),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels']
    ))

if __name__ == "__main__":
    augment = create_defect_specific_augment()
    print("缺陷检测数据增强 pipeline 创建完成")
