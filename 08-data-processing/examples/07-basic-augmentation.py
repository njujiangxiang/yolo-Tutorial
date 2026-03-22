# examples/07-basic-augmentation.py
"""
基础数据增强 pipeline

包含：翻转、旋转、亮度调整、模糊
"""
import cv2
import numpy as np
import albumentations as A

def create_basic_augment():
    """创建基础数据增强 pipeline"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.Blur(blur_limit=3, p=0.1),
        A.MotionBlur(blur_limit=3, p=0.1),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.1),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels']
    ))

def augment_image(image, bboxes, class_labels, augment_fn):
    """
    对图片和标注进行增强
    """
    transformed = augment_fn(
        image=image,
        bboxes=bboxes,
        class_labels=class_labels
    )
    return transformed['image'], transformed['bboxes'], transformed['class_labels']

if __name__ == "__main__":
    augment = create_basic_augment()
    print("基础数据增强 pipeline 创建完成")
