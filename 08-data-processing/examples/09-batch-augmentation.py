# examples/09-batch-augmentation.py
"""
批量数据增强

对整个数据集进行增强，扩充数据量
"""
import cv2
import os
from pathlib import Path
import albumentations as A
import numpy as np

class DatasetAugmenter:
    """数据集增强器"""

    def __init__(self, augment_fn, output_dir):
        self.augment = augment_fn
        self.output_dir = output_dir
        os.makedirs(f"{output_dir}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/labels", exist_ok=True)

    def load_yolo_labels(self, label_path, img_width, img_height):
        """读取 YOLO 格式标注"""
        bboxes = []
        labels = []

        if not os.path.exists(label_path):
            return bboxes, labels

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    bboxes.append([x_center, y_center, width, height])
                    labels.append(class_id)

        return bboxes, labels

    def save_yolo_labels(self, label_path, bboxes, labels):
        """保存 YOLO 格式标注"""
        with open(label_path, 'w') as f:
            for bbox, label in zip(bboxes, labels):
                f.write(f"{label} {' '.join(map(str, bbox))}\n")

    def augment_dataset(self, input_image_dir, input_label_dir, num_augments=3):
        """
        对整个数据集进行增强

        参数:
            input_image_dir: 输入图片目录
            input_label_dir: 输入标注目录
            num_augments: 每张图片增强几倍
        """
        image_files = list(Path(input_image_dir).glob("*.jpg"))

        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img.shape[:2]

            label_path = f"{input_label_dir}/{img_path.stem}.txt"
            bboxes, labels = self.load_yolo_labels(label_path, width, height)

            if not bboxes:
                cv2.imwrite(f"{self.output_dir}/images/{img_path.name}", img)
                continue

            # 保存原图
            cv2.imwrite(f"{self.output_dir}/images/{img_path.name}", img)
            self.save_yolo_labels(
                f"{self.output_dir}/labels/{img_path.stem}.txt",
                bboxes, labels
            )

            # 生成增强版本
            for i in range(num_augments):
                try:
                    transformed = self.augment(
                        image=img_rgb,
                        bboxes=bboxes,
                        class_labels=labels
                    )

                    aug_img = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
                    aug_bboxes = transformed['bboxes']
                    aug_labels = transformed['class_labels']

                    out_name = f"{img_path.stem}_aug{i}.jpg"
                    cv2.imwrite(f"{self.output_dir}/images/{out_name}", aug_img)
                    self.save_yolo_labels(
                        f"{self.output_dir}/labels/{out_name.replace('.jpg', '.txt')}",
                        aug_bboxes, aug_labels
                    )

                except Exception as e:
                    print(f"增强失败 {img_path}: {e}")

        print(f"增强完成，输出目录：{self.output_dir}")

if __name__ == "__main__":
    from examples_08 import create_defect_specific_augment

    augmenter = DatasetAugmenter(
        augment_fn=create_defect_specific_augment(),
        output_dir="datasets/augmented/"
    )

    augmenter.augment_dataset(
        input_image_dir="datasets/split/images/val",
        input_label_dir="datasets/split/labels/val",
        num_augments=2
    )
