# examples/02-dataset-organizer.py
"""
数据集整理脚本

功能：将下载的数据集整理为 COCO/YOLO 格式
"""
import os
import shutil
from pathlib import Path

def organize_coco_format(source_dir, output_dir):
    """
    将下载的数据集整理为 COCO/YOLO 格式

    目录结构:
    output_dir/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
    """
    # 创建目录结构
    for split in ['train', 'val']:
        os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

    print(f"整理数据集：{source_dir} -> {output_dir}")
    print("提示：根据具体数据集格式编写解析逻辑")

if __name__ == "__main__":
    organize_coco_format("datasets/raw/", "datasets/organized/")
