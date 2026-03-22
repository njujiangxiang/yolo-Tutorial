# examples/06-split-dataset.py
"""
分层抽样划分数据集

保证每个类别在训练集/验证集/测试集中的比例相同
"""
import os
import random
import shutil
from pathlib import Path
from collections import defaultdict

def stratified_split(input_dir, output_dir, train_ratio=0.7, val_ratio=0.2, seed=42):
    """
    分层抽样划分数据集
    """
    random.seed(seed)

    # 创建输出目录
    for split in ['train', 'val', 'test']:
        os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

    # 按类别分组
    class_images = defaultdict(list)

    label_dir = Path(f"{input_dir}/labels")
    if label_dir.exists():
        for label_file in label_dir.glob("*.txt"):
            with open(label_file, 'r') as f:
                labels = [line.strip().split()[0] for line in f if line.strip()]

            if labels:
                class_id = labels[0]
                image_name = label_file.stem
                class_images[class_id].append(image_name)

    # 对每个类别进行划分
    train_count = val_count = test_count = 0

    for class_id, images in class_images.items():
        random.shuffle(images)

        n = len(images)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]

        # 复制图片和标注
        for split, imgs in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
            for img_name in imgs:
                src_img = f"{input_dir}/images/{img_name}.jpg"
                dst_img = f"{output_dir}/images/{split}/{img_name}.jpg"
                if os.path.exists(src_img):
                    shutil.copy(src_img, dst_img)

                src_lbl = f"{input_dir}/labels/{img_name}.txt"
                dst_lbl = f"{output_dir}/labels/{split}/{img_name}.txt"
                if os.path.exists(src_lbl):
                    shutil.copy(src_lbl, dst_lbl)

                if split == 'train':
                    train_count += 1
                elif split == 'val':
                    val_count += 1
                else:
                    test_count += 1

    print(f"划分完成:")
    print(f"  训练集：{train_count} 张")
    print(f"  验证集：{val_count} 张")
    print(f"  测试集：{test_count} 张")

if __name__ == "__main__":
    stratified_split("datasets/labeled/", "datasets/split/")
