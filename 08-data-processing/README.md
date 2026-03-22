# 08-Data Processing - 数据处理

> 学习如何清洗、增强和准备 YOLO 训练数据

---

## 🎯 学习目标

完成本章后，你将能够：

- ✅ 图像预处理基础操作
- ✅ 数据清洗和去重
- ✅ 训练集/验证集/测试集划分
- ✅ 实现数据增强 pipeline
- ✅ YOLO 格式转换
- ✅ 生成数据质量报告

---

## 📊 数据处理流程

```
原始数据 → 数据清洗 → 数据标注 → 格式转换 → 数据增强 → 训练数据
    ↓           ↓           ↓           ↓           ↓         ↓
  收集的图片   删除模糊    LabelImg    COCO/YOLO   增强变换   data.yaml
           重复图片      标注       转换        扩充数据    配置完成
```

---

## 1️⃣ 图像预处理

### 1.1 调整尺寸

保持宽高比的调整方法：

```python
# examples/01-resize-images.py
import cv2
import os
from pathlib import Path

def resize_with_aspect_ratio(image_path, output_path, target_size=640):
    """
    调整图片尺寸，保持宽高比

    参数:
        image_path: 输入图片路径
        output_path: 输出图片路径
        target_size: 目标尺寸（长边）
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片：{image_path}")
        return

    height, width = img.shape[:2]

    # 计算缩放比例
    scale = target_size / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)

    # 调整尺寸
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # 保存（添加灰边填充到正方形）
    square_img = cv2.copyMakeBorder(
        resized,
        (target_size - new_height) // 2,
        target_size - new_height - (target_size - new_height) // 2,
        (target_size - new_width) // 2,
        target_size - new_width - (target_size - new_width) // 2,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)  # YOLO 默认灰色
    )

    cv2.imwrite(output_path, square_img)
    print(f"处理完成：{output_path}")

def batch_resize(input_dir, output_dir, target_size=640):
    """批量处理图片"""
    os.makedirs(output_dir, exist_ok=True)

    image_files = list(Path(input_dir).glob("*.jpg")) + \
                  list(Path(input_dir).glob("*.png"))

    for img_path in image_files:
        resize_with_aspect_ratio(
            str(img_path),
            f"{output_dir}/{img_path.stem}.jpg",
            target_size
        )

if __name__ == "__main__":
    batch_resize("datasets/raw/", "datasets/resized/", target_size=640)
```

### 1.2 归一化

```python
# 归一化到 [0, 1]
normalized = image / 255.0

# 标准化（均值为 0，标准差为 1）
mean = np.mean(image, axis=(0, 1, 2))
std = np.std(image, axis=(0, 1, 2))
normalized = (image - mean) / std
```

### 1.3 色彩空间转换

```python
# examples/02-color-space.py
import cv2

# BGR 转 RGB
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# BGR 转 HSV（用于颜色分割）
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

# BGR 转灰度
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
```

---

## 2️⃣ 数据清洗

### 2.1 删除模糊图片

```python
# examples/03-remove-blur.py
import cv2
import os
from pathlib import Path

def calculate_blur_score(image_path):
    """
    计算图片模糊度（拉普拉斯方差）

    返回值越小，图片越模糊
    """
    img = cv2.imread(image_path)
    if img is None:
        return 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance

def remove_blur_images(input_dir, output_dir, threshold=100):
    """
    删除模糊图片

    参数:
        threshold: 模糊度阈值，低于此值的图片被删除
    """
    os.makedirs(output_dir, exist_ok=True)

    kept = 0
    removed = 0

    for img_path in Path(input_dir).glob("*.jpg"):
        score = calculate_blur_score(str(img_path))

        if score >= threshold:
            # 清晰图片，保留
            os.system(f"cp {img_path} {output_dir}/")
            kept += 1
        else:
            # 模糊图片，删除
            print(f"删除模糊图片：{img_path} (分数：{score:.2f})")
            removed += 1

    print(f"处理完成：保留 {kept} 张，删除 {removed} 张")

if __name__ == "__main__":
    remove_blur_images("datasets/raw/", "datasets/cleaned/", threshold=100)
```

### 2.2 删除重复图片

```python
# examples/04-remove-duplicates.py
import cv2
import numpy as np np
from pathlib import Path
import os
from PIL import Image
import imagehash

def find_duplicate_images(input_dir, threshold=0.95):
    """
    查找重复图片（使用感知哈希）

    返回重复图片列表
    """
    hashes = {}
    duplicates = []

    for img_path in Path(input_dir).glob("*.jpg"):
        try:
            with Image.open(img_path) as img:
                # 计算感知哈希
                phash = imagehash.phash(img)

                # 检查是否与已有图片重复
                for existing_path, existing_hash in hashes.items():
                    similarity = 1 - (phash - existing_hash) / len(phash) ** 2

                    if similarity >= threshold:
                        duplicates.append((str(img_path), str(existing_path), similarity))
                        break

                if not duplicates or duplicates[-1][0] != str(img_path):
                    hashes[str(img_path)] = phash

        except Exception as e:
            print(f"处理失败 {img_path}: {e}")

    return duplicates

def remove_duplicates(input_dir, output_dir):
    """删除重复图片，保留一份"""
    os.makedirs(output_dir, exist_ok=True)

    duplicates = find_duplicate_images(input_dir)
    removed_paths = set([d[0] for d in duplicates])

    kept = 0
    for img_path in Path(input_dir).glob("*.jpg"):
        if str(img_path) not in removed_paths:
            os.system(f"cp {img_path} {output_dir}/")
            kept += 1

    print(f"删除 {len(removed_paths)} 张重复图片，保留 {kept} 张")

if __name__ == "__main__":
    # 先安装：pip install Pillow imagehash
    remove_duplicates("datasets/cleaned/", "datasets/deduped/")
```

### 2.3 数据清洗检查清单

```python
# examples/05-data-quality-check.py
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def generate_quality_report(input_dir, report_path="quality_report.txt"):
    """
    生成数据质量报告

    检查项目:
    - 图片数量
    - 图片尺寸分布
    - 模糊度分布
    - 亮度分布
    """
    results = {
        'count': 0,
        'sizes': [],
        'blur_scores': [],
        'brightness': []
    }

    for img_path in Path(input_dir).glob("*.jpg"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        results['count'] += 1
        results['sizes'].append(img.shape[:2])

        # 模糊度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results['blur_scores'].append(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 亮度
        results['brightness'].append(np.mean(gray))

    # 生成报告
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("数据质量报告\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"图片总数：{results['count']}\n")

        if results['sizes']:
            sizes = np.array(results['sizes'])
            f.write(f"平均尺寸：{sizes.mean(axis=0).astype(int)}\n")
            f.write(f"最小尺寸：{sizes.min(axis=0)}\n")
            f.write(f"最大尺寸：{sizes.max(axis=0)}\n\n")

        if results['blur_scores']:
            blur = np.array(results['blur_scores'])
            f.write(f"平均模糊度：{blur.mean():.2f}\n")
            f.write(f"最小模糊度：{blur.min():.2f}\n")
            f.write(f"最大模糊度：{blur.max():.2f}\n")
            f.write(f"模糊图片数（<100）: {(blur < 100).sum()}\n\n")

        if results['brightness']:
            bright = np.array(results['brightness'])
            f.write(f"平均亮度：{bright.mean():.2f}\n")
            f.write(f"最暗：{bright.min():.2f}\n")
            f.write(f"最亮：{bright.max():.2f}\n")

    print(f"报告已生成：{report_path}")
    return results

if __name__ == "__main__":
    generate_quality_report("datasets/deduped/")
```

---

## 3️⃣ 数据划分

### 3.1 训练集/验证集/测试集比例

| 场景 | 训练集 | 验证集 | 测试集 | 说明 |
|------|--------|--------|--------|------|
| 小数据集（<500） | 70% | 20% | 10% | 保证足够训练数据 |
| 中等数据集 | 80% | 10% | 10% | 常用比例 |
| 大数据集（>10000） | 98% | 1% | 1% | 验证/测试集足够 |

### 3.2 分层抽样划分

```python
# examples/06-split-dataset.py
import os
import random
import shutil
from pathlib import Path
from collections import defaultdict

def stratified_split(input_dir, output_dir, train_ratio=0.7, val_ratio=0.2, seed=42):
    """
    分层抽样划分数据集

    保证每个类别在训练集/验证集/测试集中的比例相同
    """
    random.seed(seed)

    # 创建输出目录
    for split in ['train', 'val', 'test']:
        os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

    # 按类别分组
    # 假设标注文件命名：image1.txt, image2.txt...
    # 需要根据标注文件统计类别
    class_images = defaultdict(list)

    # 读取所有标注文件，按类别分组
    label_dir = Path(f"{input_dir}/labels")
    if label_dir.exists():
        for label_file in label_dir.glob("*.txt"):
            # 读取标注，获取主要类别
            with open(label_file, 'r') as f:
                labels = [line.strip().split()[0] for line in f if line.strip()]

            if labels:
                # 使用第一个类别作为分组依据
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

        # 复制图片
        for split, imgs in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
            for img_name in imgs:
                # 复制图片
                src_img = f"{input_dir}/images/{img_name}.jpg"
                dst_img = f"{output_dir}/images/{split}/{img_name}.jpg"
                if os.path.exists(src_img):
                    shutil.copy(src_img, dst_img)

                # 复制标注
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
```

### 3.3 避免数据泄露

**数据泄露**：训练集和测试集有重复或相似数据，导致评估结果虚高

**防止方法：**
- 同一场景/同一批次的图片只能在一个集合中
- 删除高度相似的图片（使用图像相似度检测）
- 时间序列数据要按时间划分

---

## 4️⃣ 数据增强

### 4.1 基础增强

```python
# examples/07-basic-augmentation.py
import cv2
import numpy as np
import albumentations as A

def create_basic_augment():
    """
    创建基础数据增强 pipeline
    """
    return A.Compose([
        # 几何变换
        A.HorizontalFlip(p=0.5),  # 水平翻转 50% 概率
        A.VerticalFlip(p=0.2),    # 垂直翻转 20% 概率
        A.RandomRotate90(p=0.5),  # 随机旋转 90 度

        # 光照变化
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),

        # 模糊
        A.Blur(blur_limit=3, p=0.1),
        A.MotionBlur(blur_limit=3, p=0.1),

        # 噪声
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.1),
    ], bbox_params=A.BboxParams(
        format='yolo',  # YOLO 格式 [x_center, y_center, width, height]
        label_fields=['class_labels']
    ))

def augment_image(image, bboxes, class_labels, augment_fn):
    """
    对图片和标注进行增强

    参数:
        image: 图片数组
        bboxes: 标注框列表 [[x_center, y_center, width, height], ...]
        class_labels: 类别标签列表
        augment_fn: 增强函数

    返回:
        augmented_image, augmented_bboxes, augmented_labels
    """
    transformed = augment_fn(
        image=image,
        bboxes=bboxes,
        class_labels=class_labels
    )
    return transformed['image'], transformed['bboxes'], transformed['class_labels']

if __name__ == "__main__":
    # 使用示例
    augment = create_basic_augment()

    # 读取图片
    img = cv2.imread("datasets/sample.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 示例标注
    bboxes = [[0.5, 0.5, 0.3, 0.3]]  # [x_center, y_center, w, h]
    labels = [0]

    # 应用增强
    aug_img, aug_bbox, aug_label = augment_image(img, bboxes, labels, augment)

    # 保存结果
    cv2.imwrite("augmented.jpg", cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
```

### 4.2 高级增强

```python
# examples/08-advanced-augmentation.py
import albumentations as A

def create_advanced_augment():
    """
    创建高级数据增强 pipeline

    包含 YOLO 常用的 Mosaic 和 MixUp 增强
    """
    return A.Compose([
        # Mosaic 增强（需要特殊处理，这里用 CutOut 模拟）
        A.CoarseDropout(
            max_holes=8,
            max_height=50,
            max_width=50,
            min_holes=1,
            min_height=20,
            min_width=20,
            fill_value=114,  # YOLO 灰色
            p=0.5
        ),

        # MixUp 风格的颜色混合
        A.OneOf([
            A.CLAHE(clip_limit=2),      # 对比度受限自适应直方图均衡
            A.Sharpen(),                 # 锐化
            A.Emboss(),                  # 浮雕
        ], p=0.3),

        # 随机阴影
        A.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1),
            num_shadows_lower=1,
            num_shadows_upper=2,
            p=0.2
        ),

        # 透视变换
        A.Perspective(
            scale=(0.05, 0.1),
            p=0.2
        ),

        # 弹性变形
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

    因此：
    - 增强小目标可见性
    - 多使用旋转/翻转
    - 谨慎使用颜色变换
    """
    return A.Compose([
        # 必须：翻转和旋转
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        # 可选：光照变化（如果颜色不重要）
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.3
        ),

        # 可选：模糊（模拟不同焦距）
        A.Blur(blur_limit=3, p=0.15),

        # 推荐：锐化（增强缺陷边缘）
        A.Sharpen(alpha=(0.2, 0.5), p=0.3),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels']
    ))

if __name__ == "__main__":
    # 创建针对缺陷检测的增强
    augment = create_defect_specific_augment()
    print("缺陷检测数据增强 pipeline 创建完成")
```

### 4.3 批量增强并保存

```python
# examples/09-batch-augmentation.py
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
            # 读取图片
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img.shape[:2]

            # 读取标注
            label_path = f"{input_label_dir}/{img_path.stem}.txt"
            bboxes, labels = self.load_yolo_labels(label_path, width, height)

            if not bboxes:
                # 没有标注，直接复制原图
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

                    # 保存增强后的数据
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
    # 创建增强器
    from examples_08 import create_defect_specific_augment

    augmenter = DatasetAugmenter(
        augment_fn=create_defect_specific_augment(),
        output_dir="datasets/augmented/"
    )

    # 增强数据集
    augmenter.augment_dataset(
        input_image_dir="datasets/split/images/val",
        input_label_dir="datasets/split/labels/val",
        num_augments=2  # 每张图片生成 2 个增强版本
    )
```

---

## 5️⃣ YOLO 格式转换

### 5.1 LabelImg XML → YOLO

```python
# examples/10-xml-to-yolo.py
import xml.etree.ElementTree as ET
import os
from pathlib import Path

def xml_to_yolo(xml_path, image_path, output_path):
    """
    将 LabelImg XML 格式转换为 YOLO 格式

    XML 格式:
    <annotation>
        <size><width>640</width><height>480</height></size>
        <object>
            <name>defect</name>
            <bndbox><xmin>100</xmin><ymin>100</ymin><xmax>200</xmax><ymax>200</ymax></bndbox>
        </object>
    </annotation>

    YOLO 格式:
    <class_id> <x_center> <y_center> <width> <height>
    所有值归一化到 [0, 1]
    """
    # 读取 XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 获取图片尺寸
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)

    # 转换标注
    yolo_lines = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        bbox = obj.find('bndbox')

        # XML 是 [xmin, ymin, xmax, ymax]
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        # 转换为 YOLO 格式 [x_center, y_center, width, height]
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        w = (xmax - xmin) / img_width
        h = (ymax - ymin) / img_height

        # 归一化
        yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    # 保存
    with open(output_path, 'w') as f:
        f.writelines(yolo_lines)

def batch_xml_to_yolo(xml_dir, output_dir):
    """批量转换"""
    os.makedirs(output_dir, exist_ok=True)

    for xml_file in Path(xml_dir).glob("*.xml"):
        xml_to_yolo(
            str(xml_file),
            f"{xml_dir}/{xml_file.stem}.jpg",
            f"{output_dir}/{xml_file.stem}.txt"
        )

    print(f"转换完成：{output_dir}")

if __name__ == "__main__":
    batch_xml_to_yolo("datasets/xml_labels/", "datasets/yolo_labels/")
```

### 5.2 创建 data.yaml

```python
# examples/11-create-data-yaml.py
import os
import yaml
from pathlib import Path

def create_data_yaml(dataset_dir, class_names, output_path):
    """
    创建 YOLO data.yaml 配置文件

    参数:
        dataset_dir: 数据集根目录
        class_names: 类别名称列表
        output_path: 输出路径
    """
    config = {
        'path': os.path.abspath(dataset_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',  # 可选
        'nc': len(class_names),
        'names': class_names
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    print(f"data.yaml 已创建：{output_path}")
    print("\n内容预览:")
    print(yaml.dump(config, allow_unicode=True))

# 使用示例
if __name__ == "__main__":
    # PCB 缺陷检测示例
    classes = ['hole', 'scratch', 'short', 'open', 'spurious']

    create_data_yaml(
        dataset_dir="datasets/split/",
        class_names=classes,
        output_path="datasets/split/data.yaml"
    )
```

---

## 6️⃣ 数据质量检查清单

### 6.1 自动检查脚本

```python
# examples/12-final-quality-check.py
import os
from pathlib import Path
import yaml

def final_quality_check(dataset_dir):
    """
    最终数据质量检查

    检查项目:
    1. 目录结构是否正确
    2. 图片和标注数量是否匹配
    3. 标注格式是否正确
    4. data.yaml 是否存在
    5. 类别是否平衡
    """
    issues = []
    warnings = []

    # 1. 检查目录结构
    required_dirs = [
        f"{dataset_dir}/images/train",
        f"{dataset_dir}/images/val",
        f"{dataset_dir}/labels/train",
        f"{dataset_dir}/labels/val"
    ]

    for d in required_dirs:
        if not os.path.exists(d):
            issues.append(f"缺少目录：{d}")

    # 2. 检查图片和标注匹配
    for split in ['train', 'val']:
        img_dir = f"{dataset_dir}/images/{split}"
        lbl_dir = f"{dataset_dir}/labels/{split}"

        if os.path.exists(img_dir) and os.path.exists(lbl_dir):
            img_files = set(f.stem for f in Path(img_dir).glob("*.jpg"))
            lbl_files = set(f.stem for f in Path(lbl_dir).glob("*.txt"))

            missing_labels = img_files - lbl_files
            missing_images = lbl_files - img_files

            if missing_labels:
                warnings.append(f"{split} 集：{len(missing_labels)} 张图片缺少标注")
            if missing_images:
                warnings.append(f"{split} 集：{len(missing_images)} 个标注缺少图片")

    # 3. 检查标注格式
    for lbl_file in Path(f"{dataset_dir}/labels/train").glob("*.txt"):
        with open(lbl_file, 'r') as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) != 5:
                    issues.append(f"标注格式错误：{lbl_file}:{i+1}")
                    break

                # 检查归一化值
                try:
                    values = [float(p) for p in parts[1:]]
                    if not all(0 <= v <= 1 for v in values):
                        warnings.append(f"标注值超出 [0,1]: {lbl_file}:{i+1}")
                except ValueError:
                    issues.append(f"标注值不是数字：{lbl_file}:{i+1}")
                    break

    # 4. 检查 data.yaml
    yaml_path = f"{dataset_dir}/data.yaml"
    if not os.path.exists(yaml_path):
        warnings.append("缺少 data.yaml 配置文件")
    else:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
            if 'names' not in config:
                issues.append("data.yaml 缺少 'names' 字段")
            if 'nc' not in config:
                warnings.append("data.yaml 缺少 'nc' 字段")

    # 输出报告
    print("=" * 50)
    print("数据质量检查报告")
    print("=" * 50)

    if issues:
        print(f"\n❌ 发现 {len(issues)} 个问题:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ 没有发现严重问题")

    if warnings:
        print(f"\n⚠️ 发现 {len(warnings)} 个警告:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("\n✅ 没有发现警告")

    print("\n" + "=" * 50)

    return len(issues) == 0

if __name__ == "__main__":
    final_quality_check("datasets/split/")
```

---

## 📝 实战练习

### 练习 1：数据清洗（30 分钟）
```bash
# 1. 运行模糊检测脚本，删除模糊图片
python examples/03-remove-blur.py

# 2. 运行去重脚本
python examples/04-remove-duplicates.py

# 3. 生成质量报告
python examples/05-data-quality-check.py
```

### 练习 2：数据划分（30 分钟）
```bash
# 运行分层抽样划分
python examples/06-split-dataset.py
```

### 练习 3：数据增强（60 分钟）
```bash
# 1. 创建缺陷检测增强 pipeline
# 2. 对验证集进行 2 倍增强
# 3. 检查增强后的数据质量
```

### 练习 4：格式转换（30 分钟）
```bash
# 1. 将 LabelImg XML 转换为 YOLO 格式
python examples/10-xml-to-yolo.py

# 2. 创建 data.yaml 配置文件
python examples/11-create-data-yaml.py

# 3. 运行最终质量检查
python examples/12-final-quality-check.py
```

---

## ✅ 数据处理检查清单

完成数据处理后，确保：

- [ ] 删除了模糊和重复图片
- [ ] 训练集/验证集/测试集比例合理（7:2:1）
- [ ] 每个类别在划分中比例均衡
- [ ] 数据增强已应用（至少 2 倍）
- [ ] 所有标注转换为 YOLO 格式
- [ ] data.yaml 配置文件正确
- [ ] 最终质量检查通过

---

## 🔗 相关资源

- [Albumentations 文档](https://albumentations.ai/docs/)
- [YOLO 数据格式详解](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/)
- [数据增强最佳实践](https://blog.roboflow.com/data-augmentation/)

---

**下一步：[09-Custom-Dataset - 数据标注实战](../09-custom-dataset/README.md)** 🚀
