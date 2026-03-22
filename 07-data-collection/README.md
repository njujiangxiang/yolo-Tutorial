# 07-Data Collection - 数据收集

> 学习如何收集高质量的训练数据

---

## 🎯 学习目标

完成本章后，你将能够：

- ✅ 掌握 5 种数据收集方法
- ✅ 使用 Python 爬虫收集网络图片
- ✅ 找到并下载公开数据集
- ✅ 自己拍摄符合要求的训练数据
- ✅ 整理和管理收集的数据

---

## 📚 为什么数据很重要？

**"Garbage In, Garbage Out"** —— 垃圾进，垃圾出

再好的模型，如果训练数据质量差，效果也不会好。数据收集是目标检测项目中最耗时但最重要的环节。

### 好数据的标准

| 标准 | 说明 | 检查方法 |
|------|------|----------|
| **数量足够** | 每类至少 50-100 张 | 统计各类别数量 |
| **质量合格** | 清晰、光线充足 | 人工抽检 |
| **覆盖全面** | 不同角度、光照、背景 | 检查多样性 |
| **标注准确** | 标注框贴合目标 | 人工审核 |

---

## 🛠️ 方法 1：网络爬虫收集

### 1.1 使用 Python 爬虫

```python
# examples/01-web-scraper.py
import requests
from bs4 import BeautifulSoup
import os
import time

def download_images(query, num_images=50):
    """
    从搜索引擎下载图片

    参数:
        query: 搜索关键词
        num_images: 下载数量
    """
    # 创建保存目录
    save_dir = f"datasets/raw/{query}"
    os.makedirs(save_dir, exist_ok=True)

    # 注意：实际使用需要调用搜索引擎 API
    # 这里只是示例结构
    print(f"搜索关键词：{query}")
    print(f"保存目录：{save_dir}")
    print(f"计划下载：{num_images} 张图片")

    # TODO: 实现具体的下载逻辑
    # 建议使用：google-images-download 或 imglyb 库

if __name__ == "__main__":
    download_images("PCB 缺陷", 50)
```

### 1.2 使用现成工具

#### 方法 A：google-images-download

```bash
# 安装
pip install google-images-download

# 使用
googleimagesdownload --keywords "PCB defect" --limit 100 --output_dir datasets/raw/
```

#### 方法 B：imglyb（推荐）

```bash
# 安装
pip install imglyb

# 使用
python -m imglyb -s "PCB defect" -n 100 -o datasets/raw/
```

### 1.3 注意事项

- ⚠️ **版权问题**：确保图片可用于学习和研究
- ⚠️ **图片质量**：过滤掉太小的图片（建议 > 200x200）
- ⚠️ **相关性**：人工筛选，删除不相关的图片

---

## 📦 方法 2：公开数据集

### 2.1 通用目标检测数据集

| 数据集 | 类别数 | 图片数 | 网址 |
|--------|--------|--------|------|
| COCO | 80 | 330K | https://cocodataset.org |
| Pascal VOC | 20 | 27K | http://host.robots.ox.ac.uk/pascal/VOC |
| Open Images | 500+ | 9M | https://storage.googleapis.com/openimages/web/index.html |

### 2.2 缺陷检测专用数据集

#### PCB 缺陷数据集

```bash
# 示例：下载 PCB 缺陷数据集
# 地址：https://github.com/CharlesAverill/PCB-defect-detection

git clone https://github.com/CharlesAverill/PCB-defect-detection.git datasets/pcb-defect/
```

#### NEU 表面缺陷数据集

```bash
# NEU-DET 钢铁表面缺陷数据集
# 包含 6 种缺陷类型：Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches
# 下载地址：http://surface.ustb.edu.cn/
```

#### MVTec AD 异常检测数据集

```bash
# MVTec Anomaly Detection
# 包含 15 类物体的缺陷样本
# 下载地址：https://www.mvtec.com/company/research/datasets/mvtec-ad/
```

### 2.3 数据集整理脚本

```python
# examples/02-dataset-organizer.py
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

    # TODO: 实现具体的整理逻辑
    print(f"整理数据集：{source_dir} -> {output_dir}")

if __name__ == "__main__":
    organize_coco_format("datasets/raw/", "datasets/organized/")
```

---

## 📸 方法 3：自己拍摄

### 3.1 拍摄设备选择

| 设备 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| 手机 | 方便、快速 | 质量一般 | 快速原型 |
| 数码相机 | 质量好 | 需要传输 | 高质量需求 |
| 工业相机 | 稳定、可控制 | 昂贵 | 生产环境 |

### 3.2 拍摄要点

#### 光线
- ✅ 使用均匀的光源
- ✅ 避免强烈反光
- ❌ 避免阴影遮挡目标

#### 角度
- ✅ 保持相机与被摄物平行
- ✅ 尝试多个角度（0°, 30°, 45°, 60°）
- ❌ 避免极端俯视/仰视

#### 背景
- ✅ 使用纯色背景（白色/黑色）
- ✅ 背景与目标有明显对比
- ❌ 避免杂乱背景

#### 比例尺
- ✅ 在画面中放置比例尺或参照物
- ✅ 保持一致的拍摄距离

### 3.3 批量拍摄技巧

```python
# examples/03-photo-organizer.py
"""
批量拍摄后整理脚本
"""
import os
from datetime import datetime
import shutil

def organize_photos(source_dir, output_dir):
    """
    整理拍摄的照片

    - 删除模糊照片
    - 按日期分类
    - 重命名
    """
    print(f"整理照片：{source_dir} -> {output_dir}")

    # TODO: 实现照片筛选和整理逻辑

if __name__ == "__main__":
    organize_photos("photos/raw/", "datasets/photos/")
```

---

## 🎨 方法 4：数据合成

### 4.1 背景替换

使用图像处理软件（Photoshop）或代码实现：

```python
# examples/04-data-synthesis.py
import cv2
import numpy as np
import os

def synthesize_defect_images(defect_dir, background_dir, output_dir):
    """
    合成缺陷图片

    原理：将缺陷区域抠图，粘贴到不同背景上
    """
    os.makedirs(output_dir, exist_ok=True)

    # 读取缺陷图片和背景
    # TODO: 实现图像合成逻辑

    print(f"合成图片保存到：{output_dir}")

if __name__ == "__main__":
    synthesize_defect_images(
        "datasets/defects/",
        "datasets/backgrounds/",
        "datasets/synthesized/"
    )
```

### 4.2 使用数据增强生成

参考 [08-data-processing](../08-data-processing/README.md) 中的数据增强章节

---

## 🌐 方法 5：开源社区

### 5.1 Roboflow

https://universe.roboflow.com/

- 超过 50,000 个公开数据集
- 可以直接导出为 YOLO 格式
- 支持在线标注和数据增强

```bash
# 使用 Roboflow 数据集
# 1. 在网站上找到需要的数据集
# 2. 选择 YOLOv8 格式导出
# 3. 下载并解压到 datasets/ 目录
```

### 5.2 Kaggle

https://www.kaggle.com/datasets

- 搜索 "object detection" 或 "defect detection"
- 需要注册账号
- 通常包含完整的数据和说明

### 5.3 GitHub

搜索关键词：
- `defect detection dataset`
- `PCB defect`
- `surface defect`
- `industrial inspection`

---

## 📝 实战练习

完成以下任务：

### 练习 1：爬虫收集（30 分钟）
```bash
# 使用 google-images-download 或 imglyb
# 收集 "PCB defect" 或 "surface defect" 相关图片 30 张
```

### 练习 2：下载公开数据集（30 分钟）
```bash
# 从以下选择一个数据集下载:
# - PCB 缺陷数据集
# - NEU 表面缺陷
# - Roboflow 上的缺陷检测数据集
```

### 练习 3：整理数据（30 分钟）
```bash
# 将收集的数据整理为以下结构:
datasets/
├── raw/           # 原始下载的数据
├── organized/     # 整理后的数据
└── scratch/       # 练习用临时目录
```

---

## ✅ 数据收集检查清单

完成数据收集后，检查：

- [ ] 图片总数 ≥ 50 张
- [ ] 每类缺陷至少有 10 张图片
- [ ] 图片清晰度高（> 200x200）
- [ ] 包含不同角度和光照条件
- [ ] 删除了不相关的图片
- [ ] 目录结构清晰

---

## 🔗 相关资源

- [Roboflow 博客 - 数据收集最佳实践](https://blog.roboflow.com/computer-vision-data-collection/)
- [Kaggle 数据集搜索](https://www.kaggle.com/datasets)
- [Papers With Code - 目标检测数据集](https://paperswithcode.com/task/object-detection)

---

**下一步：[08-Data-Processing - 数据处理](../08-data-processing/README.md)** 🚀
