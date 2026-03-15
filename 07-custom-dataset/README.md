# 07-Custom Dataset - 自定义数据集

> 学习准备和标注自定义数据集

---

## 🎯 学习目标

- ✅ 数据收集
- ✅ 标注工具使用
- ✅ YOLO 格式转换
- ✅ 数据增强

---

## 📊 YOLO 数据格式

### 目录结构

```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

### 标注格式

每个目标一行：
```
<class_id> <x_center> <y_center> <width> <height>
```

所有值归一化到 [0, 1]

### data.yaml 配置

```yaml
path: /path/to/dataset
train: images/train
val: images/val

nc: 3  # 类别数
names: ['cat', 'dog', 'person']  # 类别名称
```

---

## 🛠️ 标注工具

### 1. LabelImg

```bash
pip install labelImg
labelImg
```

### 2. Roboflow

在线标注平台：
- 自动标注
- 数据增强
- 格式转换

### 3. CVAT

开源标注工具：
- 支持视频
- 团队协作
- 自动跟踪

---

## 💻 示例代码

### 数据增强

```python
from ultralytics import YOLO
import albumentations as A

# 定义增强
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Blur(p=0.1),
], bbox_params=A.BboxParams(format='yolo'))
```

---

## 📝 练习

1. 收集 50 张目标图片
2. 使用 LabelImg 标注
3. 转换为 YOLO 格式
4. 编写 data.yaml 配置文件

---

**继续学习：[08-Model-Training](../08-model-training/README.md)** 🚀
