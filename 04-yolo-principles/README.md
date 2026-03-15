# 04-YOLO Principles - YOLO 原理

> 深入理解 YOLO 目标检测算法的核心原理

---

## 🎯 学习目标

- ✅ YOLO 发展历史
- ✅ YOLO 架构详解
- ✅ 损失函数设计
- ✅ 多尺度检测

---

## 📚 YOLO 发展历史

| 版本 | 年份 | 特点 |
|------|------|------|
| YOLOv1 | 2016 | 首次提出实时检测 |
| YOLOv2 | 2017 | 引入锚框 |
| YOLOv3 | 2018 | 多尺度预测 |
| YOLOv4 | 2020 | 大量优化技巧 |
| YOLOv5 | 2020 | Ultralytics 实现 |
| YOLOv6 | 2022 | 美团开源 |
| YOLOv7 | 2022 | 高效架构 |
| YOLOv8 | 2023 | SOTA 性能 |

---

## 🏗️ YOLO 架构

```
输入图像
    ↓
Backbone (特征提取)
    ↓
Neck (特征融合)
    ↓
Head (检测头)
    ↓
输出 (边界框 + 类别)
```

### 核心组件

**Backbone**: CSPDarknet
- 特征提取
- 多尺度特征

**Neck**: PANet/FPN
- 特征金字塔
- 上下采样融合

**Head**: 检测头
- 边界框回归
- 类别预测

---

## 📐 关键概念

### 1. 锚框 (Anchor Boxes)

预定义的边界框尺寸和比例：
```
小目标：[10,13], [16,30], [33,23]
中目标：[30,61], [62,45], [59,119]
大目标：[116,90], [156,198], [373,326]
```

### 2. IoU (交并比)

```
IoU = 预测框 ∩ 真实框 / 预测框 ∪ 真实框
```

### 3. 非极大值抑制 (NMS)

去除重叠的检测框，保留最优结果。

---

## 💻 示例代码

### YOLO 推理示例

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('yolov8n.pt')

# 推理
results = model('image.jpg')

# 处理结果
for result in results:
    boxes = result.boxes
    for box in boxes:
        print(f"类别：{box.cls}, 置信度：{box.conf}")
```

---

## 📝 练习

1. 绘制 YOLO 架构图
2. 计算两个边界框的 IoU
3. 理解 NMS 算法原理
4. 比较不同 YOLO 版本

---

**继续学习：[05-YOLOv5](../05-yolov5/README.md)** 🚀
