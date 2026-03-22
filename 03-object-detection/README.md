# 03-Object-Detection - 目标检测基础

> 学习目标检测的基础概念和评估指标

---

## 🎯 学习目标

完成本章后，你将能够：

- ✅ 理解目标检测任务
- ✅ 掌握评估指标（IoU, mAP, Precision, Recall）
- ✅ 了解锚框（Anchor）概念
- ✅ 比较主流检测算法

---

## 📚 1. 目标检测任务

### 1.1 什么是目标检测

**目标检测 = 分类 + 定位**

```
输入：一张图片
输出：检测框 + 类别

例如:
┌─────────────────┐
│  ┌────┐  ┌───┐  │
│  │猫 65%│  │狗 89%│ │
│  └────┘  └───┘  │
└─────────────────┘
```

### 1.2 与图像分类的区别

| 任务 | 输入 | 输出 | 应用 |
|------|------|------|------|
| 图像分类 | 图片 | 类别 | 这是什么？ |
| 目标检测 | 图片 | 框 + 类别 | 有什么？在哪里？ |
| 语义分割 | 图片 | 像素级标签 | 每个像素是什么？ |

---

## 📊 2. 评估指标

### 2.1 IoU（交并比）

**定义：** 预测框和真实框的交集与并集的比值

$$IoU = \frac{Area_{overlap}}{Area_{union}}$$

```python
# examples/01-iou.py
"""
计算 IoU（交并比）
"""
def calculate_iou(box1, box2):
    """
    计算两个框的 IoU

    参数:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    """
    # 计算交集
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # 计算面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集
    union = area1 + area2 - intersection

    # IoU
    iou = intersection / union if union > 0 else 0

    return iou

# 示例
box_a = [50, 50, 150, 150]
box_b = [100, 100, 200, 200]

iou = calculate_iou(box_a, box_b)
print(f"IoU: {iou:.4f}")
```

**IoU 阈值：**
- IoU ≥ 0.5：通常认为是正确检测
- IoU ≥ 0.75：严格正确
- IoU = 1：完美重合

### 2.2 TP, FP, FN, TN

目标检测中的分类结果：

| 预测\真实 | 有目标 | 无目标 |
|-----------|--------|--------|
| 预测有 | **TP** (正确) | **FP** (误检) |
| 预测无 | **FN** (漏检) | **TN** (正确) |

**缺陷检测示例：**
- **TP**: 真有划痕，检测到了 ✓
- **FP**: 没有划痕，误报了 ✗
- **FN**: 有划痕，没检测到 ✗（最危险！）
- **TN**: 没有划痕，没报错 ✓

### 2.3 Precision 和 Recall

**Precision（查准率）：** 检测出的目标中有多少是真的

$$Precision = \frac{TP}{TP + FP}$$

**Recall（查全率）：** 真实目标中有多少被检测到了

$$Recall = \frac{TP}{TP + FN}$$

**示例：**
```
场景：100 个真实缺陷

模型检测到 80 个
其中 70 个是真的，10 个是误报
还有 30 个没检测到

Precision = 70 / (70 + 10) = 87.5%
Recall = 70 / (70 + 30) = 70%
```

**权衡关系：**
- 提高置信度阈值 → Precision↑, Recall↓
- 降低置信度阈值 → Precision↓, Recall↑

### 2.4 mAP（平均精度均值）

**AP（Average Precision）：** 单个类别的平均精度

**mAP（mean AP）：** 所有类别的平均

```
mAP50: IoU 阈值为 0.5 时的 mAP
mAP50-95: IoU 从 0.5 到 0.95 的平均 mAP
```

**评估标准：**

| mAP50 | mAP50-95 | 评价 |
|-------|----------|------|
| >0.90 | >0.70 | 优秀 |
| >0.80 | >0.50 | 良好 |
| >0.70 | >0.40 | 可用 |
| <0.70 | <0.40 | 需改进 |

---

## ⚓ 3. 锚框（Anchor Boxes）

### 3.1 什么是锚框

**锚框**是预定义的边界框尺寸和比例。

```
在每个位置预设多个锚框:
- 小框：[10,13], [16,30], [33,23]
- 中框：[30,61], [62,45], [59,119]
- 大框：[116,90], [156,198], [373,326]
```

### 3.2 为什么需要锚框

**解决的问题：**
1. 多尺度检测（大目标和小目标）
2. 加快收敛（有先验知识）
3. 提高定位精度

```python
# examples/02-anchor-boxes.py
"""
锚框可视化演示
"""
import matplotlib.pyplot as plt
import numpy as np

# YOLOv3 的锚框（归一化前）
anchors = [
    [10,13], [16,30], [33,23],  # 小目标
    [30,61], [62,45], [59,119], # 中目标
    [116,90], [156,198], [373,326] # 大目标
]

# 可视化
fig, ax = plt.subplots(figsize=(8, 8))

for i, (w, h) in enumerate(anchors):
    # 以原点为中心绘制矩形
    rect = plt.Rectangle((-w/2, -h/2), w, h,
                         fill=False, linewidth=2,
                         label=f'{w}x{h}')
    ax.add_patch(rect)

ax.set_xlim(-200, 200)
ax.set_ylim(-200, 200)
ax.set_aspect('equal')
ax.legend()
ax.set_title('YOLO Anchor Boxes')
plt.grid(True, alpha=0.3)
plt.savefig('anchor_boxes.png', dpi=150)
print("锚框图已保存：anchor_boxes.png")
```

---

## 🔍 4. 主流检测算法

### 4.1 算法发展历史

```
2014: R-CNN (开山之作)
  ↓
2015: Fast R-CNN (速度提升)
  ↓
2016: Faster R-CNN (端到端)
  ↓
2016: YOLOv1 (实时检测)
  ↓
2017: YOLOv2 (引入锚框)
  ↓
2018: YOLOv3 (多尺度)
  ↓
2020: YOLOv4/v5 (工程优化)
  ↓
2023: YOLOv8 (SOTA)
```

### 4.2 Two-Stage vs One-Stage

**Two-Stage (如 R-CNN 系列):**
```
步骤 1: 生成候选框 (Region Proposal)
  ↓
步骤 2: 分类和回归
```
- 优点：精度高
- 缺点：速度慢

**One-Stage (如 YOLO 系列):**
```
直接输出类别和位置
```
- 优点：速度快（实时）
- 缺点：早期精度较低（现在已接近）

### 4.3 算法对比

| 算法 | 类型 | 速度 | 精度 | 应用 |
|------|------|------|------|------|
| Faster R-CNN | Two-Stage | 5 FPS | 高 | 精度优先 |
| YOLOv8 | One-Stage | 80 FPS | 很高 | 实时检测 |
| SSD | One-Stage | 45 FPS | 中 | 平衡 |
| RetinaNet | One-Stage | 30 FPS | 高 | 密集检测 |

---

## 🛠️ 5. YOLO 检测流程

### 5.1 推理过程

```
输入图片 (640x640)
    ↓
Backbone 特征提取
    ↓
Neck 特征融合
    ↓
Head 预测 (21504 个框)
    ↓
置信度过滤 (>0.25)
    ↓
NMS 去重
    ↓
最终输出
```

### 5.2 NMS（非极大值抑制）

**作用：** 去除重叠的检测框，保留最优结果。

```python
# examples/03-nms.py
"""
NMS（非极大值抑制）演示
"""
import numpy as np

def nms(boxes, scores, iou_threshold=0.5):
    """
    非极大值抑制

    参数:
        boxes: 边界框 [[x1,y1,x2,y2], ...]
        scores: 置信度分数
        iou_threshold: IoU 阈值
    """
    # 按分数排序
    indices = np.argsort(scores)[::-1]
    keep = []

    while len(indices) > 0:
        # 选择分数最高的
        current = indices[0]
        keep.append(current)

        # 计算与其余框的 IoU
        if len(indices) > 1:
            ious = []
            for i in range(1, len(indices)):
                iou = calculate_iou(boxes[current], boxes[indices[i]])
                ious.append(iou)

            # 保留 IoU 低于阈值的
            indices = indices[1:][np.array(ious) < iou_threshold]
        else:
            break

    return keep

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

# 示例
boxes = [
    [50, 50, 150, 150],   # 框 1
    [55, 55, 155, 155],   # 框 2 (与框 1 重叠)
    [200, 200, 300, 300], # 框 3 (独立)
]
scores = [0.9, 0.8, 0.95]

keep = nms(boxes, scores, iou_threshold=0.5)
print(f"保留的框索引：{keep}")
print("NMS 后只保留不重叠的最优框")
```

---

## 💻 实战练习

### 练习 1：计算 IoU（30 分钟）

```python
# exercises/01-calculate-iou.py
"""
练习：计算 IoU

任务:
1. 实现 IoU 计算函数
2. 测试不同重叠情况
3. 可视化结果
"""
import cv2
import numpy as np

def calculate_iou(box1, box2):
    """
    TODO: 实现 IoU 计算

    box: [x1, y1, x2, y2]
    """
    # TODO: 计算交集
    # TODO: 计算面积
    # TODO: 计算并集
    # TODO: 返回 IoU
    pass

# 测试
box_a = [50, 50, 150, 150]
box_b = [100, 100, 200, 200]
box_c = [300, 300, 400, 400]  # 不重叠

# TODO: 计算并打印 IoU
```

### 练习 2：评估指标计算（30 分钟）

```python
# exercises/02-metrics.py
"""
练习：计算 Precision, Recall, mAP

任务:
1. 给定预测和真实标注
2. 计算 TP, FP, FN
3. 计算 Precision 和 Recall
"""

# 真实标注
ground_truth = [
    {'box': [50, 50, 100, 100], 'class': 'defect'},
    {'box': [200, 200, 250, 250], 'class': 'defect'},
    {'box': [300, 300, 350, 350], 'class': 'defect'},
]

# 模型预测
predictions = [
    {'box': [55, 55, 105, 105], 'conf': 0.9, 'class': 'defect'},
    {'box': [210, 210, 260, 260], 'conf': 0.8, 'class': 'defect'},
    {'box': [400, 400, 450, 450], 'conf': 0.7, 'class': 'defect'},  # FP
]

# TODO: 计算 TP, FP, FN
# TODO: 计算 Precision 和 Recall
```

### 练习 3：使用 YOLO 推理（60 分钟）

```python
# exercises/03-yolo-inference.py
"""
练习：使用 YOLO 进行目标检测

任务:
1. 加载 YOLO 模型
2. 推理一张图片
3. 绘制检测结果
4. 输出评估指标
"""
from ultralytics import YOLO
import cv2

# TODO: 加载模型
# TODO: 推理
# TODO: 绘制结果
# TODO: 保存
```

---

## ✅ 学习检查清单

完成本章后，确保你能够：

- [ ] 解释目标检测的任务定义
- [ ] 计算两个框的 IoU
- [ ] 说明 TP, FP, FN, TN 的含义
- [ ] 计算 Precision 和 Recall
- [ ] 解释 mAP 的含义
- [ ] 说明锚框的作用
- [ ] 理解 NMS 的原理
- [ ] 比较 Two-Stage 和 One-Stage

---

## 🔗 相关资源

- [YOLO 官方文档](https://docs.ultralytics.com/)
- [目标检测综述](https://arxiv.org/abs/2103.04037)
- [mAP 详解](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)
- [IoU 可视化](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)

---

**下一步：[04-YOLO-Principles - YOLO 原理](../04-yolo-principles/README.md)** 🚀
