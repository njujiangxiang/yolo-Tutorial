# 06-YOLOv8 - YOLOv8 实战

> 学习使用最新的 YOLOv8 进行目标检测，掌握 Ultralytics 框架

---

## 🎯 学习目标

完成本章后，你将能够：

- ✅ 理解 YOLOv8 的新特性
- ✅ 使用 Ultralytics 框架进行推理
- ✅ 训练和验证模型
- ✅ 导出模型用于部署
- ✅ 应用 YOLOv8 进行缺陷检测

---

## 📦 1. 安装

```bash
# 方式一：pip 安装（推荐）
pip install ultralytics

# 方式二：从源码安装
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
pip install -e .
```

---

## 🆕 2. YOLOv8 新特性

### 2.1 核心改进

| 特性 | YOLOv5/v6 | YOLOv8 | 优势 |
|------|-----------|--------|------|
| 锚框设计 | 有锚框 | 无锚框 | 更简单、更通用 |
| 检测头 | 耦合头 | 解耦头 | 更好的特征学习 |
| 分类损失 | BCE | VFL | 更好的类别预测 |
| 框损失 | CIoU | DFL + CIoU | 更精确的定位 |
| 正样本匹配 | IoU 阈值 | Task Aligned | 更高效的训练 |
| 骨干网络 | C3 模块 | C2f 模块 | 更丰富的梯度流 |

### 2.2 C2f 模块

```
input → Conv → Split
         ├→ C2f_Block → C2f_Block → Concat
         └─────────────────────────→    ↑
                                        ↓
                                     Conv → output
```

**C2f 相比 C3 的优势：**
- 更多的跳跃连接
- 更丰富的梯度流
- 更好的特征复用

### 2.3 任务支持

YOLOv8 支持多种视觉任务：

```python
from ultralytics import YOLO

# 目标检测
model = YOLO('yolov8n.pt')

# 实例分割
model = YOLO('yolov8n-seg.pt')

# 姿态估计
model = YOLO('yolov8n-pose.pt')

# 图像分类
model = YOLO('yolov8n-cls.pt')
```

---

## 📊 3. 模型规格

| 模型 | 参数量 | FLOPs | 输入尺寸 | mAP | 速度 |
|------|--------|-------|----------|-----|------|
| YOLOv8n | 3.2M | 8.7G | 640 | 37.3 | 最快 |
| YOLOv8s | 11.2M | 28.6G | 640 | 44.9 | 快 |
| YOLOv8m | 25.9M | 78.9G | 640 | 50.2 | 中 |
| YOLOv8l | 43.7M | 165.2G | 640 | 52.9 | 慢 |
| YOLOv8x | 68.2M | 257.8G | 640 | 53.9 | 最慢 |

**选择建议：**
- **n/s**: 边缘设备、实时应用
- **m**: 平衡速度与精度（推荐）
- **l/x**: 高精度要求、离线处理

---

## 💻 4. 推理使用

### 4.1 基础推理

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('yolov8n.pt')

# 推理
results = model('image.jpg')

# 显示结果
results[0].show()

# 获取检测结果
boxes = results[0].boxes
print(f"检测到 {len(boxes)} 个目标")

# 获取边界框
for box in boxes:
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    conf = float(box.conf[0])
    cls = int(box.cls[0])
    print(f"类别：{cls}, 置信度：{conf:.2f}, 位置：[{x1}, {y1}, {x2}, {y2}]")
```

### 4.2 批量推理

```python
# 多张图片
images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = model(images)

for result in results:
    result.show()
```

### 4.3 视频推理

```python
# 视频文件
results = model('video.mp4', stream=True)

for result in results:
    result.show()

# 摄像头
results = model(0, stream=True)  # 0 表示默认摄像头

for result in results:
    result.show()
```

### 4.4 自定义参数

```python
results = model(
    'image.jpg',
    imgsz=640,        # 推理尺寸
    conf=0.25,        # 置信度阈值
    iou=0.45,         # NMS IoU 阈值
    max_det=300,      # 最大检测数
    classes=[0, 2],   # 只检测特定类别
    agnostic_nms=False,  # 类别无关 NMS
    augment=False,    # TTA 测试时增强
    visualize=False,  # 可视化特征图
    retina_masks=False  # 高分辨率掩码
)
```

---

## 🏋️ 5. 训练模型

### 5.1 数据集配置

创建 `custom.yaml`：

```yaml
# 数据集路径
train: ./data/train/images
val: ./data/val/images

# 类别数
nc: 2

# 类别名称
names:
  - defect
  - component
```

### 5.2 训练方式

**方式一：Python API**

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('yolov8n.pt')  # 从预训练权重开始

# 训练
model.train(
    data='custom.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    workers=8,
    optimizer='SGD',
    lr0=0.01,
    device=0  # GPU 0
)
```

**方式二：命令行**

```bash
# 训练
yolo detect train \
    data=custom.yaml \
    model=yolov8n.pt \
    epochs=100 \
    batch=16 \
    imgsz=640 \
    device=0
```

### 5.3 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| epochs | 100 | 训练轮数 |
| batch | 16 | 批次大小 |
| imgsz | 640 | 输入图像尺寸 |
| workers | 8 | 数据加载线程数 |
| optimizer | SGD | 优化器 (SGD/Adam/AdamW) |
| lr0 | 0.01 | 初始学习率 |
| lrf | 0.01 | 最终学习率 (lr0 * lrf) |
| momentum | 0.937 | SGD 动量 |
| weight_decay | 0.0005 | 权重衰减 |
| warmup_epochs | 3.0 | 预热轮数 |
| box | 7.5 | 框损失权重 |
| cls | 0.5 | 分类损失权重 |
| dfl | 1.5 | DFL 损失权重 |
| patience | 100 | 早停耐心值 |

### 5.4 从检查结果点

```python
# 从最后一个 checkpoint 继续训练
model.train(
    data='custom.yaml',
    resume=True  # 自动查找最新的 checkpoint
)
```

---

## 📊 6. 验证与评估

### 6.1 验证模型

**方式一：Python API**

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/detect/train/weights/best.pt')

# 验证
metrics = model.val(data='custom.yaml')

# 获取指标
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
```

**方式二：命令行**

```bash
yolo detect val \
    model=runs/detect/train/weights/best.pt \
    data=custom.yaml \
    split=val
```

### 6.2 预测新数据

```python
# 预测
results = model.predict(
    source='test_images/',
    save=True,
    save_txt=True,      # 保存为 TXT
    save_conf=True,     # 保存置信度
    project='runs/predict',
    name='exp1'
)
```

---

## 📤 7. 模型导出

### 7.1 导出格式

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# 导出为 ONNX
model.export(format='onnx')

# 导出为 TensorRT
model.export(format='engine', device=0, half=True)

# 导出为 OpenVINO
model.export(format='openvino')

# 导出为 CoreML
model.export(format='coreml')

# 导出为 TorchScript
model.export(format='torchscript')
```

### 7.2 命令行导出

```bash
# ONNX 导出
yolo export model=yolov8n.pt format=onnx

# TensorRT 导出
yolo export model=yolov8n.pt format=engine device=0 half=True

# 动态轴 ONNX
yolo export model=yolov8n.pt format=onnx dynamic=True
```

### 7.3 格式对比

| 格式 | 文件大小 | CPU 速度 | GPU 速度 | 兼容性 |
|------|----------|----------|----------|--------|
| PyTorch | 大 | 慢 | 中 | 最好 |
| ONNX | 中 | 中 | 中 | 好 |
| TensorRT | 小 | - | 最快 | NVIDIA |
| OpenVINO | 中 | 快 | - | Intel |
| CoreML | 中 | 快 | - | Apple |

---

## 🔍 8. 缺陷检测应用

### 8.1 实时检测

```python
from ultralytics import YOLO
import cv2

# 加载模型
model = YOLO('runs/detect/train/weights/best.pt')

# 摄像头检测
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            if float(box.conf[0]) > 0.5:  # 置信度 > 0.5
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Defect Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 8.2 批量处理与统计

```python
from ultralytics import YOLO
import os

model = YOLO('runs/detect/train/weights/best.pt')

# 批量处理产品图片
product_dir = 'products/'
total = 0
defective = 0

for img_name in os.listdir(product_dir):
    if img_name.endswith('.jpg'):
        img_path = os.path.join(product_dir, img_name)
        results = model(img_path)

        has_defect = len(results[0].boxes) > 0
        if has_defect:
            defective += 1
        total += 1

defect_rate = defective / total * 100
print(f"检测 {total} 个产品，缺陷 {defective} 个，缺陷率：{defect_rate:.2f}%")
```

---

## 💡 9. YOLOv8 vs YOLOv5

### 9.1 架构对比

| 组件 | YOLOv5 | YOLOv8 |
|------|--------|--------|
| Backbone | CSPDarknet | Modified CSPDarknet |
| Neck | PAN-FPN | PAN-FPN |
| Head | Coupled | Decoupled |
| Anchor | Yes | No |
| Loss | CIoU | DFL + CIoU |

### 9.2 性能对比

在 COCO 数据集上：

| 模型 | mAP | 参数量 | 速度 (V100) |
|------|-----|--------|-------------|
| YOLOv5s | 37.4 | 7.2M | 1.4ms |
| YOLOv8s | 44.9 | 11.2M | 1.6ms |
| YOLOv5m | 45.4 | 21.2M | 2.2ms |
| YOLOv8m | 50.2 | 25.9M | 2.5ms |

### 9.3 使用体验

| 特性 | YOLOv5 | YOLOv8 |
|------|--------|--------|
| API 设计 | 传统 | 统一 |
| 文档 | 完善 | 完善 |
| 社区 | 成熟 | 活跃 |
| 部署工具 | 完善 | 完善 |
| 多任务 | 有限 | 原生支持 |

---

## 📝 实战练习

### 练习 1：YOLOv8 推理（30 分钟）

```python
# exercises/01-yolov8-inference.py
"""
使用 YOLOv8 进行推理

任务:
1. 加载 YOLOv8n 模型
2. 推理一张图片
3. 获取并打印检测结果
"""
from ultralytics import YOLO

# TODO: 加载模型
# TODO: 推理图片
# TODO: 打印检测结果
```

### 练习 2：模型训练（60 分钟）

```python
# exercises/02-model-training.py
"""
训练 YOLOv8 模型

任务:
1. 配置数据集
2. 设置训练参数
3. 开始训练
4. 查看训练结果
"""
from ultralytics import YOLO

# TODO: 加载预训练模型
# TODO: 配置训练参数
# TODO: 开始训练
```

### 练习 3：模型导出（30 分钟）

```python
# exercises/03-model-export.py
"""
导出 YOLOv8 模型

任务:
1. 加载训练好的模型
2. 导出为 ONNX 格式
3. 验证导出结果
"""
from ultralytics import YOLO

# TODO: 加载模型
# TODO: 导出为 ONNX
# TODO: 测试导出模型
```

### 练习 4：缺陷检测项目（90 分钟）

```python
# exercises/04-defect-detection-app.py
"""
缺陷检测应用

任务:
1. 加载自定义模型
2. 批量处理产品图片
3. 统计缺陷率
4. 生成报告
"""
from ultralytics import YOLO
import os

# TODO: 加载模型
# TODO: 批量检测
# TODO: 统计并生成报告
```

---

## ✅ 学习检查清单

完成本章后，确保你能够：

- [ ] 说明 YOLOv8 相比 YOLOv5 的改进
- [ ] 使用 Ultralytics 框架进行推理
- [ ] 配置和训练自定义模型
- [ ] 验证和评估模型性能
- [ ] 导出模型为 ONNX/TensorRT 格式
- [ ] 应用 YOLOv8 进行缺陷检测
- [ ] 选择合适的模型规格

---

## 🔗 相关资源

- [Ultralytics 官方文档](https://docs.ultralytics.com/)
- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
- [缺陷检测数据集](https://github.com/RobinJZhao/PCB-defect-detection)

---

**下一步：[07-Data-Collection](../07-data-collection/README.md)** 🚀
