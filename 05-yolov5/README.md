# 05-YOLOv5 - YOLOv5 实战

> 学习使用 YOLOv5 进行目标检测，理解工程化实现的精髓

---

## 🎯 学习目标

完成本章后，你将能够：

- ✅ 理解 YOLOv5 的架构特点
- ✅ 使用 YOLOv5 进行推理
- ✅ 训练自定义数据集
- ✅ 导出模型用于部署
- ✅ 比较 YOLOv5 与 YOLOv8 的差异

---

## 📦 1. 安装 YOLOv5

### 1.1 方式一：使用 torch.hub（推荐）

```bash
# 自动下载 YOLOv5 代码和依赖
pip install torch torchvision
```

```python
import torch

# 加载 YOLOv5s 模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
```

### 1.2 方式二：克隆仓库

```bash
# 克隆 YOLOv5 仓库
git clone https://github.com/ultralytics/yolov5.git
cd yolov5

# 安装依赖
pip install -r requirements.txt
```

### 1.3 模型规格

| 模型 | 参数量 | 权重文件 | 适用场景 |
|------|--------|----------|----------|
| YOLOv5n | 1.9M | yolov5n.pt | 边缘设备、实时性要求高 |
| YOLOv5s | 7.2M | yolov5s.pt | 平衡速度与精度 |
| YOLOv5m | 21.2M | yolov5m.pt | 精度要求较高 |
| YOLOv5l | 46.5M | yolov5l.pt | 高精度场景 |
| YOLOv5x | 86.7M | yolov5x.pt | 最高精度要求 |

---

## 🏗️ 2. YOLOv5 架构详解

### 2.1 整体架构

```
输入 (640x640x3)
    ↓
┌─────────────────────────────────┐
│ Backbone (CSPDarknet)            │
│ - Focus 层 (v5.0) / Conv 层 (v6.0+)│
│ - C3 模块                        │
│ - SPPF 模块                     │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ Neck (PANet)                     │
│ - FPN + PAN 融合                │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ Head (检测头)                    │
│ - 三尺度检测                    │
│ - 锚框预测                      │
└─────────────────────────────────┘
    ↓
输出 (边界框 + 类别 + 置信度)
```

### 2.2 C3 模块（CSP Bottleneck with 3 convolutions）

```python
# C3 模块结构
input → Split
         ├→ C3_Block → C3_Block → C3_Block → Concat
         └─────────────────────────→        ↑
                                            ↓
                                         Conv → output
```

**C3 模块特点：**
- 跨阶段部分连接，减少计算量
- 多梯度路径，提升训练效果
- YOLOv5 的核心创新之一

### 2.3 SPPF 模块（Spatial Pyramid Pooling - Fast）

```
input → Conv → MaxPool(5) → MaxPool(5) → MaxPool(5) → Concat → output
         ↓                                         ↑
         └─────────────────────────────────────────┘
```

**SPPF 作用：**
- 增加感受野
- 融合多尺度特征
- 比 SPP 更快

---

## 💻 3. 推理使用

### 3.1 基础推理

```python
import torch

# 加载模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# 推理
img = 'image.jpg'
results = model(img)

# 显示结果
results.show()

# 获取检测结果
df = results.pandas().xyxy[0]
print(df)
```

**输出格式：**
```
    xmin    ymin    xmax    ymax  confidence    class    name
0  100.5   200.3   150.8   280.6    0.92        0      person
1  300.2   150.1   350.4   220.5    0.87        2      car
```

### 3.2 批量推理

```python
# 多张图片推理
imgs = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = model(imgs)

# 批量处理，自动 resize 和 padding
results.print()
results.save()   # 保存到 runs/detect/
```

### 3.3 视频推理

```python
# 视频文件
results = model('video.mp4')
results.save()

# 摄像头
results = model(0)  # 0 表示默认摄像头
results.show()
```

### 3.4 自定义推理参数

```python
results = model(
    'image.jpg',
    size=640,           # 推理尺寸
    conf=0.25,          # 置信度阈值
    iou=0.45,           # NMS IoU 阈值
    augment=False,      # TTA 测试时增强
    agnostic=False      # 类别无关 NMS
)
```

---

## 🔧 4. 训练自定义数据集

### 4.1 数据集配置

创建 `custom.yaml`：

```yaml
# 数据集路径
train: ./data/train/images
val: ./data/val/images
test: ./data/test/images

# 类别数
nc: 2

# 类别名称
names: ['defect', 'component']
```

### 4.2 开始训练

```bash
# 单 GPU 训练
python train.py \
    --data custom.yaml \
    --weights yolov5s.pt \
    --epochs 100 \
    --batch-size 16 \
    --img 640 \
    --device 0

# 多 GPU 训练
python train.py \
    --data custom.yaml \
    --weights yolov5s.pt \
    --epochs 100 \
    --batch-size 64 \
    --device 0,1,2,3
```

### 4.3 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --epochs | 100 | 训练轮数 |
| --batch-size | 16 | 批次大小 |
| --img | 640 | 输入图像尺寸 |
| --weights | yolov5s.pt | 预训练权重 |
| --device | 0 | GPU 设备 |
| --workers | 8 | 数据加载线程数 |
| --optimizer | SGD | 优化器 |
| --lr0 | 0.01 | 初始学习率 |

### 4.4 从检查结果点

```bash
# 从最后一个 checkpoint 继续训练
python train.py --data custom.yaml --weights runs/train/exp/weights/last.pt --resume
```

---

## 📊 5. 验证与评估

### 5.1 验证模型

```bash
# 验证训练好的模型
python val.py \
    --data custom.yaml \
    --weights runs/train/exp/weights/best.pt \
    --batch-size 16 \
    --img 640
```

### 5.2 输出指标

```
                 Class     Images  Instances      P      R   mAP50  mAP50-95
                   all          100        500   0.85   0.82    0.88      0.65
                defect          100        300   0.83   0.80    0.86      0.62
              component          100        200   0.87   0.84    0.90      0.68
```

---

## 📤 6. 模型导出

### 6.1 导出为 ONNX

```bash
python export.py \
    --weights runs/train/exp/weights/best.pt \
    --include onnx \
    --dynamic
```

### 6.2 导出为 TorchScript

```bash
python export.py \
    --weights runs/train/exp/weights/best.pt \
    --include torchscript
```

### 6.3 导出为 TensorRT

```bash
# 需要安装 tensorrt
python export.py \
    --weights runs/train/exp/weights/best.pt \
    --include engine \
    --device 0 \
    --half  # FP16
```

### 6.4 导出格式对比

| 格式 | 文件大小 | 推理速度 | 兼容性 |
|------|----------|----------|--------|
| PyTorch | 大 | 慢 | 最好 |
| ONNX | 中 | 中 | 好 |
| TorchScript | 中 | 中 | 较好 |
| TensorRT | 小 | 最快 | NVIDIA GPU |

---

## 🎯 7. 缺陷检测应用示例

### 7.1 PCB 缺陷检测

```python
import torch
import cv2

# 加载自定义模型
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='runs/train/pcb-defect/weights/best.pt')

# 推理
img = 'pcb_board.jpg'
results = model(img)

# 获取结果
detections = results.pandas().xyxy[0]

# 筛选缺陷（置信度 > 0.5）
defects = detections[
    (detections['confidence'] > 0.5) &
    (detections['name'] == 'defect')
]

print(f"发现 {len(defects)} 个缺陷")

# 保存结果
results.save()
```

### 7.2 表面划痕检测

```python
# 批量检测
img_list = ['product_001.jpg', 'product_002.jpg', ...]
results = model(img_list)

# 统计缺陷率
total = len(img_list)
defective = 0

for det in results.pandas().xyxy:
    if len(det[det['name'] == 'scratch']) > 0:
        defective += 1

defect_rate = defective / total
print(f"缺陷率：{defect_rate:.2%}")
```

---

## 💡 8. YOLOv5 vs YOLOv8

| 特性 | YOLOv5 | YOLOv8 |
|------|--------|--------|
| 锚框设计 | 有锚框 | 无锚框 |
| 检测头 | 耦合头 | 解耦头 |
| 分类损失 | BCE | VFL |
| 框损失 | CIoU | DFL + CIoU |
| 正样本匹配 | IoU 阈值 | Task Aligned |
| 工程化 | 成熟 | 较新 |
| 部署支持 | 完善 | 完善 |

---

## 📝 实战练习

### 练习 1：YOLOv5 推理（30 分钟）

```python
# exercises/01-yolov5-inference.py
"""
使用 YOLOv5 进行推理

任务:
1. 加载 YOLOv5s 模型
2. 推理一张图片
3. 绘制并保存检测结果
"""
import torch

# TODO: 加载模型
# TODO: 推理图片
# TODO: 显示/保存结果
```

### 练习 2：视频处理（45 分钟）

```python
# exercises/02-video-processing.py
"""
使用 YOLOv5 处理视频

任务:
1. 读取视频文件
2. 逐帧检测
3. 保存带标注的视频
"""
import torch
import cv2

# TODO: 加载模型
# TODO: 打开视频
# TODO: 逐帧处理
# TODO: 保存结果视频
```

### 练习 3：模型比较（30 分钟）

```python
# exercises/03-model-comparison.py
"""
比较 YOLOv5 不同规格模型

任务:
1. 加载 yolov5n, yolov5s, yolov5m
2. 推理同一张图片
3. 比较速度和精度
"""
import torch
import time

# TODO: 加载不同模型
# TODO: 推理并计时
# TODO: 比较结果
```

---

## ✅ 学习检查清单

完成本章后，确保你能够：

- [ ] 说明 YOLOv5 的架构组成
- [ ] 使用 YOLOv5 进行图片和视频推理
- [ ] 配置自定义数据集
- [ ] 训练 YOLOv5 模型
- [ ] 验证和评估模型
- [ ] 导出模型为 ONNX/TensorRT 格式
- [ ] 应用 YOLOv5 进行缺陷检测

---

## 🔗 相关资源

- [YOLOv5 官方仓库](https://github.com/ultralytics/yolov5)
- [YOLOv5 文档](https://docs.ultralytics.com/yolov5/)
- [缺陷检测数据集](https://github.com/RobinJZhao/PCB-defect-detection)

---

**下一步：[06-YOLOv8](../06-yolov8/README.md)** 🚀
