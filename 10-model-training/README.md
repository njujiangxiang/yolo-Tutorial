# 10-Model Training - 模型训练

> 学习训练和优化 YOLO 模型

---

## 🎯 学习目标

完成本章后，你将能够：

- ✅ 选择合适的 YOLO 模型
- ✅ 配置训练参数
- ✅ 使用 TensorBoard 监控训练
- ✅ 诊断训练问题（过拟合/欠拟合）
- ✅ 使用迁移学习加速训练
- ✅ 优化模型性能

---

## 📦 选择模型

### YOLOv8 模型系列

| 模型 | 参数量 | 速度 | 精度 | 适用场景 |
|------|--------|------|------|----------|
| YOLOv8n | 3.2M | 80 FPS | 37.3 mAP | 移动端、边缘设备 |
| YOLOv8s | 11.2M | 45 FPS | 44.9 mAP | 平衡速度和精度 |
| YOLOv8m | 25.9M | 23 FPS | 50.2 mAP | 服务器部署 |
| YOLOv8l | 43.7M | 15 FPS | 52.9 mAP | 高精度需求 |
| YOLOv8x | 68.2M | 12 FPS | 53.9 mAP | 离线推理、最高精度 |

### 模型选择建议

```
快速原型 → yolov8n.pt
平衡方案 → yolov8s.pt
生产部署 → yolov8m.pt
高精度   → yolov8l.pt 或 yolov8x.pt
```

### 针对缺陷检测的推荐

**推荐：YOLOv8s 或 YOLOv8m**

原因：
- 缺陷通常较小，需要更好的检测能力
- 工业场景通常有 GPU，速度要求不高
- 精度比速度更重要

```python
from ultralytics import YOLO

# 推荐起点
model = YOLO('yolov8s.pt')

# 如果精度不够
model = YOLO('yolov8m.pt')

# 如果需要更快
model = YOLO('yolov8n.pt')
```

---

## 🚀 训练流程

### 完整训练代码

```python
# examples/train-custom.py
from ultralytics import YOLO

def train_defect_detection():
    """
    训练缺陷检测模型
    """
    # 1. 加载预训练模型
    model = YOLO('yolov8s.pt')

    # 2. 配置训练参数
    results = model.train(
        # 数据配置
        data='datasets/labeled/data.yaml',

        # 训练轮数
        epochs=100,

        # 图片尺寸
        imgsz=640,

        # 批次大小
        batch=16,

        # 数据加载线程数
        workers=8,

        # 设备
        device=0,  # GPU
        # device='cpu',  # CPU

        # 优化器
        optimizer='SGD',

        # 学习率
        lr0=0.01,

        # 早停耐心值
        patience=50,

        # 保存频率（每 n 个 epoch 保存一次）
        save_period=10,

        # 数据增强
        augment=True,

        # 混合精度训练
        amp=True,

        # 投影名称
        project='runs/detect',
        name='defect-train',

        # 可视化
        plots=True,
        verbose=True
    )

    return results

if __name__ == "__main__":
    train_defect_detection()
```

### 命令行训练（推荐）

```bash
# 基础训练
yolo detect train data=datasets/labeled/data.yaml model=yolov8s.pt epochs=100

# 完整配置
yolo detect train \
    data=datasets/labeled/data.yaml \
    model=yolov8s.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=0 \
    workers=8 \
    optimizer=SGD \
    lr0=0.01 \
    patience=50 \
    amp=True \
    project=runs/detect \
    name=defect-train
```

---

## 📊 训练参数详解

### 核心参数

| 参数 | 说明 | 推荐值 | 调优建议 |
|------|------|--------|----------|
| `epochs` | 训练轮数 | 100-300 | 小数据集 300，大数据集 100 |
| `imgsz` | 输入尺寸 | 640 | 小目标用 800-1280 |
| `batch` | 批次大小 | 16-64 | GPU 显存允许下越大越好 |
| `lr0` | 初始学习率 | 0.01 | 不收敛降低到 0.001 |
| `patience` | 早停耐心值 | 50-100 | 防止过拟合 |

### 优化器相关

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `optimizer` | 优化器类型 | SGD / AdamW |
| `lr0` | 初始学习率 | 0.01 (SGD) / 0.001 (Adam) |
| `lrf` | 最终学习率 | 0.01-0.1 (相对 lr0) |
| `momentum` | SGD 动量 | 0.937 |
| `weight_decay` | 权重衰减 | 0.0005 |

### 数据增强相关

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `augment` | 是否增强 | True |
| `hsv_h` | HSV 色相增强 | 0.015 |
| `hsv_s` | HSV 饱和度增强 | 0.7 |
| `hsv_v` | HSV 亮度增强 | 0.4 |
| `flipud` | 垂直翻转概率 | 0.0 |
| `fliplr` | 水平翻转概率 | 0.5 |
| `mosaic` | Mosaic 增强 | 1.0 |
| `mixup` | MixUp 增强 | 0.0 |

### 缺陷检测推荐配置

```python
# 针对缺陷检测的特殊配置
model.train(
    data='data.yaml',

    # 小目标检测 - 增大图片尺寸
    imgsz=800,

    # 小批次，更稳定
    batch=8,

    # 更多训练轮数
    epochs=200,

    # 降低学习率，精细训练
    lr0=0.001,

    # 更多的耐心
    patience=100,

    # 缺陷检测不需要垂直翻转（缺陷不会倒过来）
    flipud=0.0,

    # 水平翻转可以
    fliplr=0.5,

    # 关闭 Mosaic（可能破坏小缺陷）
    mosaic=0.5,
)
```

---

## 📈 训练监控

### TensorBoard 监控

```bash
# 启动 TensorBoard
tensorboard --logdir runs/detect/

# 在浏览器打开
# http://localhost:6006
```

### 关键指标

训练时关注以下指标：

```
Training:
  Epoch     GPU_mem   box_loss   cls_loss   dfl_loss   Instances   Size
    1/100     1.45G      1.234      0.567      1.012         123     640: ...

Validation:
   Epoch   precision    recall      mAP50   mAP50-95:
           0.856       0.789       0.823       0.654
```

### 指标解释

| 指标 | 含义 | 目标值 |
|------|------|--------|
| `box_loss` | 边界框损失 | 越低越好 |
| `cls_loss` | 分类损失 | 越低越好 |
| `dfl_loss` | 分布焦点损失 | 越低越好 |
| `precision` | 查准率 | 越高越好 |
| `recall` | 查全率 | 越高越好 |
| `mAP50` | IoU=0.5 时的 mAP | >0.8 优秀 |
| `mAP50-95` | 严格 mAP | >0.5 良好 |

### 训练曲线解读

**正常训练曲线：**
```
loss
  ↑
  │  ┌──────────
  │  │
  │   \
  │    \
  │     ────────  ← 收敛
  └──────────────→ epoch
```

**学习率太高：**
```
loss
  ↑
  │  ┌─┐ ┌─┐
  │  │ │ │ │  ← 震荡
  │   \│/ \|/
  │    ─   ─
  └─────────────→ epoch
```

**过拟合：**
```
      训练 loss   验证 loss
  ↑   ───────    ────┐
  │                 │
  │                 │
  │   ──────────────┘
  └──────────────────→ epoch
```

---

## 🔧 训练问题诊断

### 问题 1：训练不收敛

**现象：** loss 一直很高，不下降

**可能原因：**

| 原因 | 检查方法 | 解决方法 |
|------|----------|----------|
| 学习率太高 | loss 震荡 | 降低 lr0 到 0.001 |
| 标注错误 | 可视化标注 | 重新检查标注 |
| 数据太少 | 统计数量 | 增加数据或增强 |
| 模型不合适 | 换模型试试 | yolov8s → yolov8n |

**调试步骤：**

```bash
# 1. 用小模型快速测试
yolo detect train data=data.yaml model=yolov8n.pt epochs=10

# 2. 如果小模型能收敛，说明数据没问题
# 3. 换回大模型，降低学习率
yolo detect train data=data.yaml model=yolov8s.pt lr0=0.001
```

---

### 问题 2：过拟合

**现象：** 训练集 mAP 很高，验证集 mAP 低

**判断标准：**
```
训练 mAP50: 0.95
验证 mAP50: 0.65  ← 差距大说明过拟合
```

**解决方法：**

```python
# 1. 增加数据增强
model.train(
    augment=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
)

# 2. 早停
model.train(patience=50)

# 3. 换小模型
model = YOLO('yolov8n.pt')

# 4. 增加权重衰减
model.train(weight_decay=0.001)
```

---

### 问题 3：欠拟合

**现象：** 训练集和验证集 mAP 都很低

**解决方法：**

```python
# 1. 增加训练轮数
model.train(epochs=300)

# 2. 换大模型
model = YOLO('yolov8m.pt')

# 3. 提高学习率
model.train(lr0=0.02)

# 4. 关闭早停
model.train(patience=200)
```

---

### 问题 4：CUDA Out of Memory

**错误信息：**
```
RuntimeError: CUDA out of memory
```

**解决方法：**

```python
# 1. 减小 batch size
model.train(batch=8)

# 2. 减小图片尺寸
model.train(imgsz=416)

# 3. 换小模型
model = YOLO('yolov8n.pt')

# 4. 使用混合精度
model.train(amp=True)
```

---

## 🎯 迁移学习

### 为什么用迁移学习？

- ✅ 训练更快收敛
- ✅ 需要数据更少
- ✅ 精度更高

### 训练策略

```python
from ultralytics import YOLO

# 策略 1：使用 COCO 预训练（推荐）
model = YOLO('yolov8s.pt')
model.train(data='data.yaml', epochs=100)

# 策略 2：从检查点继续训练
model = YOLO('runs/detect/train/weights/last.pt')
model.train(resume=True)

# 策略 3：微调特定层
model = YOLO('yolov8s.pt')
model.train(freeze=10)  # 冻结前 10 层
```

### 缺陷检测训练策略

```python
# 针对缺陷检测的迁移学习

# 阶段 1：训练 backbone
model = YOLO('yolov8s.pt')
model.train(
    data='data.yaml',
    epochs=50,
    freeze=50,  # 冻结 backbone
    lr0=0.01
)

# 阶段 2：训练全部
model.train(
    data='data.yaml',
    epochs=150,
    freeze=0,   # 解冻所有层
    lr0=0.001   # 降低学习率
)
```

---

## 💾 训练结果

### 输出目录结构

```
runs/detect/defect-train/
├── weights/
│   ├── last.pt      # 最后一个 checkpoint
│   └── best.pt      # 验证集 mAP 最高的权重
├── args.yaml        # 训练参数
├── results.csv      # 训练日志
├── confusion_matrix.png
├── F1_curve.png
├── labels.jpg
├── PR_curve.png
├── precision_recall.png
├── results.png
└── train_batch*.jpg  # 训练图片可视化
```

### 使用训练好的模型

```python
from ultralytics import YOLO

# 加载最佳模型
model = YOLO('runs/detect/defect-train/weights/best.pt')

# 推理
results = model.predict('test_image.jpg')

# 批量推理
results = model.predict('datasets/val/images/', save=True)

# 评估
metrics = model.val(data='data.yaml')
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
```

---

## 📝 实战练习

### 练习 1：训练第一个模型（60 分钟）

```bash
# 1. 使用默认配置训练
yolo detect train data=datasets/labeled/data.yaml model=yolov8s.pt epochs=50

# 2. 观察训练输出
# 3. 记录最终 mAP
```

### 练习 2：TensorBoard 监控（30 分钟）

```bash
# 1. 启动 TensorBoard
tensorboard --logdir runs/detect/

# 2. 在浏览器打开 http://localhost:6006
# 3. 查看 loss 曲线和 mAP 曲线
# 4. 截图保存
```

### 练习 3：超参数调优（60 分钟）

```bash
# 1. 尝试不同学习率
yolo detect train data=data.yaml model=yolov8s.pt lr0=0.001

# 2. 尝试不同批次
yolo detect train data=data.yaml model=yolov8s.pt batch=8

# 3. 比较结果
```

### 练习 4：模型对比（60 分钟）

```bash
# 1. 训练 nano 模型
yolo detect train data=data.yaml model=yolov8n.pt epochs=50

# 2. 训练 small 模型
yolo detect train data=data.yaml model=yolov8s.pt epochs=50

# 3. 训练 medium 模型
yolo detect train data=data.yaml model=yolov8m.pt epochs=50

# 4. 对比 mAP 和速度
```

---

## ✅ 训练检查清单

训练完成后，确保：

- [ ] 训练 loss 下降并收敛
- [ ] 验证 mAP50 > 0.7
- [ ] 验证 mAP50-95 > 0.4
- [ ] 没有明显过拟合
- [ ] 保存了 best.pt 和 last.pt
- [ ] TensorBoard 曲线正常

---

## 🔗 相关资源

- [Ultralytics 训练文档](https://docs.ultralytics.com/modes/train/)
- [YOLOv8 超参数调优指南](https://docs.ultralytics.com/guides/hyperparameter-tuning/)
- [TensorBoard 使用教程](https://www.tensorflow.org/tensorboard)

---

**下一步：[11-Model-Evaluation - 模型评估](../11-model-evaluation/README.md)** 🚀
