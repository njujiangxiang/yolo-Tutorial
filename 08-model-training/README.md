# 08-Model Training - 模型训练

> 学习训练和优化 YOLO 模型

---

## 🎯 学习目标

- ✅ 训练配置
- ✅ 迁移学习
- ✅ 训练监控
- ✅ 性能调优

---

## 🚀 训练流程

### 1. 准备数据

```yaml
# data.yaml
path: /path/to/dataset
train: images/train
val: images/val

nc: 3
names: ['class1', 'class2', 'class3']
```

### 2. 配置训练参数

```python
model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    workers=8,
    optimizer='SGD',
    lr0=0.01,
    patience=50,
    save_period=10,
    device=0  # GPU
)
```

### 3. 训练监控

```bash
# 使用 TensorBoard
tensorboard --logdir=runs/detect/train

# 查看训练结果
ls runs/detect/train/weights/
```

---

## 📊 训练参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| epochs | 训练轮数 | 100-300 |
| imgsz | 输入尺寸 | 640 |
| batch | 批次大小 | 16-64 |
| lr0 | 初始学习率 | 0.01 |
| patience | 早停耐心值 | 50 |
| augmentation | 数据增强 | True |

---

## 🔧 优化技巧

### 1. 迁移学习

```python
# 使用预训练权重
model = YOLO('yolov8n.pt')
```

### 2. 混合精度训练

```python
model.train(amp=True)  # 加速训练
```

### 3. 多 GPU 训练

```python
model.train(device=[0, 1, 2, 3])
```

---

## 📝 练习

1. 训练一个自定义模型
2. 使用 TensorBoard 监控训练
3. 尝试不同超参数
4. 比较不同模型大小

---

**继续学习：[09-Model-Deployment](../09-model-deployment/README.md)** 🚀
