# 06-YOLOv8 - YOLOv8 实战

> 学习使用最新的 YOLOv8 进行目标检测

---

## 🎯 学习目标

- ✅ YOLOv8 新特性
- ✅ Ultralytics 框架
- ✅ 训练与验证
- ✅ 导出与部署

---

## 🆕 YOLOv8 新特性

- 无锚框设计
- 改进的损失函数
- 更好的骨干网络
- 支持多种任务 (检测、分割、分类、姿态)

---

## 💻 示例代码

### 1. 安装

```bash
pip install ultralytics
```

### 2. 基础推理

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('yolov8n.pt')

# 推理
results = model('image.jpg')

# 显示结果
results[0].show()
```

### 3. 训练

```python
# 加载模型
model = YOLO('yolov8n.pt')

# 训练
model.train(
    data='coco8.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

### 4. 验证

```python
# 验证模型
metrics = model.val()
print(f"mAP: {metrics.box.map}")
```

### 5. 导出

```python
# 导出为 ONNX
model.export(format='onnx')

# 导出为 TensorRT
model.export(format='engine')
```

---

## 📊 模型规格

| 模型 | 参数量 | 速度 | 精度 |
|------|--------|------|------|
| YOLOv8n | 3.2M | 最快 | 基础 |
| YOLOv8s | 11.2M | 快 | 好 |
| YOLOv8m | 25.9M | 中 | 更好 |
| YOLOv8l | 43.7M | 慢 | 优秀 |
| YOLOv8x | 68.2M | 最慢 | 最佳 |

---

## 📝 练习

1. 使用 YOLOv8 检测图片
2. 训练自定义模型
3. 导出为 ONNX 格式
4. 比较不同模型规格

---

**继续学习：[07-Custom-Dataset](../07-custom-dataset/README.md)** 🚀
