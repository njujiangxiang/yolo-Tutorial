# 09-Model Deployment - 模型部署

> 学习将 YOLO 模型部署到生产环境

---

## 🎯 学习目标

- ✅ ONNX 导出
- ✅ TensorRT 加速
- ✅ OpenVINO 部署
- ✅ Web/移动端部署

---

## 📦 导出格式

| 格式 | 说明 | 适用场景 |
|------|------|----------|
| PyTorch | 原始格式 | 训练/研究 |
| ONNX | 通用格式 | 跨平台 |
| TensorRT | NVIDIA 加速 | 边缘部署 |
| OpenVINO | Intel 优化 | CPU 部署 |
| CoreML | Apple 优化 | iOS/macOS |
| TFLite | TensorFlow Lite | 移动端 |

---

## 💻 示例代码

### 1. ONNX 导出

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.export(format='onnx')
```

### 2. ONNX 推理

```python
import onnxruntime as ort
import numpy as np

# 加载模型
session = ort.InferenceSession('yolov8n.onnx')

# 推理
inputs = preprocess(image)
outputs = session.run(None, {'images': inputs})
```

### 3. TensorRT 导出

```python
# 导出为 TensorRT
model.export(format='engine', device=0)
```

---

## 📝 练习

1. 导出模型为 ONNX
2. 使用 ONNX Runtime 推理
3. 尝试 TensorRT 加速
4. 部署到 Web 服务

---

**继续学习：[10-Advanced-Topics](../10-advanced-topics/README.md)** 🚀
