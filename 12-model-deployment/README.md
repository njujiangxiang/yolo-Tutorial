# 12-Model-Deployment - 模型部署

> 学习将 YOLO 模型部署到生产环境

---

## 🎯 学习目标

完成本章后，你将能够：

- ✅ 导出模型为 ONNX 格式
- ✅ 使用 ONNX Runtime 推理
- ✅ 图片/视频/摄像头推理
- ✅ 部署为 FastAPI Web 服务
- ✅ 了解 TensorRT 加速
- ✅ 选择适合的部署方案

---

## 📦 部署方案对比

| 方案 | 速度 | 易用性 | 适用场景 |
|------|------|--------|----------|
| PyTorch | 中 | 简单 | 原型验证 |
| ONNX Runtime | 中高 | 简单 | 通用部署 |
| TensorRT | 很快 | 中等 | NVIDIA GPU |
| OpenVINO | 中快 | 中等 | Intel CPU |
| FastAPI Web | 中 | 简单 | 服务端 API |

---

## 1️⃣ 导出模型为 ONNX

### 简单导出

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/detect/defect-train/weights/best.pt')

# 导出为 ONNX
model.export(format='onnx', simplify=True)

# 输出：runs/detect/defect-train/weights/best.onnx
```

### 完整导出配置

```python
# examples/export-onnx.py
from ultralytics import YOLO

def export_model():
    """
    导出模型为 ONNX 格式
    """
    # 加载模型
    model = YOLO('runs/detect/defect-train/weights/best.pt')

    # 导出
    onnx_path = model.export(
        format='onnx',
        simplify=True,        # 简化模型（推荐）
        dynamic_axes=False,   # 固定输入尺寸
        opset=12,            # ONNX 算子版本
    )

    print(f"模型已导出：{onnx_path}")
    return onnx_path

if __name__ == "__main__":
    export_model()
```

### 验证导出成功

```bash
# 使用 onnxsim 验证
python -m onnxsim best.onnx best_simplified.onnx

# 使用 Netron 可视化（需要安装）
# https://github.com/lutzroeder/netron
```

---

## 2️⃣ ONNX Runtime 推理

### 基础推理

```python
# examples/onnx-inference.py
import onnxruntime as ort
import cv2
import numpy as np

class ONNXDetector:
    """ONNX 模型推理器"""

    def __init__(self, onnx_path):
        # 加载模型
        self.session = ort.InferenceSession(onnx_path)

        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        print(f"输入形状：{self.input_shape}")

    def preprocess(self, image, img_size=640):
        """
        预处理图片

        步骤:
        1. BGR 转 RGB
        2. 调整尺寸
        3. 归一化到 [0, 1]
        4. 添加 batch 维度
        """
        # BGR → RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 调整尺寸
        img_resized = cv2.resize(img_rgb, (img_size, img_size))

        # 归一化
        img_normalized = img_resized.astype(np.float32) / 255.0

        # HWC → CHW
        img_transposed = np.transpose(img_normalized, (2, 0, 1))

        # 添加 batch 维度 (1, 3, H, W)
        img_batch = np.expand_dims(img_transposed, axis=0)

        return img_batch

    def detect(self, image, conf_threshold=0.25):
        """
        检测图片中的目标

        参数:
            image: OpenCV 图片数组
            conf_threshold: 置信度阈值

        返回:
            检测结果列表
        """
        # 预处理
        input_tensor = self.preprocess(image)

        # 推理
        outputs = self.session.run(
            None,
            {self.input_name: input_tensor}
        )

        # 后处理（解析输出）
        results = self.postprocess(outputs, conf_threshold)

        return results

    def postprocess(self, outputs, conf_threshold):
        """
        解析模型输出

        YOLOv8 输出格式：
        - [batch, 4+nc, 8400]
        - 4 个框坐标 + nc 个类别概率
        """
        # 简化处理，实际使用建议用 ultralytics 的后处理
        predictions = outputs[0][0]  # (4+nc, 8400)

        results = []
        for i in range(predictions.shape[1]):
            box = predictions[:4, i]
            scores = predictions[4:, i]

            if scores.max() > conf_threshold:
                class_id = scores.argmax()
                conf = scores[class_id]
                results.append({
                    'box': box,
                    'class': int(class_id),
                    'conf': float(conf)
                })

        return results

def main():
    # 加载模型
    detector = ONNXDetector('best.onnx')

    # 读取图片
    img = cv2.imread('test.jpg')

    # 检测
    results = detector.detect(img)

    print(f"检测到 {len(results)} 个目标")

if __name__ == "__main__":
    main()
```

### 使用 Ultralytics 简化推理

```python
# examples/simple-onnx-inference.py
from ultralytics import YOLO
import cv2

# 加载 ONNX 模型
model = YOLO('best.onnx')

# 推理
results = model.predict('test.jpg')

# 获取结果
for r in results:
    boxes = r.boxes  # 边界框
    probs = r.probs  # 类别概率
    names = r.names  # 类别名称

    # 画出结果
    im = r.plot()

    # 保存
    cv2.imwrite('result.jpg', im)

    # 打印结果
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"检测到 {names[cls]} ({conf:.2f}) at [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
```

---

## 3️⃣ 图片/视频/摄像头推理

### 图片推理

```python
# examples/image-inference.py
from ultralytics import YOLO
import cv2
from pathlib import Path

def batch_inference(model_path, image_dir, output_dir):
    """
    批量推理图片

    参数:
        model_path: 模型路径
        image_dir: 图片目录
        output_dir: 输出目录
    """
    model = YOLO(model_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 批量推理
    results = model.predict(
        source=image_dir,
        save=True,
        save_dir=output_dir,
        conf=0.25,
        verbose=True
    )

    print(f"推理完成，结果保存在：{output_dir}")

if __name__ == "__main__":
    batch_inference(
        model_path='runs/detect/defect-train/weights/best.pt',
        image_dir='datasets/val/images/',
        output_dir='inference_results/'
    )
```

### 视频推理

```python
# examples/video-inference.py
from ultralytics import YOLO
import cv2

def video_inference(model_path, video_path, output_path):
    """
    视频推理

    参数:
        model_path: 模型路径
        video_path: 输入视频路径
        output_path: 输出视频路径
    """
    model = YOLO(model_path)

    # 打开视频
    cap = cv2.VideoCapture(video_path)

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"开始处理视频：{total_frames} 帧")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 推理
        results = model.predict(frame, conf=0.25, verbose=False)

        # 绘制结果
        result_frame = results[0].plot()

        # 写入输出
        out.write(result_frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"已处理 {frame_count}/{total_frames} 帧")

    cap.release()
    out.release()
    print(f"视频处理完成：{output_path}")

if __name__ == "__main__":
    video_inference(
        model_path='runs/detect/defect-train/weights/best.pt',
        video_path='test_video.mp4',
        output_path='output_video.mp4'
    )
```

### 摄像头实时推理

```python
# examples/camera-inference.py
from ultralytics import YOLO
import cv2

def camera_inference(model_path, camera_id=0):
    """
    摄像头实时推理

    参数:
        model_path: 模型路径
        camera_id: 摄像头 ID（0 为默认摄像头）
    """
    model = YOLO(model_path)

    # 打开摄像头
    cap = cv2.VideoCapture(camera_id)

    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("按 'q' 键退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 推理
        results = model.predict(frame, conf=0.25, verbose=False)

        # 绘制结果
        result_frame = results[0].plot()

        # 显示
        cv2.imshow('YOLO Detection', result_frame)

        # 按 q 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    camera_inference('runs/detect/defect-train/weights/best.pt')
```

---

## 4️⃣ Web 服务部署（FastAPI）

### 安装依赖

```bash
pip install fastapi uvicorn python-multipart pillow
```

### 创建 Web 服务

```python
# examples/fastapi-server.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io

app = FastAPI(title="YOLO 缺陷检测 API")

# 加载模型
model = YOLO('runs/detect/defect-train/weights/best.pt')

@app.get("/")
async def root():
    """API 根路径"""
    return {
        "message": "YOLO 缺陷检测 API",
        "endpoints": {
            "POST /predict": "上传图片进行检测",
            "GET /health": "健康检查"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    上传图片进行缺陷检测

    返回:
    - detections: 检测结果列表
    - image: 绘制结果的图片（base64）
    """
    # 读取图片
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # 转换为 OpenCV 格式
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # 推理
    results = model.predict(img_cv, conf=0.25)

    # 解析结果
    result = results[0]
    detections = []

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        detections.append({
            "class_id": cls,
            "class_name": result.names[cls],
            "confidence": conf,
            "bbox": [x1, y1, x2, y2]
        })

    # 绘制结果
    result_image = result.plot()
    _, buffer = cv2.imencode('.jpg', result_image)

    return JSONResponse({
        "detections": detections,
        "image": f"data:image/jpeg;base64,{buffer.tobytes().hex()}"
    })

@app.post("/predict/stream")
async def predict_stream(file: UploadFile = File(...)):
    """
    返回绘制结果图片的流
    """
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    results = model.predict(img_cv, conf=0.25)
    result_image = results[0].plot()

    # 转换为 JPEG
    _, buffer = cv2.imencode('.jpg', result_image)
    image_bytes = buffer.tobytes()

    return StreamingResponse(
        io.BytesIO(image_bytes),
        media_type="image/jpeg"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 启动服务

```bash
# 启动服务
python examples/fastapi-server.py

# 或使用 uvicorn
uvicorn examples.fastapi-server:app --reload --host 0.0.0.0 --port 8000

# 访问 http://localhost:8000/docs 查看 API 文档
```

### 测试 API

```bash
# 使用 curl 测试
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_image.jpg"

# 或使用 Python
```

```python
# examples/test-api.py
import requests

# 上传图片
with open('test.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f}
    )

# 打印结果
print(response.json())
```

---

## 5️⃣ TensorRT 加速（NVIDIA GPU）

### 导出为 TensorRT

```python
# examples/export-tensorrt.py
from ultralytics import YOLO

def export_to_tensorrt():
    """
    导出为 TensorRT Engine
    """
    model = YOLO('runs/detect/defect-train/weights/best.pt')

    # 导出
    trt_path = model.export(
        format='engine',
        device=0,
        half=True,  # FP16
        simplify=True,
    )

    print(f"TensorRT 模型已导出：{trt_path}")
    return trt_path

if __name__ == "__main__":
    export_to_tensorrt()
```

### TensorRT 推理

```python
# 使用 TensorRT 模型推理
from ultralytics import YOLO

# 加载 TensorRT 模型
model = YOLO('best.engine')

# 推理
results = model.predict('test.jpg')

# TensorRT 比 ONNX 快 2-3 倍
```

---

## 📊 部署方案选择指南

### 场景 1：原型验证/研究

```yaml
推荐：PyTorch 原始模型
原因:
  - 最简单
  - 不需要转换
  - 可以直接用训练代码
```

### 场景 2：服务器部署

```yaml
推荐：FastAPI + ONNX
原因:
  - 跨平台
  - 性能足够
  - 易于维护
```

### 场景 3：边缘设备（NVIDIA Jetson）

```yaml
推荐：TensorRT
原因:
  - 最快
  - 功耗低
  - 专为 NVIDIA 优化
```

### 场景 4：Intel CPU 部署

```yaml
推荐：OpenVINO
原因:
  - Intel 优化
  - 无需 GPU
  - 性能较好
```

### 场景 5：移动端（iOS/Android）

```yaml
推荐：CoreML (iOS) / TFLite (Android)
原因:
  - 原生支持
  - 功耗低
```

---

## 📝 实战练习

### 练习 1：导出 ONNX 模型（15 分钟）

```bash
# 1. 导出模型
python examples/export-onnx.py

# 2. 验证文件大小
ls -lh runs/detect/defect-train/weights/
```

### 练习 2：ONNX 推理（30 分钟）

```bash
# 1. 运行推理脚本
python examples/onnx-inference.py

# 2. 对比 PyTorch 和 ONNX 的速度
```

### 练习 3：批量图片推理（30 分钟）

```bash
# 1. 准备测试图片
# 2. 批量推理
python examples/image-inference.py

# 3. 查看结果
```

### 练习 4：创建 Web 服务（60 分钟）

```bash
# 1. 安装依赖
pip install fastapi uvicorn

# 2. 启动服务
python examples/fastapi-server.py

# 3. 访问 http://localhost:8000/docs
# 4. 使用 API 文档测试
```

---

## ✅ 部署检查清单

部署完成后，确保：

- [ ] ONNX 模型导出成功
- [ ] ONNX 推理结果与 PyTorch 一致
- [ ] 图片/视频推理正常
- [ ] Web 服务可以访问
- [ ] API 返回正确结果
- [ ] 推理速度满足要求

---

## 🔗 相关资源

- [ONNX Runtime 文档](https://onnxruntime.ai/)
- [FastAPI 官方教程](https://fastapi.tiangolo.com/tutorial/)
- [TensorRT 文档](https://docs.nvidia.com/deeplearning/tensorrt/)
- [OpenVINO 工具包](https://docs.openvino.ai/)

---

**下一步：[14-Projects - 实战项目](../14-projects/README.md)** 🚀
