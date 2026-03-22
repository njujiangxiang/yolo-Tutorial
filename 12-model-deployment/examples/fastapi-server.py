# examples/fastapi-server.py
"""
FastAPI Web 服务部署

将 YOLO 模型部署为 REST API 服务
"""
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

    return JSONResponse({"detections": detections})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
