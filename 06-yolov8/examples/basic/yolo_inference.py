"""
YOLOv8 基础推理示例
"""

from ultralytics import YOLO
import cv2

def basic_inference():
    """基础推理示例"""
    # 加载模型
    model = YOLO('yolov8n.pt')
    
    # 推理
    results = model('image.jpg')
    
    # 显示结果
    results[0].show()
    
    # 获取检测结果
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 边界框
            x1, y1, x2, y2 = box.xyxy[0]
            
            # 类别
            cls = int(box.cls[0])
            class_name = model.names[cls]
            
            # 置信度
            conf = float(box.conf[0])
            
            print(f"检测到：{class_name} ({conf:.2f})")

def video_inference():
    """视频推理示例"""
    model = YOLO('yolov8n.pt')
    
    # 打开视频
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 推理
        results = model(frame)
        
        # 显示结果
        annotated = results[0].plot()
        cv2.imshow('YOLOv8 Detection', annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("YOLOv8 基础推理示例")
    basic_inference()
