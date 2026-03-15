"""
实时目标检测示例
"""

from ultralytics import YOLO
import cv2

def real_time_detection():
    """实时目标检测"""
    # 加载模型
    model = YOLO('yolov8n.pt')
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    print("按 'q' 键退出")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 推理
        results = model(frame, verbose=False)
        
        # 绘制结果
        annotated = results[0].plot()
        
        # 显示统计信息
        num_objects = len(results[0].boxes)
        cv2.putText(annotated, f'Objects: {num_objects}', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示画面
        cv2.imshow('Real-time Detection', annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_detection()
