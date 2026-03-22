# examples/camera-inference.py
"""
摄像头实时推理

从摄像头捕获画面并进行实时推理
"""
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
