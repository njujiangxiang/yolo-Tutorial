# examples/video-inference.py
"""
视频推理

对视频文件进行逐帧推理
"""
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
