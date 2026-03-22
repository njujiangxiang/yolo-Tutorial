# exercises/02-video-processing.py
"""
使用 YOLOv5 处理视频

任务:
1. 读取视频文件
2. 逐帧检测
3. 保存带标注的视频
"""
import torch
import cv2
import time


def process_video(video_path, output_path, conf_threshold=0.5):
    """
    处理视频文件

    参数:
        video_path: 输入视频路径
        output_path: 输出视频路径
        conf_threshold: 置信度阈值
    """
    print("=" * 50)
    print("YOLOv5 视频处理")
    print("=" * 50)

    # 加载模型
    print("\n正在加载 YOLOv5s 模型...")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.eval()
    print("模型加载完成！")

    # 打开视频
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"错误：无法打开视频 {video_path}")
        return

    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n视频信息:")
    print(f"  分辨率：{width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  总帧数：{total_frames}")

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"\n开始处理视频...")
    start_time = time.time()
    frame_count = 0
    detection_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 推理
        results = model(frame)

        # 获取检测结果
        df = results.pandas().xyxy[0]

        # 过滤低置信度检测
        detections = df[df['confidence'] > conf_threshold]

        if len(detections) > 0:
            detection_count += 1

        # 在帧上绘制检测结果
        results.render()

        # 获取渲染后的图像
        rendered_img = results.img[0]
        rendered_img = cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR)

        # 写入输出视频
        out.write(rendered_img)

        frame_count += 1

        # 显示进度
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            progress = frame_count / total_frames * 100
            print(f"  进度：{frame_count}/{total_frames} ({progress:.1f}%), "
                  f"已检测到有目标的帧：{detection_count}, "
                  f"速度：{frame_count/elapsed:.1f} FPS")

    # 释放资源
    cap.release()
    out.release()

    elapsed_time = time.time() - start_time
    print(f"\n" + "=" * 50)
    print(f"处理完成!")
    print(f"  输出文件：{output_path}")
    print(f"  处理时间：{elapsed_time:.1f} 秒")
    print(f"  处理速度：{frame_count/elapsed_time:.1f} FPS")
    print(f"  检测到目标的帧数：{detection_count}/{frame_count}")


def create_sample_video(output_path, duration=3):
    """创建示例视频用于测试"""
    print("正在创建示例视频...")

    # 创建视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    width, height = 640, 480
    total_frames = fps * duration

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 生成简单动画
    for i in range(total_frames):
        # 创建背景
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # 移动的形状
        x = int(100 + i * 2) % width
        y = int(100 + abs((i // 30) % 4) * 100)

        # 绘制移动的目标
        cv2.circle(frame, (x, y), 30, (0, 255, 0), -1)
        cv2.rectangle(frame, (x + 50, y - 30), (x + 110, y + 30), (255, 0, 0), -1)

        # 显示帧号
        cv2.putText(frame, f"Frame {i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame)

    out.release()
    print(f"示例视频已保存：{output_path}")


def main():
    import os
    import sys

    sample_video = 'sample_video.mp4'
    output_video = 'detection_result.mp4'

    print("=" * 50)
    print("YOLOv5 视频处理演示")
    print("=" * 50)

    # 检查是否有视频文件
    if len(sys.argv) > 1:
        # 使用命令行参数指定的视频
        video_path = sys.argv[1]
    elif os.path.exists(sample_video):
        # 使用现有视频
        video_path = sample_video
    else:
        # 创建示例视频
        print("\n未找到视频文件，创建示例视频...")
        create_sample_video(sample_video, duration=2)
        video_path = sample_video

    # 处理视频
    process_video(video_path, output_video, conf_threshold=0.3)

    print("\n" + "=" * 50)
    print("使用方法:")
    print("=" * 50)
    print(f"1. 处理默认视频：python {sys.argv[0]}")
    print(f"2. 处理自定义视频：python {sys.argv[0]} <视频路径>")
    print(f"3. 输出文件：{output_video}")


if __name__ == "__main__":
    import numpy as np
    main()
