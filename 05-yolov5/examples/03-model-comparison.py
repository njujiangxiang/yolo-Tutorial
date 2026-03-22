# exercises/03-model-comparison.py
"""
比较 YOLOv5 不同规格模型

任务:
1. 加载 yolov5n, yolov5s, yolov5m
2. 推理同一张图片
3. 比较速度和精度
"""
import torch
import time
import cv2
import numpy as np


def create_test_image():
    """创建测试图片"""
    img = np.zeros((640, 640, 3), dtype=np.uint8)

    # 添加多个目标
    targets = [
        ((50, 50, 150, 150), (255, 0, 0)),      # 左上
        ((250, 100, 350, 200), (0, 255, 0)),    # 中上
        ((450, 150, 550, 250), (0, 0, 255)),    # 右上
        ((100, 300, 200, 400), (255, 255, 0)),  # 左中
        ((300, 350, 400, 450), (255, 0, 255)),  # 中中
        ((500, 400, 600, 500), (0, 255, 255)),  # 右中
        ((150, 500, 250, 600), (128, 128, 128)),# 左下
    ]

    for (x1, y1, x2, y2), color in targets:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

    return img


def benchmark_model(model_name, img, num_runs=5):
    """
    基准测试模型

    参数:
        model_name: 模型名称
        img: 测试图片
        num_runs: 运行次数

    返回:
        平均推理时间，检测结果
    """
    print(f"\n加载模型：{model_name}")
    model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
    model.eval()

    # 预热
    _ = model(img)

    # 基准测试
    times = []
    for i in range(num_runs):
        start = time.time()
        results = model(img)
        end = time.time()
        times.append(end - start)

    avg_time = np.mean(times)
    detections = results.pandas().xyxy[0]

    return avg_time, detections


def main():
    print("=" * 50)
    print("YOLOv5 模型比较")
    print("=" * 50)

    # 创建测试图片
    print("\n创建测试图片...")
    img = create_test_image()
    cv2.imwrite('test_comparison.jpg', img)
    print("测试图片已保存：test_comparison.jpg")

    # 要比较的模型
    models = [
        ('yolov5n', 'Nano - 最小最快'),
        ('yolov5s', 'Small - 平衡'),
        ('yolov5m', 'Medium - 更准确'),
    ]

    results = []

    print("\n" + "=" * 50)
    print("开始基准测试")
    print("=" * 50)

    for model_name, description in models:
        print(f"\n测试 {model_name} ({description})")

        avg_time, detections = benchmark_model(model_name, img)
        detection_count = len(detections)

        print(f"  平均推理时间：{avg_time:.4f} 秒")
        print(f"  检测目标数：{detection_count}")

        # 计算 FPS
        fps = 1 / avg_time
        print(f"  FPS: {fps:.1f}")

        results.append({
            'model': model_name,
            'description': description,
            'avg_time': avg_time,
            'fps': fps,
            'detections': detection_count
        })

    # 汇总比较
    print("\n" + "=" * 50)
    print("模型比较汇总")
    print("=" * 50)
    print(f"{'模型':<12} {'描述':<20} {'时间 (s)':<10} {'FPS':<10} {'检测数':<8}")
    print("-" * 60)

    for r in results:
        print(f"{r['model']:<12} {r['description']:<20} "
              f"{r['avg_time']:<10.4f} {r['fps']:<10.1f} {r['detections']:<8}")

    # 相对性能
    print("\n" + "=" * 50)
    print("相对性能 (以 yolov5n 为基准)")
    print("=" * 50)

    baseline = results[0]
    for r in results:
        time_ratio = r['avg_time'] / baseline['avg_time']
        fps_ratio = r['fps'] / baseline['fps']
        print(f"{r['model']}: {time_ratio:.2f}x 时间，{fps_ratio:.2f}x FPS")

    # 选择建议
    print("\n" + "=" * 50)
    print("选择建议:")
    print("=" * 50)
    print("yolov5n: 边缘设备、实时性要求极高场景")
    print("yolov5s: 平衡速度与精度，推荐首选")
    print("yolov5m: 精度要求较高、有 GPU 资源")
    print("yolov5l/x: 最高精度要求、离线处理")

    print("\n" + "=" * 50)
    print("使用方法:")
    print("=" * 50)
    print(f"1. 运行比较：python {__file__}")
    print(f"2. 查看结果图片：test_comparison.jpg")


if __name__ == "__main__":
    main()
