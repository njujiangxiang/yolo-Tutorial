# exercises/03-compare-yolo-versions.py
"""
比较 YOLOv5 和 YOLOv8

任务:
1. 分别加载 YOLOv5 和 YOLOv8
2. 推理同一张图片
3. 比较检测结果和速度
"""
import time
import cv2
import numpy as np


def create_test_image():
    """创建测试图片"""
    img = np.zeros((640, 640, 3), dtype=np.uint8)

    # 添加多个目标
    targets = [
        ((50, 50, 150, 150), (255, 0, 0)),
        ((250, 100, 350, 200), (0, 255, 0)),
        ((450, 150, 550, 250), (0, 0, 255)),
        ((100, 300, 200, 400), (255, 255, 0)),
        ((300, 350, 400, 450), (255, 0, 255)),
    ]

    for (x1, y1, x2, y2), color in targets:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

    return img


def benchmark_yolov8(img, num_runs=3):
    """基准测试 YOLOv8"""
    from ultralytics import YOLO

    print("\n加载 YOLOv8n 模型...")
    model = YOLO('yolov8n.pt')

    # 预热
    _ = model(img, verbose=False)

    # 基准测试
    times = []
    for i in range(num_runs):
        start = time.time()
        results = model(img, verbose=False)
        end = time.time()
        times.append(end - start)

    avg_time = np.mean(times)
    result = results[0]
    num_detections = len(result.boxes) if result.boxes else 0

    return avg_time, num_detections, "YOLOv8n"


def benchmark_yolov5(img, num_runs=3):
    """基准测试 YOLOv5"""
    import torch

    print("\n加载 YOLOv5s 模型...")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
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
    df = results.pandas().xyxy[0]
    num_detections = len(df)

    return avg_time, num_detections, "YOLOv5s"


def main():
    print("=" * 50)
    print("YOLOv5 vs YOLOv8 比较")
    print("=" * 50)

    # 创建测试图片
    print("\n创建测试图片...")
    img = create_test_image()
    cv2.imwrite('test_comparison.jpg', img)
    print("测试图片已保存：test_comparison.jpg")

    results = []

    # 测试 YOLOv8
    try:
        v8_result = benchmark_yolov8(img)
        results.append(v8_result)
        print(f"  平均推理时间：{v8_result[0]:.4f} 秒")
        print(f"  检测目标数：{v8_result[1]}")
    except Exception as e:
        print(f"YOLOv8 测试失败：{e}")

    # 测试 YOLOv5
    try:
        v5_result = benchmark_yolov5(img)
        results.append(v5_result)
        print(f"  平均推理时间：{v5_result[0]:.4f} 秒")
        print(f"  检测目标数：{v5_result[1]}")
    except Exception as e:
        print(f"YOLOv5 测试失败：{e}")

    # 汇总比较
    if len(results) >= 2:
        print("\n" + "=" * 50)
        print("比较结果:")
        print("=" * 50)

        v8_time, v8_det, v8_name = results[0]
        v5_time, v5_det, v5_name = results[1]

        print(f"{'模型':<15} {'时间 (s)':<10} {'检测数':<10} {'FPS':<10}")
        print("-" * 45)
        print(f"{v8_name:<15} {v8_time:<10.4f} {v8_det:<10} {1/v8_time:<10.1f}")
        print(f"{v5_name:<15} {v5_time:<10.4f} {v5_det:<10} {1/v5_time:<10.1f}")

        # 相对性能
        speed_diff = (v5_time - v8_time) / v5_time * 100
        print(f"\nYOLOv8 比 YOLOv5 {'快' if speed_diff > 0 else '慢'} {abs(speed_diff):.1f}%")


if __name__ == "__main__":
    main()
