# exercises/04-custom-threshold.py
"""
调整置信度阈值和 NMS 阈值

任务:
1. 使用不同置信度阈值推理
2. 观察检测结果变化
3. 找到最适合的阈值
"""
from ultralytics import YOLO
import cv2
import numpy as np


def create_test_scene():
    """创建测试场景"""
    img = np.zeros((640, 640, 3), dtype=np.uint8)

    # 添加不同大小的目标
    targets = [
        # 大目标 - 高置信度
        ((100, 100, 250, 250), (255, 0, 0)),
        # 中目标 - 中置信度
        ((300, 150, 400, 250), (0, 255, 0)),
        # 小目标 - 低置信度
        ((450, 300, 500, 350), (0, 0, 255)),
        # 重叠目标
        ((200, 350, 280, 430), (255, 255, 0)),
        ((220, 370, 300, 450), (255, 0, 255)),
        # 边缘目标
        ((10, 500, 80, 570), (0, 255, 255)),
    ]

    for (x1, y1, x2, y2), color in targets:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

    # 添加一些噪声
    for _ in range(20):
        x, y = np.random.randint(0, 640), np.random.randint(0, 640)
        w, h = np.random.randint(10, 40), np.random.randint(10, 40)
        noise_color = np.random.randint(100, 200, 3)
        cv2.rectangle(img, (x, y), (x + w, y + h), noise_color.tolist(), -1)

    return img


def test_thresholds(img, conf_thresholds, iou_thresholds):
    """
    测试不同阈值组合

    参数:
        img: 测试图片
        conf_thresholds: 置信度阈值列表
        iou_thresholds: NMS IoU 阈值列表
    """
    print("\n加载模型...")
    model = YOLO('yolov8n.pt')

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print(f"{'置信度':<12} {'NMS IoU':<12} {'检测数':<10}")
    print("-" * 34)

    results = []

    for conf in conf_thresholds:
        for iou in iou_thresholds:
            results = model(img, conf=conf, iou=iou, verbose=False)
            num_det = len(results[0].boxes) if results[0].boxes else 0
            print(f"{conf:<12.2f} {iou:<12.2f} {num_det:<10}")
            results.append({
                'conf': conf,
                'iou': iou,
                'detections': num_det
            })

    return results


def visualize_results(img, conf_threshold=0.25, iou_threshold=0.45):
    """可视化特定阈值下的结果"""
    print(f"\n可视化阈值设置：conf={conf_threshold}, iou={iou_threshold}")

    model = YOLO('yolov8n.pt')
    results = model(img, conf=conf_threshold, iou=iou_threshold, verbose=False)

    # 保存结果
    output_path = f'threshold_vis_conf{conf_threshold}_iou{iou_threshold}.jpg'
    results[0].save(output_path)
    print(f"结果已保存：{output_path}")


def main():
    print("=" * 50)
    print("置信度阈值和 NMS 阈值调整")
    print("=" * 50)

    # 创建测试场景
    print("\n创建测试场景...")
    img = create_test_scene()
    cv2.imwrite('test_threshold.jpg', img)
    print("测试图片已保存：test_threshold.jpg")

    # 测试不同阈值组合
    conf_thresholds = [0.1, 0.25, 0.5, 0.75]
    iou_thresholds = [0.3, 0.45, 0.6]

    print("\n测试不同阈值组合...")
    test_thresholds(img, conf_thresholds, iou_thresholds)

    # 可视化典型设置
    print("\n" + "=" * 50)
    print("可视化典型阈值设置")
    print("=" * 50)

    visualize_results(img, conf_threshold=0.25, iou_threshold=0.45)  # 默认
    visualize_results(img, conf_threshold=0.1, iou_threshold=0.45)   # 低置信度
    visualize_results(img, conf_threshold=0.5, iou_threshold=0.45)   # 高置信度

    print("\n" + "=" * 50)
    print("阈值选择建议:")
    print("=" * 50)
    print("conf=0.10: 检测更多目标，但可能有误检")
    print("conf=0.25: 平衡设置，推荐默认")
    print("conf=0.50: 只保留高置信度检测，减少误检")
    print("conf=0.75: 非常严格，可能漏检")
    print("")
    print("iou=0.30: 去除更多重叠框")
    print("iou=0.45: 平衡设置，推荐默认")
    print("iou=0.60: 保留更多重叠框")


if __name__ == "__main__":
    main()
