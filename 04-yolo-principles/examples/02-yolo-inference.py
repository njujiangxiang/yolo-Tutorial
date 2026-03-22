# exercises/02-yolo-inference.py
"""
使用 YOLOv8 进行推理

任务:
1. 加载 YOLOv8 模型
2. 推理一张图片
3. 绘制并保存结果
"""
from ultralytics import YOLO
import cv2
import numpy as np


def main():
    print("=" * 50)
    print("YOLO 推理演示")
    print("=" * 50)

    # 创建测试图片
    print("\n创建测试图片...")
    img = np.zeros((640, 640, 3), dtype=np.uint8)

    # 画一些形状模拟目标
    cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), -1)
    cv2.circle(img, (400, 300), 50, (0, 255, 0), -1)
    cv2.rectangle(img, (300, 400), (450, 550), (0, 0, 255), -1)

    cv2.imwrite('test_image.jpg', img)
    print("测试图片已保存：test_image.jpg")

    # 加载模型
    print("\n加载 YOLOv8n 模型...")
    model = YOLO('yolov8n.pt')
    print("模型加载完成！")

    # 推理
    print("\n正在推理...")
    results = model(img, verbose=False)

    # 获取结果
    result = results[0]

    # 打印检测结果
    print("\n" + "=" * 50)
    print("检测结果:")
    print("=" * 50)

    boxes = result.boxes
    print(f"检测到 {len(boxes)} 个目标")

    if len(boxes) > 0:
        print(f"\n{'类别':<15} {'置信度':<10} {'位置':<30}")
        print("-" * 55)

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_name = model.names[cls]
            print(f"{class_name:<15} {conf:<10.4f} [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

    # 显示并保存结果
    result.show()
    result.save()

    print("\n" + "=" * 50)
    print("提示:")
    print("=" * 50)
    print("1. 替换为自己的图片：model('your_image.jpg')")
    print("2. 调整置信度阈值：model(img, conf=0.5)")
    print("3. 批量推理：model(['img1.jpg', 'img2.jpg'])")


if __name__ == "__main__":
    main()
