# exercises/01-yolov5-inference.py
"""
使用 YOLOv5 进行推理

任务:
1. 加载 YOLOv5s 模型
2. 推理一张图片
3. 绘制并保存检测结果
"""
import torch
import cv2
import numpy as np


def main():
    print("=" * 50)
    print("YOLOv5 推理演示")
    print("=" * 50)

    # 加载模型
    print("\n正在加载 YOLOv5s 模型...")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    print("模型加载完成！")

    # 设置模型为评估模式
    model.eval()

    # 创建测试图片
    print("\n正在创建测试图片...")
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    # 画一些形状模拟目标
    cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), -1)  # 蓝色矩形
    cv2.circle(img, (400, 300), 50, (0, 255, 0), -1)  # 绿色圆形
    cv2.rectangle(img, (300, 400), (450, 550), (0, 0, 255), -1)  # 红色矩形

    # 保存测试图片
    cv2.imwrite('test_image.jpg', img)
    print("测试图片已保存：test_image.jpg")

    # 推理
    print("\n正在推理...")
    results = model(img)

    # 显示结果
    results.show()

    # 获取检测结果
    df = results.pandas().xyxy[0]
    print("\n检测结果:")
    print(df)

    # 保存结果图片
    output_path = 'yolov5_result.jpg'
    results.save()
    print(f"\n结果已保存：{output_path}")

    # 统计检测到的目标
    print("\n" + "=" * 50)
    print("检测统计:")
    print("=" * 50)
    if len(df) > 0:
        for name in df['name'].unique():
            count = len(df[df['name'] == name])
            print(f"  {name}: {count} 个")
    else:
        print("  未检测到目标")

    print("\n" + "=" * 50)
    print("提示:")
    print("=" * 50)
    print("1. 可以替换为自己的图片进行推理")
    print("2. 调整 conf 和 iou 参数改变检测灵敏度")
    print("3. 使用 model(imgs) 可以批量推理多张图片")


if __name__ == "__main__":
    main()
