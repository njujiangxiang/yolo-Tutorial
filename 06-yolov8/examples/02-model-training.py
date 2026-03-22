# exercises/02-model-training.py
"""
训练 YOLOv8 模型

任务:
1. 配置数据集
2. 设置训练参数
3. 开始训练
4. 查看训练结果
"""
from ultralytics import YOLO
import os
import yaml


def create_sample_dataset():
    """创建示例数据集配置"""
    # 创建目录结构
    os.makedirs('sample_data/train/images', exist_ok=True)
    os.makedirs('sample_data/val/images', exist_ok=True)
    os.makedirs('sample_data/train/labels', exist_ok=True)
    os.makedirs('sample_data/val/labels', exist_ok=True)

    # 创建数据集配置
    data_config = {
        'path': './sample_data',
        'train': 'train/images',
        'val': 'val/images',
        'nc': 2,
        'names': ['defect', 'component']
    }

    with open('sample_data.yaml', 'w') as f:
        yaml.dump(data_config, f)

    print("示例数据集配置已创建：sample_data.yaml")
    return 'sample_data.yaml'


def main():
    print("=" * 50)
    print("YOLOv8 模型训练")
    print("=" * 50)

    # 创建示例数据集配置
    print("\n创建示例数据集配置...")
    data_config = create_sample_dataset()

    # 加载模型
    print("\n加载 YOLOv8n 预训练模型...")
    model = YOLO('yolov8n.pt')
    print("模型加载完成！")

    # 训练配置
    print("\n" + "=" * 50)
    print("训练配置:")
    print("=" * 50)

    training_args = {
        'data': data_config,
        'epochs': 50,  # 示例用较少轮数
        'batch': 8,    # 小批量用于演示
        'imgsz': 640,
        'device': 0 if torch.cuda.is_available() else 'cpu',
        'workers': 4,
        'optimizer': 'SGD',
        'lr0': 0.01,
        'patience': 20,
        'project': 'runs/train',
        'name': 'yolov8_demo',
        'exist_ok': True,
        'verbose': True,
        'plots': True,  # 生成训练图表
    }

    print("训练参数:")
    for key, value in training_args.items():
        print(f"  {key}: {value}")

    # 检查 GPU
    if torch.cuda.is_available():
        print(f"\n使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("\n未检测到 GPU，使用 CPU 训练（速度较慢）")

    # 开始训练
    print("\n" + "=" * 50)
    print("开始训练...")
    print("=" * 50)

    try:
        results = model.train(**training_args)

        # 训练完成
        print("\n" + "=" * 50)
        print("训练完成!")
        print("=" * 50)

        # 显示训练结果
        print(f"训练轮数：{results.epoch}")
        print(f"最终损失：{results.loss.item():.4f}")

        # 模型保存位置
        best_model = 'runs/train/yolov8_demo/weights/best.pt'
        last_model = 'runs/train/yolov8_demo/weights/last.pt'

        print(f"\n模型已保存:")
        print(f"  最佳模型：{best_model}")
        print(f"  最新模型：{last_model}")

        # 训练图表
        print(f"\n训练图表：runs/train/yolov8_demo/results.png")

    except Exception as e:
        print(f"\n训练出错：{e}")
        print("\n提示:")
        print("1. 确保有足够的数据集图片")
        print("2. 检查数据集配置是否正确")
        print("3. 确保有足够的内存/显存")


if __name__ == "__main__":
    import torch
    main()
