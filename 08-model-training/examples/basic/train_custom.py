"""
YOLOv8 自定义训练示例
"""

from ultralytics import YOLO

def train_custom_model():
    """训练自定义模型"""
    # 加载预训练模型
    model = YOLO('yolov8n.pt')
    
    # 训练
    model.train(
        data='data.yaml',      # 数据集配置
        epochs=100,            # 训练轮数
        imgsz=640,            # 输入尺寸
        batch=16,             # 批次大小
        device=0,             # GPU
        workers=8,            # 数据加载线程
        optimizer='SGD',      # 优化器
        lr0=0.01,             # 初始学习率
        patience=50,          # 早停耐心值
        save_period=10,       # 保存间隔
        augmentation=True,    # 数据增强
    )
    
    # 验证
    metrics = model.val()
    print(f"mAP: {metrics.box.map:.3f}")

if __name__ == "__main__":
    train_custom_model()
