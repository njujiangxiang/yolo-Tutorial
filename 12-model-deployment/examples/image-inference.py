# examples/image-inference.py
"""
批量图片推理

对目录中的所有图片进行推理
"""
from ultralytics import YOLO
import cv2
from pathlib import Path

def batch_inference(model_path, image_dir, output_dir):
    """
    批量推理图片

    参数:
        model_path: 模型路径
        image_dir: 图片目录
        output_dir: 输出目录
    """
    model = YOLO(model_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 批量推理
    results = model.predict(
        source=image_dir,
        save=True,
        save_dir=output_dir,
        conf=0.25,
        verbose=True
    )

    print(f"推理完成，结果保存在：{output_dir}")

if __name__ == "__main__":
    batch_inference(
        model_path='runs/detect/defect-train/weights/best.pt',
        image_dir='datasets/val/images/',
        output_dir='inference_results/'
    )
