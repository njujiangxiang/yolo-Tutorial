# exercises/03-model-export.py
"""
导出 YOLOv8 模型

任务:
1. 加载训练好的模型
2. 导出为 ONNX 格式
3. 验证导出结果
"""
from ultralytics import YOLO
import os


def main():
    print("=" * 50)
    print("YOLOv8 模型导出")
    print("=" * 50)

    # 加载模型
    print("\n加载 YOLOv8n 模型...")
    model = YOLO('yolov8n.pt')
    print("模型加载完成！")

    # 导出格式选择
    print("\n" + "=" * 50)
    print("支持的导出格式:")
    print("=" * 50)

    formats = [
        ('onnx', 'ONNX - 通用格式，支持多平台'),
        ('torchscript', 'TorchScript - PyTorch 原生'),
        ('openvino', 'OpenVINO - Intel 硬件加速'),
        ('tensorrt', 'TensorRT - NVIDIA GPU 加速'),
        ('coreml', 'CoreML - Apple 设备'),
        ('saved_model', 'SavedModel - TensorFlow'),
        ('pb', 'TensorFlow ProtoBuf'),
        ('tflite', 'TensorFlow Lite - 移动端'),
        ('engine', 'TensorRT Engine'),
        ('ncnn', 'NCNN - 移动端推理'),
    ]

    for fmt, desc in formats:
        print(f"  {fmt:<15} {desc}")

    # 导出为 ONNX
    print("\n" + "=" * 50)
    print("导出为 ONNX 格式...")
    print("=" * 50)

    try:
        # 基础导出
        onnx_path = model.export(format='onnx')
        print(f"ONNX 模型已保存：{onnx_path}")

        # 动态轴导出（支持可变输入尺寸）
        print("\n导出动态轴 ONNX 模型...")
        onnx_dynamic_path = model.export(format='onnx', dynamic=True)
        print(f"动态 ONNX 模型已保存：{onnx_dynamic_path}")

        # 验证导出
        print("\n" + "=" * 50)
        print("验证导出模型...")
        print("=" * 50)

        # 加载导出的模型
        exported_model = YOLO(onnx_path)

        # 测试推理
        import numpy as np
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        results = exported_model(test_img, verbose=False)

        print(f"推理成功！检测到 {len(results[0].boxes)} 个目标")
        print("ONNX 模型验证通过！")

        # 格式对比
        print("\n" + "=" * 50)
        print("文件格式对比:")
        print("=" * 50)

        original_size = os.path.getsize('yolov8n.pt')
        onnx_size = os.path.getsize(onnx_path)

        print(f"  PyTorch: {original_size / 1024 / 1024:.2f} MB")
        print(f"  ONNX:    {onnx_size / 1024 / 1024:.2f} MB")

    except Exception as e:
        print(f"\n导出失败：{e}")

    # 导出为其他格式示例
    print("\n" + "=" * 50)
    print("其他格式导出示例:")
    print("=" * 50)

    examples = """
# 导出为 TorchScript
model.export(format='torchscript')

# 导出为 TensorRT (需要 NVIDIA GPU)
model.export(format='engine', device=0, half=True)

# 导出为 OpenVINO
model.export(format='openvino')

# 导出为 CoreML (macOS)
model.export(format='coreml')

# 导出为 TFLite
model.export(format='tflite')

# 批量导出多种格式
model.export(format='onnx')
model.export(format='torchscript')
model.export(format='openvino')
    """
    print(examples)

    # 使用建议
    print("=" * 50)
    print("导出格式选择建议:")
    print("=" * 50)
    print("  ONNX:       通用部署，跨平台")
    print("  TensorRT:   NVIDIA GPU 部署，追求极致性能")
    print("  OpenVINO:   Intel CPU/集成显卡部署")
    print("  CoreML:     Apple 设备部署")
    print("  TFLite:     移动端/嵌入式部署")


if __name__ == "__main__":
    main()
