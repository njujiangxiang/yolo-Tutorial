# examples/export-onnx.py
"""
导出模型为 ONNX 格式

ONNX 是通用的模型格式，支持跨平台部署
"""
from ultralytics import YOLO

def export_model():
    """导出模型为 ONNX 格式"""
    # 加载模型
    model = YOLO('runs/detect/defect-train/weights/best.pt')

    # 导出
    onnx_path = model.export(
        format='onnx',
        simplify=True,        # 简化模型（推荐）
        dynamic_axes=False,   # 固定输入尺寸
        opset=12,            # ONNX 算子版本
    )

    print(f"模型已导出：{onnx_path}")
    return onnx_path

if __name__ == "__main__":
    export_model()
