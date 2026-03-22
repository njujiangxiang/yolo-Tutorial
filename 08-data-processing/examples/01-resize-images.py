# examples/01-resize-images.py
"""
调整图片尺寸，保持宽高比
"""
import cv2
import os
from pathlib import Path

def resize_with_aspect_ratio(image_path, output_path, target_size=640):
    """
    调整图片尺寸，保持宽高比

    参数:
        image_path: 输入图片路径
        output_path: 输出图片路径
        target_size: 目标尺寸（长边）
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片：{image_path}")
        return

    height, width = img.shape[:2]

    # 计算缩放比例
    scale = target_size / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)

    # 调整尺寸
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # 保存（添加灰边填充到正方形）
    square_img = cv2.copyMakeBorder(
        resized,
        (target_size - new_height) // 2,
        target_size - new_height - (target_size - new_height) // 2,
        (target_size - new_width) // 2,
        target_size - new_width - (target_size - new_width) // 2,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)  # YOLO 默认灰色
    )

    cv2.imwrite(output_path, square_img)
    print(f"处理完成：{output_path}")

def batch_resize(input_dir, output_dir, target_size=640):
    """批量处理图片"""
    os.makedirs(output_dir, exist_ok=True)

    image_files = list(Path(input_dir).glob("*.jpg")) + \
                  list(Path(input_dir).glob("*.png"))

    for img_path in image_files:
        resize_with_aspect_ratio(
            str(img_path),
            f"{output_dir}/{img_path.stem}.jpg",
            target_size
        )

if __name__ == "__main__":
    batch_resize("datasets/raw/", "datasets/resized/", target_size=640)
