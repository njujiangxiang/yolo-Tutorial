# examples/03-remove-blur.py
"""
删除模糊图片

使用拉普拉斯方差计算模糊度
"""
import cv2
import os
from pathlib import Path

def calculate_blur_score(image_path):
    """
    计算图片模糊度（拉普拉斯方差）

    返回值越小，图片越模糊
    """
    img = cv2.imread(image_path)
    if img is None:
        return 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance

def remove_blur_images(input_dir, output_dir, threshold=100):
    """
    删除模糊图片

    参数:
        threshold: 模糊度阈值，低于此值的图片被删除
    """
    os.makedirs(output_dir, exist_ok=True)

    kept = 0
    removed = 0

    for img_path in Path(input_dir).glob("*.jpg"):
        score = calculate_blur_score(str(img_path))

        if score >= threshold:
            # 清晰图片，保留
            os.system(f"cp {img_path} {output_dir}/")
            kept += 1
        else:
            # 模糊图片，删除
            print(f"删除模糊图片：{img_path} (分数：{score:.2f})")
            removed += 1

    print(f"处理完成：保留 {kept} 张，删除 {removed} 张")

if __name__ == "__main__":
    remove_blur_images("datasets/raw/", "datasets/cleaned/", threshold=100)
