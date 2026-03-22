# examples/04-data-synthesis.py
"""
数据合成脚本

功能：合成缺陷图片，将缺陷区域抠图粘贴到不同背景上
"""
import cv2
import numpy as np
import os

def synthesize_defect_images(defect_dir, background_dir, output_dir):
    """
    合成缺陷图片

    原理：将缺陷区域抠图，粘贴到不同背景上
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"合成图片保存到：{output_dir}")

if __name__ == "__main__":
    synthesize_defect_images(
        "datasets/defects/",
        "datasets/backgrounds/",
        "datasets/synthesized/"
    )
