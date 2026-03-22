# examples/05-data-quality-check.py
"""
生成数据质量报告

检查项目:
- 图片数量
- 图片尺寸分布
- 模糊度分布
- 亮度分布
"""
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def generate_quality_report(input_dir, report_path="quality_report.txt"):
    """
    生成数据质量报告
    """
    results = {
        'count': 0,
        'sizes': [],
        'blur_scores': [],
        'brightness': []
    }

    for img_path in Path(input_dir).glob("*.jpg"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        results['count'] += 1
        results['sizes'].append(img.shape[:2])

        # 模糊度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results['blur_scores'].append(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 亮度
        results['brightness'].append(np.mean(gray))

    # 生成报告
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("数据质量报告\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"图片总数：{results['count']}\n")

        if results['sizes']:
            sizes = np.array(results['sizes'])
            f.write(f"平均尺寸：{sizes.mean(axis=0).astype(int)}\n")
            f.write(f"最小尺寸：{sizes.min(axis=0)}\n")
            f.write(f"最大尺寸：{sizes.max(axis=0)}\n\n")

        if results['blur_scores']:
            blur = np.array(results['blur_scores'])
            f.write(f"平均模糊度：{blur.mean():.2f}\n")
            f.write(f"最小模糊度：{blur.min():.2f}\n")
            f.write(f"最大模糊度：{blur.max():.2f}\n")
            f.write(f"模糊图片数（<100）: {(blur < 100).sum()}\n\n")

        if results['brightness']:
            bright = np.array(results['brightness'])
            f.write(f"平均亮度：{bright.mean():.2f}\n")
            f.write(f"最暗：{bright.min():.2f}\n")
            f.write(f"最亮：{bright.max():.2f}\n")

    print(f"报告已生成：{report_path}")
    return results

if __name__ == "__main__":
    generate_quality_report("datasets/deduped/")
