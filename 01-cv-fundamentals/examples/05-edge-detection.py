# examples/05-edge-detection.py
"""
边缘检测

学习使用 Canny 和 Sobel 算子检测图像边缘
"""
import cv2
import numpy as np

def main():
    # 读取图像
    img = cv2.imread('test.jpg')

    if img is None:
        print("无法读取图像")
        return

    # 转换为灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Canny 边缘检测
    # 最常用的边缘检测方法
    edges_canny = cv2.Canny(gray, 100, 200)
    cv2.imwrite('edges_canny.jpg', edges_canny)
    print("已保存 Canny 边缘图像：edges_canny.jpg")

    # 2. Sobel 边缘检测
    # 分别检测水平和垂直方向的边缘
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # 合并两个方向的边缘
    sobel_combined = cv2.addWeighted(
        cv2.convertScaleAbs(sobel_x), 0.5,
        cv2.convertScaleAbs(sobel_y), 0.5,
        0
    )
    cv2.imwrite('edges_sobel.jpg', sobel_combined)
    print("已保存 Sobel 边缘图像：edges_sobel.jpg")

    # 3. Laplacian 边缘检测
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    cv2.imwrite('edges_laplacian.jpg', laplacian)
    print("已保存 Laplacian 边缘图像：edges_laplacian.jpg")

    print("\n提示:")
    print("- Canny: 效果好，最常用")
    print("- Sobel: 可以检测特定方向的边缘")
    print("- Laplacian: 对噪声敏感，通常先模糊再用")

if __name__ == "__main__":
    main()
