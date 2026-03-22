# examples/06-threshold.py
"""
阈值处理

学习使用阈值将图像二值化
"""
import cv2

def main():
    # 读取图像（灰度）
    gray = cv2.imread('test.jpg', 0)

    if gray is None:
        print("无法读取图像")
        return

    # 1. 简单阈值
    # 超过阈值为白色，否则为黑色
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite('threshold_binary.jpg', binary)
    print("已保存二值化图像：threshold_binary.jpg")

    # 2. 反向阈值
    _, binary_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite('threshold_binary_inv.jpg', binary_inv)
    print("已保存反向二值化图像：threshold_binary_inv.jpg")

    # 3. 自适应阈值
    # 根据局部区域自动确定阈值
    adaptive_mean = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    cv2.imwrite('threshold_adaptive_mean.jpg', adaptive_mean)
    print("已保存自适应阈值图像（均值）：threshold_adaptive_mean.jpg")

    adaptive_gaussian = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    cv2.imwrite('threshold_adaptive_gaussian.jpg', adaptive_gaussian)
    print("已保存自适应阈值图像（高斯）：threshold_adaptive_gaussian.jpg")

    # 4. Otsu 阈值（自动确定最佳阈值）
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite('threshold_otsu.jpg', otsu)
    print("已保存 Otsu 阈值图像：threshold_otsu.jpg")

    print("\n提示:")
    print("- 简单阈值适用于光照均匀的图像")
    print("- 自适应阈值适用于光照不均匀的图像")
    print("- Otsu 方法自动找到最佳阈值")

if __name__ == "__main__":
    main()
