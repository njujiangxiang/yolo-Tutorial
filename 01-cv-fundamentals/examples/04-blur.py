# examples/04-blur.py
"""
图像平滑（去噪）

学习使用不同的滤波器去除图像噪声
"""
import cv2

def main():
    # 读取图像
    img = cv2.imread('test.jpg')

    if img is None:
        print("无法读取图像")
        return

    # 1. 高斯模糊
    # 最常用的去噪方法，效果好
    gaussian = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite('blur_gaussian.jpg', gaussian)
    print("已保存高斯模糊图像：blur_gaussian.jpg")

    # 2. 中值滤波
    # 对椒盐噪声特别有效
    median = cv2.medianBlur(img, 5)
    cv2.imwrite('blur_median.jpg', median)
    print("已保存中值滤波图像：blur_median.jpg")

    # 3. 均值滤波
    # 简单快速，但效果一般
    box = cv2.blur(img, (5, 5))
    cv2.imwrite('blur_box.jpg', box)
    print("已保存均值滤波图像：blur_box.jpg")

    # 4. 双边滤波
    # 去噪同时保留边缘
    bilateral = cv2.bilateralFilter(img, 9, 75, 75)
    cv2.imwrite('blur_bilateral.jpg', bilateral)
    print("已保存双边滤波图像：blur_bilateral.jpg")

    print("\n提示：参数说明")
    print("- 高斯模糊：(kernel_size, kernel_size), sigma")
    print("- 中值滤波：kernel_size（必须是奇数）")
    print("- 双边滤波：d(邻域直径), sigmaColor, sigmaSpace")

if __name__ == "__main__":
    main()
