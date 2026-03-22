# examples/03-grayscale.py
"""
灰度转换

学习将彩色图像转换为灰度图像
"""
import cv2

def main():
    # 读取彩色图像
    img = cv2.imread('test.jpg')

    if img is None:
        print("无法读取图像")
        return

    # 转换为灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 保存灰度图像
    cv2.imwrite('gray.jpg', gray)
    print("已保存灰度图像：gray.jpg")

    # 其他颜色空间转换
    # BGR 转 HSV（用于颜色分割）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # BGR 转 LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    print(f"\n原图形状：{img.shape}")
    print(f"灰度图形状：{gray.shape}")
    print(f"HSV 形状：{hsv.shape}")

if __name__ == "__main__":
    main()
