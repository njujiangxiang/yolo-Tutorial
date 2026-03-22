# examples/02-image-basic.py
"""
图像基本操作

学习裁剪、调整大小、旋转图像
"""
import cv2

def main():
    # 读取图像
    img = cv2.imread('test.jpg')

    if img is None:
        print("无法读取图像")
        return

    # 1. 裁剪图像
    # 格式：img[y1:y2, x1:x2]
    cropped = img[100:300, 200:400]
    cv2.imwrite('cropped.jpg', cropped)
    print("已保存裁剪图像：cropped.jpg")

    # 2. 调整大小
    # 方法 1：指定目标尺寸
    resized = cv2.resize(img, (320, 240))
    cv2.imwrite('resized.jpg', resized)
    print("已保存调整大小图像：resized.jpg")

    # 方法 2：按比例缩放
    scale = 0.5
    scaled = cv2.resize(img, None, fx=scale, fy=scale)
    cv2.imwrite('scaled.jpg', scaled)
    print(f"已保存缩放图像：scaled.jpg (原图的 {scale*100}%)")

    # 3. 旋转图像
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    # 旋转 45 度
    M = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    cv2.imwrite('rotated.jpg', rotated)
    print("已保存旋转图像：rotated.jpg (旋转 45 度)")

    # 4. 翻转图像
    # 水平翻转
    flipped_h = cv2.flip(img, 1)
    cv2.imwrite('flipped_h.jpg', flipped_h)
    print("已保存水平翻转图像：flipped_h.jpg")

    # 垂直翻转
    flipped_v = cv2.flip(img, 0)
    cv2.imwrite('flipped_v.jpg', flipped_v)
    print("已保存垂直翻转图像：flipped_v.jpg")

if __name__ == "__main__":
    main()
