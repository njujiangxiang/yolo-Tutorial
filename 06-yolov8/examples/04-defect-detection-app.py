# exercises/04-defect-detection-app.py
"""
缺陷检测应用

任务:
1. 加载自定义模型
2. 批量处理产品图片
3. 统计缺陷率
4. 生成报告
"""
from ultralytics import YOLO
import cv2
import numpy as np
import os
from datetime import datetime


class DefectDetectionApp:
    """缺陷检测应用"""

    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.5):
        """
        初始化应用

        参数:
            model_path: 模型路径
            conf_threshold: 置信度阈值
        """
        print(f"加载模型：{model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        print("模型加载完成！")

    def detect(self, image):
        """
        检测单张图片

        参数:
            image: BGR 图像或图片路径

        返回:
            检测结果
        """
        results = self.model(image, verbose=False)
        return results[0]

    def has_defect(self, result, min_conf=0.5):
        """
        判断是否有缺陷

        参数:
            result: 检测结果
            min_conf: 最小置信度

        返回:
            bool
        """
        if result.boxes is None:
            return False

        for box in result.boxes:
            if float(box.conf[0]) >= min_conf:
                return True
        return False

    def batch_detect(self, image_dir, output_dir='detection_results'):
        """
        批量检测

        参数:
            image_dir: 图片目录
            output_dir: 输出目录

        返回:
            检测结果统计
        """
        os.makedirs(output_dir, exist_ok=True)

        # 获取所有图片
        image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        total = len(image_files)
        defective = 0
        all_results = []

        print(f"\n找到 {total} 张图片，开始检测...")
        print("=" * 50)

        for i, filename in enumerate(image_files):
            image_path = os.path.join(image_dir, filename)

            # 检测
            result = self.detect(image_path)

            # 判断是否有缺陷
            has_def = self.has_defect(result, self.conf_threshold)
            if has_def:
                defective += 1

            # 保存结果图片
            result.save(os.path.join(output_dir, filename))

            # 记录结果
            all_results.append({
                'filename': filename,
                'has_defect': has_def,
                'num_detections': len(result.boxes) if result.boxes else 0
            })

            # 显示进度
            if (i + 1) % 10 == 0 or (i + 1) == total:
                progress = (i + 1) / total * 100
                print(f"进度：{i + 1}/{total} ({progress:.1f}%), "
                      f"已发现缺陷：{defective}")

        # 计算统计
        stats = {
            'total': total,
            'defective': defective,
            'good': total - defective,
            'defect_rate': defective / total if total > 0 else 0,
            'results': all_results
        }

        return stats

    def generate_report(self, stats, output_path='defect_report.txt'):
        """
        生成缺陷报告

        参数:
            stats: 统计信息
            output_path: 报告路径
        """
        report = []
        report.append("=" * 50)
        report.append("缺陷检测报告")
        report.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"置信度阈值：{self.conf_threshold}")
        report.append("=" * 50)
        report.append("")
        report.append("检测统计:")
        report.append(f"  总样品数：{stats['total']}")
        report.append(f"  良品数：{stats['good']}")
        report.append(f"  缺陷品数：{stats['defective']}")
        report.append(f"  缺陷率：{stats['defect_rate']:.2%}")
        report.append("")
        report.append("详细结果:")
        report.append("-" * 50)

        for r in stats['results']:
            status = "DEFECT" if r['has_defect'] else "OK"
            report.append(f"  {r['filename']}: [{status}] - "
                         f"{r['num_detections']} 个缺陷")

        report.append("")
        report.append("=" * 50)

        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        # 打印报告
        print('\n'.join(report))
        print(f"\n报告已保存：{output_path}")


def create_sample_images(output_dir, num_images=20):
    """
    创建模拟产品图片

    参数:
        output_dir: 输出目录
        num_images: 图片数量
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"创建 {num_images} 张模拟产品图片...")

    defect_count = 0

    for i in range(num_images):
        # 创建背景（模拟产品表面）
        img = np.ones((480, 640, 3), dtype=np.uint8) * 240

        # 添加一些纹理
        for _ in range(50):
            x, y = np.random.randint(0, 640), np.random.randint(0, 480)
            img[y, x] = np.random.randint(220, 255, 3)

        # 部分图片添加缺陷
        has_defect = np.random.random() > 0.3  # 70% 良品率
        if has_defect:
            num_defects = np.random.randint(1, 4)
            for _ in range(num_defects):
                x, y = np.random.randint(50, 590), np.random.randint(50, 430)
                w, h = np.random.randint(15, 50), np.random.randint(15, 50)
                # 缺陷用暗色标记
                color = np.random.randint(0, 100, 3)
                cv2.rectangle(img, (x, y), (x + w, y + h),
                             color.tolist(), -1)
            defect_count += 1

        # 保存图片
        filename = f"{output_dir}/product_{i:03d}.jpg"
        cv2.imwrite(filename, img)

    print(f"已保存 {num_images} 张图片")
    print(f"其中 {defect_count} 张有缺陷（模拟）")


def main():
    print("=" * 50)
    print("缺陷检测应用")
    print("=" * 50)

    # 创建示例图片
    image_dir = 'product_images'
    create_sample_images(image_dir, num_images=20)

    # 初始化检测应用
    app = DefectDetectionApp(
        model_path='yolov8n.pt',  # 使用预训练模型演示
        conf_threshold=0.3
    )

    # 批量检测
    output_dir = 'detection_results'
    stats = app.batch_detect(image_dir, output_dir)

    # 生成报告
    print("\n" + "=" * 50)
    print("生成报告")
    print("=" * 50)
    app.generate_report(stats)

    print("\n" + "=" * 50)
    print("使用说明:")
    print("=" * 50)
    print("1. 替换模型路径使用自定义训练的模型")
    print("2. 调整 conf_threshold 改变检测灵敏度")
    print("3. 结果图片保存在 detection_results 目录")
    print("4. 缺陷报告保存在 defect_report.txt")


if __name__ == "__main__":
    main()
