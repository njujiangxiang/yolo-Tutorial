# exercises/04-defect-detection.py
"""
缺陷检测应用示例

任务:
1. 模拟 PCB 缺陷检测场景
2. 批量处理产品图片
3. 统计缺陷率并生成报告
"""
import torch
import cv2
import numpy as np
from datetime import datetime


class DefectDetector:
    """缺陷检测器"""

    def __init__(self, model_name='yolov5s', conf_threshold=0.5):
        """
        初始化检测器

        参数:
            model_name: 模型名称
            conf_threshold: 置信度阈值
        """
        print(f"加载模型：{model_name}")
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.model.eval()
        self.conf_threshold = conf_threshold
        print("模型加载完成！")

    def detect(self, image):
        """
        检测单张图片

        参数:
            image: BGR 图像

        返回:
            检测结果列表
        """
        results = self.model(image)
        df = results.pandas().xyxy[0]

        # 过滤低置信度
        detections = df[df['confidence'] > self.conf_threshold]

        return detections

    def detect_file(self, image_path):
        """检测文件"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"错误：无法读取图片 {image_path}")
            return None

        detections = self.detect(image)

        # 保存结果
        results = self.model(image)
        results.show()

        return detections

    def batch_detect(self, image_paths):
        """
        批量检测

        参数:
            image_paths: 图片路径列表

        返回:
            检测结果列表
        """
        all_results = []

        for i, path in enumerate(image_paths):
            print(f"处理 [{i+1}/{len(image_paths)}]: {path}")
            image = cv2.imread(path)

            if image is not None:
                detections = self.detect(image)
                all_results.append({
                    'path': path,
                    'detections': detections,
                    'has_defect': len(detections) > 0
                })

        return all_results


def create_simulated_defect_images(output_dir, num_images=10):
    """
    创建模拟缺陷图片

    参数:
        output_dir: 输出目录
        num_images: 图片数量
    """
    import os

    print(f"创建 {num_images} 张模拟缺陷图片...")

    defect_count = 0

    for i in range(num_images):
        # 创建背景（模拟 PCB 板）
        img = np.ones((480, 640, 3), dtype=np.uint8) * 200

        # 添加一些电路图案
        for _ in range(20):
            x1, y1 = np.random.randint(0, 640), np.random.randint(0, 480)
            x2, y2 = x1 + np.random.randint(10, 100), y1 + np.random.randint(10, 50)
            cv2.rectangle(img, (x1, y1), (x2, y2), (180, 180, 180), 1)

        # 部分图片添加缺陷
        has_defect = np.random.random() > 0.3  # 70% 概率有缺陷
        if has_defect:
            num_defects = np.random.randint(1, 4)
            for _ in range(num_defects):
                x, y = np.random.randint(50, 590), np.random.randint(50, 430)
                w, h = np.random.randint(20, 60), np.random.randint(20, 60)
                # 缺陷用红色标记
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), -1)
            defect_count += 1

        # 保存图片
        filename = f"{output_dir}/pcb_{i:03d}.jpg"
        cv2.imwrite(filename, img)

    print(f"已保存 {num_images} 张图片，其中 {defect_count} 张有缺陷")
    return defect_count


def generate_report(results, output_path='defect_report.txt'):
    """
    生成缺陷报告

    参数:
        results: 检测结果
        output_path: 报告输出路径
    """
    total = len(results)
    defective = sum(1 for r in results if r['has_defect'])
    defect_rate = defective / total if total > 0 else 0

    # 统计缺陷类型
    defect_types = {}
    all_detections = []

    for r in results:
        detections = r['detections']
        if detections is not None and len(detections) > 0:
            all_detections.append(detections)
            for _, row in detections.iterrows():
                name = row['name']
                defect_types[name] = defect_types.get(name, 0) + 1

    # 生成报告
    report = []
    report.append("=" * 50)
    report.append("缺陷检测报告")
    report.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 50)
    report.append("")
    report.append("检测统计:")
    report.append(f"  总样品数：{total}")
    report.append(f"  缺陷品数：{defective}")
    report.append(f"  良品数：{total - defective}")
    report.append(f"  缺陷率：{defect_rate:.2%}")
    report.append("")

    if defect_types:
        report.append("缺陷类型分布:")
        for name, count in sorted(defect_types.items(), key=lambda x: x[1], reverse=True):
            report.append(f"  {name}: {count} 个")
        report.append("")

    report.append("详细结果:")
    for r in results:
        status = "DEFECT" if r['has_defect'] else "OK"
        count = len(r['detections']) if r['detections'] is not None else 0
        report.append(f"  {r['path']}: [{status}] - {count} 个缺陷")

    report.append("")
    report.append("=" * 50)

    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    # 打印报告
    print('\n'.join(report))
    print(f"\n报告已保存：{output_path}")


def main():
    import os

    print("=" * 50)
    print("缺陷检测应用示例")
    print("=" * 50)

    # 创建输出目录
    output_dir = 'pcb_samples'
    os.makedirs(output_dir, exist_ok=True)

    # 创建模拟图片
    num_images = 15
    create_simulated_defect_images(output_dir, num_images)

    # 获取所有图片路径
    image_paths = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith('.jpg')
    ][:10]  # 只处理前 10 张

    # 初始化检测器
    detector = DefectDetector(model_name='yolov5s', conf_threshold=0.3)

    print("\n" + "=" * 50)
    print("开始批量检测")
    print("=" * 50)

    # 批量检测
    results = detector.batch_detect(image_paths)

    # 生成报告
    print("\n" + "=" * 50)
    print("生成报告")
    print("=" * 50)
    generate_report(results)

    print("\n" + "=" * 50)
    print("使用说明:")
    print("=" * 50)
    print("1. 替换为自己的 PCB 图片进行真实检测")
    print("2. 调整 conf_threshold 参数改变检测灵敏度")
    print("3. 使用自定义训练的模型替换预训练模型")


if __name__ == "__main__":
    main()
