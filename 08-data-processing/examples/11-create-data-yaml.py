# examples/11-create-data-yaml.py
"""
创建 YOLO data.yaml 配置文件
"""
import os
import yaml
from pathlib import Path

def create_data_yaml(dataset_dir, class_names, output_path):
    """
    创建 YOLO data.yaml 配置文件

    参数:
        dataset_dir: 数据集根目录
        class_names: 类别名称列表
        output_path: 输出路径
    """
    config = {
        'path': os.path.abspath(dataset_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(class_names),
        'names': class_names
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    print(f"data.yaml 已创建：{output_path}")
    print("\n内容预览:")
    print(yaml.dump(config, allow_unicode=True))

if __name__ == "__main__":
    # PCB 缺陷检测示例
    classes = ['hole', 'scratch', 'short', 'open', 'spurious']

    create_data_yaml(
        dataset_dir="datasets/split/",
        class_names=classes,
        output_path="datasets/split/data.yaml"
    )
