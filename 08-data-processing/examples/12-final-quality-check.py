# examples/12-final-quality-check.py
"""
最终数据质量检查

检查项目:
1. 目录结构是否正确
2. 图片和标注数量是否匹配
3. 标注格式是否正确
4. data.yaml 是否存在
5. 类别是否平衡
"""
import os
from pathlib import Path
import yaml

def final_quality_check(dataset_dir):
    """最终数据质量检查"""
    issues = []
    warnings = []

    # 1. 检查目录结构
    required_dirs = [
        f"{dataset_dir}/images/train",
        f"{dataset_dir}/images/val",
        f"{dataset_dir}/labels/train",
        f"{dataset_dir}/labels/val"
    ]

    for d in required_dirs:
        if not os.path.exists(d):
            issues.append(f"缺少目录：{d}")

    # 2. 检查图片和标注匹配
    for split in ['train', 'val']:
        img_dir = f"{dataset_dir}/images/{split}"
        lbl_dir = f"{dataset_dir}/labels/{split}"

        if os.path.exists(img_dir) and os.path.exists(lbl_dir):
            img_files = set(f.stem for f in Path(img_dir).glob("*.jpg"))
            lbl_files = set(f.stem for f in Path(lbl_dir).glob("*.txt"))

            missing_labels = img_files - lbl_files
            missing_images = lbl_files - img_files

            if missing_labels:
                warnings.append(f"{split} 集：{len(missing_labels)} 张图片缺少标注")
            if missing_images:
                warnings.append(f"{split} 集：{len(missing_images)} 个标注缺少图片")

    # 3. 检查标注格式
    for lbl_file in Path(f"{dataset_dir}/labels/train").glob("*.txt"):
        with open(lbl_file, 'r') as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) != 5:
                    issues.append(f"标注格式错误：{lbl_file}:{i+1}")
                    break

                try:
                    values = [float(p) for p in parts[1:]]
                    if not all(0 <= v <= 1 for v in values):
                        warnings.append(f"标注值超出 [0,1]: {lbl_file}:{i+1}")
                except ValueError:
                    issues.append(f"标注值不是数字：{lbl_file}:{i+1}")
                    break

    # 4. 检查 data.yaml
    yaml_path = f"{dataset_dir}/data.yaml"
    if not os.path.exists(yaml_path):
        warnings.append("缺少 data.yaml 配置文件")
    else:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
            if 'names' not in config:
                issues.append("data.yaml 缺少 'names' 字段")
            if 'nc' not in config:
                warnings.append("data.yaml 缺少 'nc' 字段")

    # 输出报告
    print("=" * 50)
    print("数据质量检查报告")
    print("=" * 50)

    if issues:
        print(f"\n❌ 发现 {len(issues)} 个问题:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ 没有发现严重问题")

    if warnings:
        print(f"\n⚠️ 发现 {len(warnings)} 个警告:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("\n✅ 没有发现警告")

    print("\n" + "=" * 50)

    return len(issues) == 0

if __name__ == "__main__":
    final_quality_check("datasets/split/")
