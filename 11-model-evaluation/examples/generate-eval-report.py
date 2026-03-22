# examples/generate-eval-report.py
"""
生成完整的评估报告

包含:
- 总体指标（precision, recall, mAP）
- 各类别指标
- 混淆矩阵
"""
from ultralytics import YOLO
import yaml
import json
from datetime import datetime

def generate_evaluation_report(model_path, data_yaml, output_path):
    """
    生成完整的评估报告
    """
    # 加载模型和数据
    model = YOLO(model_path)
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)

    # 运行验证
    metrics = model.val(data=data_yaml, plots=True, save_json=True)

    # 生成报告
    report = {
        'model': model_path,
        'dataset': data_yaml,
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
        },
        'per_class': []
    }

    # 各类别指标
    for i, name in enumerate(data['names']):
        report['per_class'].append({
            'class_id': i,
            'class_name': name,
            'precision': float(metrics.box.mp[i]) if hasattr(metrics.box.mp, '__iter__') else float(metrics.box.mp),
            'recall': float(metrics.box.mr[i]) if hasattr(metrics.box.mr, '__iter__') else float(metrics.box.mr),
            'mAP50': float(metrics.box.map50[i]) if hasattr(metrics.box.map50, '__iter__') else float(metrics.box.map50),
        })

    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"评估报告已保存：{output_path}")

    # 打印摘要
    print("\n" + "=" * 50)
    print("评估摘要")
    print("=" * 50)
    print(f"Precision:  {report['metrics']['precision']:.2%}")
    print(f"Recall:     {report['metrics']['recall']:.2%}")
    print(f"mAP50:      {report['metrics']['mAP50']:.2%}")
    print(f"mAP50-95:   {report['metrics']['mAP50-95']:.2%}")
    print("=" * 50)

    return report

if __name__ == "__main__":
    generate_evaluation_report(
        model_path='runs/detect/defect-train/weights/best.pt',
        data_yaml='datasets/labeled/data.yaml',
        output_path='evaluation_report.json'
    )
