# examples/plot-training-results.py
"""
绘制训练结果曲线

包含:
- Loss 曲线
- mAP 曲线
- Precision/Recall 曲线
"""
import matplotlib.pyplot as plt
import pandas as pd

def plot_results(results_csv_path):
    """
    绘制训练结果曲线
    """
    # 读取结果
    df = pd.read_csv(results_csv_path)

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Loss 曲线
    axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='box_loss')
    axes[0, 0].plot(df['epoch'], df['train/cls_loss'], label='cls_loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].set_title('Training Loss')

    # 2. mAP 曲线
    axes[0, 1].plot(df['epoch'], df['metrics/precision'], label='precision')
    axes[0, 1].plot(df['epoch'], df['metrics/recall'], label='recall')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].legend()
    axes[0, 1].set_title('Metrics')

    # 3. mAP50
    axes[1, 0].plot(df['epoch'], df['metrics/mAP50'], label='mAP50', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('mAP50')
    axes[1, 0].legend()
    axes[1, 0].set_title('mAP50')

    # 4. mAP50-95
    axes[1, 1].plot(df['epoch'], df['metrics/mAP50-95'], label='mAP50-95', color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('mAP50-95')
    axes[1, 1].legend()
    axes[1, 1].set_title('mAP50-95')

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    plot_results('runs/detect/defect-train/results.csv')
