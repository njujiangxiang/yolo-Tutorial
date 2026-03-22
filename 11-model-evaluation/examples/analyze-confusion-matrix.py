# examples/analyze-confusion-matrix.py
"""
分析混淆矩阵

解读各类别的召回率、精确率，找出最容易混淆的类别对
"""
import numpy as np

def analyze_confusion_matrix(matrix, class_names):
    """
    分析混淆矩阵

    参数:
        matrix: 混淆矩阵 numpy 数组
        class_names: 类别名称列表
    """
    # 计算各类别的召回率
    recalls = np.diag(matrix) / matrix.sum(axis=1)

    # 计算各类别的精确率
    precisions = np.diag(matrix) / matrix.sum(axis=0)

    print("类别分析:")
    print("-" * 60)
    for i, name in enumerate(class_names):
        print(f"{name:10s}: Recall={recalls[i]:.2%}, Precision={precisions[i]:.2%}")

    # 找出最容易混淆的类别对
    np.fill_diagonal(matrix, 0)  # 忽略对角线
    max_confusion = np.unravel_index(np.argmax(matrix), matrix.shape)
    print(f"\n最容易混淆：{class_names[max_confusion[0]]} → {class_names[max_confusion[1]]}")
    print(f"混淆比例：{matrix[max_confusion]:.2%}")

if __name__ == "__main__":
    # 示例混淆矩阵
    matrix = np.array([
        [0.95, 0.03, 0.01, 0.01],
        [0.02, 0.90, 0.05, 0.03],
        [0.01, 0.08, 0.88, 0.03],
        [0.02, 0.05, 0.03, 0.90]
    ])

    classes = ['正常', '划痕', '孔洞', '短路']
    analyze_confusion_matrix(matrix, classes)
