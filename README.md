# 🎯 YOLO 图像识别系统学习教程

> 从入门到精通的完整 YOLO 目标检测开发学习路径
> **企业培训版 - 2 周自学计划**

---

## 📚 课程目录

```
yolo-Tutorial/
├── guides/                # 学习指南 ⭐
│   ├── 2-week-plan.md     # 2 周学习计划
│   ├── data-preparation-guide.md  # 数据准备指南
│   └── troubleshooting.md # 常见问题
├── 00-setup/              # 环境配置
├── 01-cv-fundamentals/    # 计算机视觉基础
├── 02-deep-learning/      # 深度学习基础
├── 03-object-detection/   # 目标检测基础
├── 04-yolo-principles/    # YOLO 原理
├── 05-yolov5/             # YOLOv5 实战
├── 06-yolov8/             # YOLOv8 实战
├── 07-data-collection/    # 数据收集 ⭐新增
├── 08-data-processing/    # 数据处理 ⭐新增
├── 09-custom-dataset/     # 数据标注实战
├── 10-model-training/     # 模型训练
├── 11-model-evaluation/   # 模型评估 ⭐新增
├── 12-model-deployment/   # 模型部署
├── 13-advanced-topics/    # 高级主题
├── 14-projects/           # 实战项目
└── datasets/              # 数据集 ⭐新增
    ├── demo/              # 演示数据集
    └── defect/            # 缺陷检测数据集
```

---

## 🎯 企业培训学习路径（2 周 10 天）

### 第 1 周：基础与数据篇

| 天数 | 模块 | 主题 | 预计耗时 | 产出 |
|------|------|------|----------|------|
| Day 1 | 00-setup | 环境配置与快速上手 | 2 小时 | 跑通 YOLO 推理 |
| Day 2 | 01-04 | 深度学习与 YOLO 原理 | 3 小时 | 理解核心概念 |
| Day 3 | 07-data-collection | 数据收集方法 | 3 小时 | 收集 50+ 图片 |
| Day 4 | 08-data-processing | 数据处理与增强 | 4 小时 | 处理好的数据集 |
| Day 5 | 09-custom-dataset | 数据标注实战 | 3 小时 | 标注完整数据集 |

### 第 2 周：训练与部署篇

| 天数 | 模块 | 主题 | 预计耗时 | 产出 |
|------|------|------|----------|------|
| Day 6 | 10-model-training | 模型训练 | 4 小时 | 训练出模型 |
| Day 7 | 11-model-evaluation | 模型评估与诊断 | 3 小时 | 评估报告 |
| Day 8 | 05/06 | YOLOv5/v8 进阶 | 2 小时 | 对比实验 |
| Day 9 | 12-model-deployment | 模型导出与部署 | 3 小时 | 可部署模型 |
| Day 10 | 14-projects | 缺陷检测完整实战 | 4 小时 | 完整项目 |

---

## 🚀 快速开始

### 1. 先看学习指南

```bash
# 打开 2 周学习计划
open guides/2-week-plan.md

# 打开数据准备指南
open guides/data-preparation-guide.md

# 打开常见问题
open guides/troubleshooting.md
```

### 2. 环境要求

```bash
Python 3.10+
PyTorch 2.0+
GPU (推荐，用于训练)
```

### 3. 安装依赖

```bash
cd yolo-Tutorial
pip install -r requirements.txt
```

### 4. 开始学习

```bash
# 从 Day 1 开始
cd guides
# 阅读 2-week-plan.md

# 环境配置
cd ../00-setup
python examples/verify_install.py

# 按照每个模块的 README.md 学习
```

---

## 📋 课程大纲详情

### guides - 学习指南 ⭐

| 文件 | 说明 |
|------|------|
| 2-week-plan.md | 2 周 10 天详细学习计划 |
| data-preparation-guide.md | 数据准备完整流程 |
| troubleshooting.md | 常见问题与解决方法 |

### 00-setup - 环境配置

- Python 环境设置
- PyTorch 安装
- YOLO 框架安装
- GPU 配置
- 验证安装

### 01-cv-fundamentals - 计算机视觉基础

- 图像基础
- OpenCV 使用
- 图像预处理
- 特征提取

### 02-deep-learning - 深度学习基础

- 神经网络基础
- CNN 架构
- 损失函数
- 优化器

### 03-object-detection - 目标检测基础

- 检测任务概述
- 锚框概念
- 评估指标 (mAP, IoU)
- 主流检测算法

### 04-yolo-principles - YOLO 原理

- YOLO 发展历史
- YOLO 架构详解
- 损失函数设计
- 多尺度检测

### 05-yolov5 - YOLOv5 实战

- YOLOv5 架构
- 模型配置
- 推理使用
- 训练流程

### 06-yolov8 - YOLOv8 实战

- YOLOv8 新特性
- Ultralytics 框架
- 训练与验证
- 导出与部署

### 07-data-collection - 数据收集 ⭐新增

- 5 种数据收集方法
- 网络爬虫教程
- 公开数据集整理
- 自己拍摄技巧
- 实战练习

### 08-data-processing - 数据处理 ⭐新增

- 图像预处理
- 数据清洗（去模糊、去重）
- 数据划分（训练/验证/测试）
- 数据增强（基础 + 高级）
- YOLO 格式转换
- 数据质量检查清单

### 09-custom-dataset - 数据标注实战

- LabelImg 安装与使用
- 标注快捷键
- 标注质量标准
- YOLO 格式详解
- 创建 data.yaml

### 10-model-training - 模型训练

- 模型选择指南（n/s/m/l/x）
- 训练配置参数详解
- 训练过程可视化
- 训练异常诊断
- 迁移学习策略
- 缺陷检测训练技巧

### 11-model-evaluation - 模型评估 ⭐新增

- 评估指标详解（precision, recall, mAP）
- 混淆矩阵解读
- 训练曲线分析
- 坏案例分析方法
- 常见问题诊断表
- 生成评估报告

### 12-model-deployment - 模型部署

- 导出格式对比（ONNX/TensorRT/OpenVINO）
- ONNX Runtime 推理
- 图片/视频/摄像头推理
- FastAPI Web 服务部署
- 部署方案选择指南

### 13-advanced-topics - 高级主题

- 多模型融合
- 实时优化
- 小目标检测
- 遮挡处理

### 14-projects - 实战项目

- 项目 1：PCB 缺陷检测（完整流程）
- 项目 2：表面划痕检测
- 项目 3：产品装配检测

---

## 💡 学习建议

### 企业培训建议

1. **按计划学习**
   - 每天完成一个模块
   - 在群里打卡汇报进度
   - 遇到问题先在 troubleshooting.md 查找

2. **动手优先**
   - 看懂≠会做，一定要运行代码
   - 修改参数，观察结果变化
   - 保存每次实验的结果

3. **记录问题**
   - 准备笔记本或电子文档
   - 记录遇到的错误和解决方法
   - 在 Issues 中提问

4. **学以致用**
   - 学完 immediately 应用到公司项目
   - 选择一个具体的缺陷检测场景
   - 训练专用模型

### 自学建议

1. **循序渐进**: 按顺序学习每个主题
2. **动手实践**: 运行并修改示例代码
3. **完成练习**: 每个主题都有练习题
4. **记录笔记**: 在笔记文件夹记录心得
5. **构建项目**: 用 YOLO 解决实际问题
6. **参与社区**: 加入 YOLO 社区讨论

---

## 📊 学习进度追踪

### 第 1 周检查清单

- [ ] Day 1: 环境配置完成，能跑通推理
- [ ] Day 2: 理解 YOLO 基本原理
- [ ] Day 3: 收集了 50+ 张图片
- [ ] Day 4: 完成数据清洗和增强
- [ ] Day 5: 标注完成，有完整的 data.yaml

### 第 2 周检查清单

- [ ] Day 6: 训练出一个能用的模型
- [ ] Day 7: 能看懂评估指标
- [ ] Day 8: 完成 YOLOv5/v8 对比
- [ ] Day 9: 导出了 ONNX 模型
- [ ] Day 10: 完成了完整项目

---

## 🔗 相关资源

### 官方资源
- [YOLO 官方文档](https://docs.ultralytics.com/)
- [YOLO GitHub](https://github.com/ultralytics/ultralytics)
- [Papers With Code - YOLO](https://paperswithcode.com/method/yolo)

### 教程资源
- [Roboflow YOLO 教程](https://blog.roboflow.com/what-is-yolo/)
- [YOLO 目标检测入门](https://zhuanlan.zhihu.com/p/345078239)

### 数据集资源
- [COCO 数据集](https://cocodataset.org/)
- [Roboflow Universe](https://universe.roboflow.com/)
- [Kaggle 数据集](https://www.kaggle.com/datasets)

---

## 📞 支持与反馈

学习过程中遇到问题：

1. 查看 `guides/troubleshooting.md` 常见问题
2. 在 GitHub Issues 提问
3. 联系培训课程负责人

---

## 📝 版本说明

本教程为企业培训定制版本，特点：
- ✅ 所有文档为中文
- ✅ 包含 2 周自学计划
- ✅ 新增数据收集章节
- ✅ 新增数据处理章节
- ✅ 新增模型评估章节
- ✅ 包含缺陷检测实战项目
- ✅ 提供配套数据集

---

**祝你学习愉快！🚀**

如有问题，欢迎在 Issues 中提问。
