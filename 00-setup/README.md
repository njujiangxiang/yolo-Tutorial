# 00-Setup - 环境配置

> 配置 YOLO 开发环境是学习的第一步

---

## 🎯 学习目标

- ✅ 安装 Python 3.10+
- ✅ 配置 PyTorch
- ✅ 安装 YOLO 框架
- ✅ GPU 配置 (可选)
- ✅ 验证安装

---

## 📦 环境要求

### 最低配置
- Python 3.10+
- 8GB RAM
- 4GB 可用存储

### 推荐配置
- Python 3.10+
- 16GB+ RAM
- NVIDIA GPU (8GB+ VRAM)
- 20GB+ 可用存储

---

## 🚀 快速开始

### 1. 创建虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
```

### 2. 安装 PyTorch

```bash
# CPU 版本
pip install torch torchvision torchaudio

# GPU 版本 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. 安装 YOLO

```bash
# Ultralytics YOLOv8
pip install ultralytics

# YOLOv5
pip install git+https://github.com/ultralytics/yolov5.git
```

### 4. 验证安装

```bash
python examples/verify_install.py
```

---

## 🔧 常见问题

### GPU 不可用
```python
import torch
print(torch.cuda.is_available())  # 应该返回 True
```

### 依赖冲突
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

---

## 📝 检查清单

- [ ] Python 3.10+ 已安装
- [ ] 虚拟环境已激活
- [ ] PyTorch 已安装
- [ ] YOLO 已安装
- [ ] GPU 可用 (如有)
- [ ] 验证脚本运行成功

---

**配置完成后开始学习！** 🚀
