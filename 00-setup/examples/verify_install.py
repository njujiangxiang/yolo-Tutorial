"""
YOLO 环境验证脚本
"""

import sys
import subprocess

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def check_package(name, import_name=None):
    """检查包是否安装"""
    if import_name is None:
        import_name = name
    
    try:
        __import__(import_name)
        print(f"{Colors.GREEN}✓{Colors.END} {name}")
        return True
    except ImportError:
        print(f"{Colors.RED}✗{Colors.END} {name}")
        return False

def check_gpu():
    """检查 GPU 可用性"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"{Colors.GREEN}✓{Colors.END} CUDA: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print(f"{Colors.YELLOW}⚠{Colors.END} CUDA 不可用 (使用 CPU)")
            return True
    except:
        print(f"{Colors.RED}✗{Colors.END} PyTorch 未安装")
        return False

def check_yolo():
    """检查 YOLO 安装"""
    try:
        from ultralytics import YOLO
        print(f"{Colors.GREEN}✓{Colors.END} Ultralytics YOLO")
        
        # 下载并验证模型
        model = YOLO('yolov8n.pt')
        print(f"  模型加载成功")
        return True
    except Exception as e:
        print(f"{Colors.RED}✗{Colors.END} YOLO: {str(e)}")
        return False

def main():
    print(f"\n{Colors.BLUE}🎯 YOLO 环境验证{Colors.END}\n")
    
    results = []
    
    print("Python 版本:")
    print(f"  {sys.version.split()[0]}")
    
    print("\n核心依赖:")
    results.append(check_package('torch'))
    results.append(check_package('torchvision'))
    results.append(check_package('numpy'))
    results.append(check_package('opencv-python', 'cv2'))
    results.append(check_package('ultralytics'))
    results.append(check_package('PIL', 'PIL'))
    
    print("\n硬件加速:")
    results.append(check_gpu())
    
    print("\nYOLO 框架:")
    results.append(check_yolo())
    
    print(f"\n{'='*40}")
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"通过：{passed}/{total}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}🎉 环境配置完成！{Colors.END}\n")
    else:
        print(f"\n{Colors.YELLOW}⚠️  部分检查未通过{Colors.END}\n")

if __name__ == "__main__":
    main()
