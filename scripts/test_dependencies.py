#!/usr/bin/env python3
"""
测试Python环境和依赖安装
"""

import sys

print("=" * 60)
print("Python环境测试")
print("=" * 60)
print(f"Python版本: {sys.version}")
print(f"Python路径: {sys.executable}")
print()

# 测试依赖包
dependencies = [
    ("torch", lambda: __import__('torch').__version__),
    ("torchvision", lambda: __import__('torchvision').__version__),
    ("numpy", lambda: __import__('numpy').__version__),
    ("nibabel", lambda: __import__('nibabel').__version__),
    ("scipy", lambda: __import__('scipy').__version__),
    ("PIL", lambda: __import__('PIL').__version__),
    ("monai", lambda: __import__('monai').__version__),
    ("hydra", lambda: __import__('hydra').__version__),
]

print("依赖包检查:")
print("-" * 60)

all_installed = True
for name, get_version in dependencies:
    try:
        version = get_version()
        print(f"✓ {name:20s} {version}")
    except ImportError as e:
        print(f"✗ {name:20s} 未安装")
        all_installed = False

print("-" * 60)

# 测试PyTorch CPU/GPU
try:
    import torch
    print(f"\nPyTorch设备检查:")
    print(f"  可用设备: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  GPU数量: {torch.cuda.device_count()}")
except Exception as e:
    print(f"\nPyTorch检查失败: {e}")

print("\n" + "=" * 60)
if all_installed:
    print("✅ 所有依赖已成功安装，系统可以正常运行！")
else:
    print("❌ 部分依赖未安装，请运行安装命令")
print("=" * 60)
