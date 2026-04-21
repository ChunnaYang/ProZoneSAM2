# 环境依赖安装说明

## 问题
系统显示没有torch模块。

## 解决方案
已成功安装所有必要的Python依赖包。

## 已安装的依赖

### 核心依赖
- **PyTorch**: 2.10.0+cpu (CPU版本)
- **torchvision**: 0.25.0+cpu
- **NumPy**: 2.4.2
- **Nibabel**: 5.3.3 (用于读取nii.gz文件)
- **SciPy**: 1.17.1 (用于形态学处理)
- **Pillow**: 10.4.0 (用于图像处理)
- **MONAI**: 1.5.2 (医学影像处理框架)
- **Hydra**: 1.3.2 (配置管理)
- **OmegaConf**: 2.3.0 (配置管理)

## 安装命令

### 方法1：一键安装（推荐）
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install monai hydra-core omegaconf
```

### 方法2：使用requirements.txt
```bash
pip install -r requirements.txt
```

### requirements.txt内容
```
torch==2.10.0+cpu
torchvision==0.25.0+cpu
nibabel>=5.0.0
scipy>=1.10.0
numpy>=1.24.0
pillow>=10.0.0
monai>=1.3.0
hydra-core>=1.3.0
omegaconf>=2.3.0
```

## 验证安装

运行以下命令验证所有依赖已正确安装：

```bash
/usr/bin/python3 << 'EOF'
import torch
import nibabel
import scipy
from PIL import Image
import monai

print("✓ 所有依赖已成功安装")
print(f"  PyTorch: {torch.__version__}")
print(f"  Nibabel: {nibabel.__version__}")
print(f"  SciPy: {scipy.__version__}")
print(f"  MONAI: {monai.__version__}")
EOF
```

## 注意事项

1. **CPU版本**: 当前安装的是PyTorch CPU版本，适合开发和测试
2. **GPU版本**: 如需GPU加速，需要安装CUDA版本的PyTorch：
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
3. **Python版本**: 当前使用Python 3.12.3
4. **包管理器**: 使用pip进行包管理

## 故障排除

### 问题：ModuleNotFoundError: No module named 'torch'
**解决**：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 问题：nibabel未安装
**解决**：
```bash
pip install nibabel
```

### 问题：scipy未安装
**解决**：
```bash
pip install scipy
```

### 问题：MONAI未安装
**解决**：
```bash
pip install monai
```

## 更新时间
2025-02-26
