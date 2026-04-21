# Medical SAM2 模型配置指南

本指南将帮助你配置和使用Medical SAM2模型进行医学图像分割。

## 前置条件

### 已安装的Python包
- ✅ PyTorch 2.9.1+cpu
- ✅ torchvision 0.24.1+cpu
- ✅ Pillow 10.4.0
- ✅ numpy 2.4.1

## 方案一：使用官方SAM2模型

### 1. 安装segment-anything-2包

由于网络原因，我们提供两种安装方式：

#### 方式A：使用pip安装（推荐）
```bash
pip3 install segment-anything-2
```

#### 方式B：从源码安装
```bash
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip3 install -e .
cd ..
```

### 2. 下载模型权重文件

根据你的需求选择合适的模型：

#### SAM2 Base 模型（推荐用于测试）
```bash
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt
```

#### SAM2 Large 模型（更高精度）
```bash
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

#### SAM2 Small 模型（更快速度）
```bash
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt
```

### 3. 配置模型路径

编辑 `scripts/segment.py` 文件：

```python
# 在文件顶部设置模型路径
MODEL_CHECKPOINT = "sam2_hiera_base_plus.pt"  # 改为你的模型路径
MODEL_CONFIG = "sam2_hiera_b+.yaml"  # 对应的配置文件
```

## 方案二：使用Medical SAM2

Medical SAM2 是专门为医学图像优化的SAM2变体。

### 1. 安装Medical SAM2

```bash
git clone https://github.com/bowang-lab/Medical-SAM2.git
cd Medical-SAM2
pip3 install -r requirements.txt
```

### 2. 下载Medical SAM2模型

```bash
# 从Medical SAM2的GitHub页面下载对应的模型文件
# 例如：medsam2_vit_b.pth
wget https://github.com/bowang-lab/Medical-SAM2/releases/download/v1.0/medsam2_vit_b.pth
```

### 3. 修改Python脚本

编辑 `scripts/segment.py`，使用Medical SAM2的加载方式：

```python
from medsam2 import MedSAM2Predictor

def load_model(checkpoint_path: str = None):
    """Load Medical SAM2 model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = MedSAM2Predictor(checkpoint_path, device=device)
    return predictor
```

## 启用Python模式

配置完成后，启用Python模式来使用实际模型：

### 临时启用（当前会话）
```bash
export USE_PYTHON_SEGMENTATION=true
pnpm dev
```

### 永久启用
编辑 `.env.local` 文件：
```bash
USE_PYTHON_SEGMENTATION=true
```

## 测试模型

### 1. 测试Python脚本
```bash
python3 scripts/segment.py '{"image":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==","box":{"x":10,"y":10,"width":100,"height":100}}'
```

### 2. 测试Web界面
1. 访问 http://localhost:5000
2. 上传一张测试图像
3. 画一个框选择区域
4. 点击 "Run Segmentation"

## 常见问题

### Q1: 模型加载失败
**A:** 检查：
- 模型文件路径是否正确
- 模型文件是否完整下载
- 磁盘空间是否足够

### Q2: CUDA out of memory
**A:**
- 使用更小的模型（sam2_hiera_small.pt）
- 减小输入图像尺寸
- 如果没有GPU，使用CPU版本（已安装）

### Q3: 分割结果不准确
**A:**
- 尝试使用不同的模型（base_plus 或 large）
- 调整框的大小和位置
- 对于医学图像，考虑使用Medical SAM2

### Q4: 速度太慢
**A:**
- 使用small模型
- 启用CUDA（如果有GPU）
- 减小图像分辨率

## 性能优化

### 使用GPU加速
如果你有NVIDIA GPU，安装CUDA版本的PyTorch：
```bash
pip3 uninstall torch torchvision
pip3 install torch torchvision
```

### 批量处理
修改Python脚本支持批量处理多个框：
```python
boxes = [
    [x1, y1, x2, y2],
    [x3, y3, x4, y4],
    # ...
]
masks, scores, logits = predictor.predict(box=boxes, multimask_output=True)
```

## 支持的医学图像格式

当前支持：
- PNG, JPG, JPEG - 标准图像格式
- 未来可扩展支持：
  - DICOM (.dcm) - 医学影像标准格式
  - NIfTI (.nii) - 神经影像格式

## 联系与支持

如有问题，请查看：
- [SAM2官方文档](https://github.com/facebookresearch/segment-anything-2)
- [Medical SAM2 GitHub](https://github.com/bowang-lab/Medical-SAM2)
- 本项目的README.md
