# 分割结果为方形的问题分析与解决方案

## 问题根源

您看到的"标准大方形"分割结果是因为当前系统处于 **Mock 模式**，而非真实的模型推理。

### 为什么会显示方形？

在 `src/app/api/segment/route.ts` 中，当环境变量 `USE_PYTHON_SEGMENTATION` 未设置时，默认使用 Mock 模式：

```typescript
const usePython = process.env.USE_PYTHON_SEGMENTATION === 'true';

if (usePython) {
  return await runPythonSegmentation(imageBase64, boxes, useMedical);
} else {
  // Return a mock response for testing
  const mockMasks = generateMockMasks(imageBase64, boxes, useMedical);
  return { masks: mockMasks };
}
```

Mock 模式生成的 mask 是简单的方形，用于演示 UI 功能。

## 已完成的修正

### ✅ 1. API 路由修改
修改了 `src/app/api/segment/route.ts`，现在默认尝试使用 Python 分割：

```typescript
// Default to Python segmentation for medical mode
const forceMock = process.env.USE_PYTHON_SEGMENTATION === 'false';

if (!forceMock && useMedical) {
  return await runPythonSegmentation(imageBase64, boxes, useMedical);
} else if (forceMock) {
  const mockMasks = generateMockMasks(imageBase64, boxes, useMedical);
  return { masks: mockMasks };
} else {
  return await runPythonSegmentation(imageBase64, boxes, useMedical);
}
```

### ✅ 2. Python 脚本修正
修正了 `scripts/segment_medical.py`：
- 使用正确的模型加载方式
- 修正输入格式为 `[3, 3, H, W]`（3帧，3通道）
- 添加 CPU 模式支持
- 正确传递用户绘制的框作为提示

### ✅ 3. 依赖安装
已在沙箱环境中安装了必要的依赖：
- PyTorch 2.10.0 (CPU 版本)
- torchvision 0.25.0
- numpy, scipy, PIL
- medpy, nibabel, opencv-python-headless

## 当前问题

### ❌ 模型内部硬编码 `.cuda()` 调用

用户的训练模型在多个地方硬编码了 `.cuda()` 调用：

1. **`load_video_frames_from_data`** (Seg-code-try2region-noise/sam2_train/utils/misc.py:238):
   ```python
   if not offload_video_to_cpu:
       images = images.cuda()
   ```

2. **`_get_image_feature`** (Seg-code-try2region-noise/sam2_train/sam2_video_predictor.py:1278):
   ```python
   image = inference_state["images"][frame_idx].cuda().float().unsqueeze(0)
   ```

这些硬编码的 `.cuda()` 调用导致模型在 CPU 环境下无法运行。

## 解决方案

### 方案 1：修改用户模型代码（推荐）

在用户的 SAM2 代码中将所有 `.cuda()` 替换为 `.to(device)`：

#### 修改 1: `Seg-code-try2region-noise/sam2_train/utils/misc.py`

**第 238 行**：
```python
# 修改前
if not offload_video_to_cpu:
    images = images.cuda()

# 修改后
if not offload_video_to_cpu:
    images = images.to(images.device)
```

**第 239-240 行**：
```python
# 修改前
img_mean = img_mean.cuda()
img_std = img_std.cuda()

# 修改后
img_mean = img_mean.to(img_mean.device)
img_std = img_std.to(img_std.device)
```

#### 修改 2: `Seg-code-try2region-noise/sam2_train/sam2_video_predictor.py`

**第 1278 行**：
```python
# 修改前
image = inference_state["images"][frame_idx].cuda().float().unsqueeze(0)

# 修改后
device = inference_state["images"][frame_idx].device
image = inference_state["images"][frame_idx].to(device).float().unsqueeze(0)
```

#### 修改 3: 搜索所有 `.cuda()` 调用

```bash
cd Seg-code-try2region-noise
grep -r "\.cuda()" sam2_train/
```

将所有 `.cuda()` 替换为 `.to(device)` 或 `.to('cpu')`。

### 方案 2：在支持 GPU 的环境中运行

如果您的部署环境有 GPU，可以直接在 GPU 环境中运行，无需修改代码。

### 方案 3：使用 Mock 模式进行演示

如果只是演示 UI 功能，可以继续使用 Mock 模式：

```typescript
// 设置环境变量
process.env.USE_PYTHON_SEGMENTATION = 'false';
```

## 测试步骤

### 1. 修改用户模型代码后测试

```bash
# 测试 Python 分割脚本
python3 scripts/segment_medical.py '{
  "image": "data:image/png;base64,...",
  "boxes": [{"id": "test", "x": 100, "y": 100, "width": 200, "height": 200, "type": "WG"}]
}'
```

### 2. 前端测试

1. 打开网页：http://localhost:5000
2. 点击 "Load Sample Image" 加载示例图像
3. 选择框类型（WG 或 CG）
4. 在图像上绘制框
5. 点击 "Run Segmentation"
6. 查看分割结果

### 3. 检查日志

查看 stderr 输出，确认：
- 模型加载成功
- 框坐标正确传递
- 模型推理成功
- 没有 CUDA 相关错误

## 预期效果

修复后，分割结果应该：
- ✅ 符合医学解剖结构（而非方形）
- ✅ WG 为红色区域
- ✅ CG 为绿色区域
- ✅ PZ 为蓝色区域（WG - CG）

## 文件修改清单

1. **src/app/api/segment/route.ts** - 修改为默认使用 Python 分割
2. **scripts/segment_medical.py** - 修正模型加载、输入格式、CPU 支持
3. **Seg-code-try2region-noise/sam2_train/utils/misc.py** - 需要修改（替换 `.cuda()`）
4. **Seg-code-try2region-noise/sam2_train/sam2_video_predictor.py** - 需要修改（替换 `.cuda()`）

## 下一步

请按照上述修改方案修改用户模型代码，然后重新测试。如果您无法修改用户代码，请告诉我，我会尝试其他解决方案。
