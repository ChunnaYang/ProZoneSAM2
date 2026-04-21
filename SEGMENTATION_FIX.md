# 分割功能修复说明

## 问题描述

原始分割结果不正确，显示为方形区域，而不是实际的医学图像分割。

## 根本原因分析

通过分析 `Seg-code-try2region-noise/test.py` 和 `func_3d/function.py` 中的 `test_sam` 函数，发现原始实现存在以下问题：

### 1. 输入数据格式错误

**问题**: 使用单帧输入 [1, C, H, W]

**正确**: 应该使用视频序列输入 [D, C, H, W]，其中 D=3 (video_length)

```python
# 错误
image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()  # [1, C, H, W]

# 正确
image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()  # [1, C, H, W]
image_tensor = image_tensor.repeat(3, 1, 1, 1)  # [3, C, H, W]
```

### 2. 对象ID使用错误

**问题**: 只使用 obj_id=1

**正确**: 模型使用多个对象ID表示不同区域：
- obj_id=1: CG (Central Gland) - 中央腺体
- obj_id=2: PZ (Peripheral Zone) - 外周区
- obj_id=3: WG (Whole Gland) - 全腺体

关系: WG = CG + PZ

```python
# 错误
mask_logit = segs[0][1]  # 只使用 CG

# 正确
cg_logit = segs[0].get(1, torch.zeros((H, W), device=device))  # CG
wg_logit = segs[0].get(3, torch.zeros((H, W), device=device))  # WG
pz_prob = torch.relu(wg_prob - cg_prob)  # PZ = WG - CG
```

### 3. 缺少概率转换和阈值化

**问题**: 直接使用logits

**正确**: 需要先转换为概率，再阈值化

```python
# 错误
mask = mask_logit.squeeze().cpu().numpy()
mask = (mask > 0.5).astype(np.uint8)

# 正确
mask_prob = torch.sigmoid(wg_logit)  # 转换为概率
mask = (mask_prob > 0.5).cpu().numpy().astype(np.uint8)  # 阈值化
```

## 修复方案

### 修复的关键点

1. **输入格式**: 将单帧图像复制3次，匹配训练时的 video_length=3
2. **多对象处理**: 使用正确的对象ID（1=CG, 3=WG）
3. **概率转换**: 对logits应用sigmoid得到概率
4. **阈值化**: 使用0.5阈值得到二值mask
5. **日志输出**: 添加详细日志帮助调试

### 修复后的流程

```python
# 1. 准备输入: [D, C, H, W] where D=3
image_tensor = image_tensor.repeat(3, 1, 1, 1)

# 2. 初始化推理状态
state = model.val_init_state(imgs_tensor=image_tensor)

# 3. 添加bbox提示
model.add_new_bbox(state, fid=0, obj_id=1, bbox=bbox)

# 4. 传播分割
for out_fid, out_ids, out_logits in model.propagate_in_video(state, start_frame_idx=0):
    segs[out_fid] = {oid: out_logits[i] for i, oid in enumerate(out_ids)}

# 5. 提取多个对象的logits
cg_logit = segs[0].get(1, ...)  # Central Gland
wg_logit = segs[0].get(3, ...)  # Whole Gland

# 6. 转换为概率
cg_prob = torch.sigmoid(cg_logit)
wg_prob = torch.sigmoid(wg_logit)
pz_prob = torch.relu(wg_prob - cg_prob)

# 7. 使用Whole Gland作为最终mask
mask_prob = wg_prob
mask = (mask_prob > 0.5).numpy()
```

## 测试方法

### 1. 通过前端界面测试

1. 上传医学图像
2. 绘制包含感兴趣区域的框
3. 选择"Medical Mode"
4. 点击"Run Segmentation"
5. 查看结果：应该显示红色的分割mask，而不是方形

### 2. 通过命令行测试

```bash
python3 scripts/segment_medical.py \
  '{"image":"data:image/png;base64,...","box":{"x":100,"y":100,"width":200,"height":200}}'
```

### 3. 查看日志

运行时会在stderr输出详细日志：
- 输入形状
- Bbox坐标
- CG/WG/PZ的概率范围
- 分割结果

## 预期结果

修复后的分割应该：
- ✅ 不是简单的方形
- ✅ 符合医学图像的实际解剖结构
- ✅ 红色高亮显示分割区域
- ✅ 与训练时的分割流程一致

## 技术细节

### 模型训练配置

- **Video Length**: 3 frames
- **Prompt Frequency**: 1 (每帧都有提示)
- **Memory Bank Size**: 16 frames
- **Object IDs**: 1=CG, 2=PZ, 3=WG

### SAM2VideoPredictor 工作原理

1. **初始化状态**: 加载图像特征到内存
2. **添加提示**: 在指定帧添加bbox/point提示
3. **传播分割**: 利用时序一致性传播分割结果
4. **多对象跟踪**: 同时跟踪多个对象

### 为什么使用3帧输入？

- 训练时使用3帧序列
- 模型学习时序信息
- 中间帧（frame 0）的分割结果最可靠
- 增强分割的稳定性和准确性

## 参考资料

- `Seg-code-try2region-noise/test.py` - 测试脚本
- `Seg-code-try2region-noise/func_3d/function.py` - 核心测试函数
- `Seg-code-try2region-noise/func_3d/dataset.py` - 数据加载
- `MODEL_LOADING_GUIDE.md` - 模型加载指南

## 后续优化

如需进一步改进，可以考虑：
1. 支持更多对象（单独显示CG、PZ、WG）
2. 添加置信度显示
3. 支持点提示模式
4. 优化mask边缘平滑
5. 添加实时预览
