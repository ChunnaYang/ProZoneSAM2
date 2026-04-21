# Medical SAM2 模型集成修正说明

## 修正日期
2025-01-09

## 问题背景

用户反馈当前分割结果均为方形，未调用实际模型。经过分析，发现 `scripts/segment_medical.py` 使用了不正确的模型接口和输入格式。

## 核心问题

### 1. 模型加载方式不正确

**问题**：
- 之前使用的是通用的 `build_sam2_video_predictor`
- 没有正确加载用户的自定义训练权重

**修正**：
```python
# 使用用户的 get_network 函数（与 test.py 一致）
net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=args.gpu_device)

# 加载自定义权重
custom_state_dict = torch.load(custom_checkpoint_path, map_location='cpu')
if 'model' in custom_state_dict:
    model_weights = custom_state_dict['model']
net.load_state_dict(model_weights, strict=False)
```

### 2. 输入格式错误

**问题**：
```python
# 错误：使用 RGB 三通道
image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float()  # [1, C, H, W]
image_tensor = image_tensor.repeat(3, 1, 1, 1)  # [3, C, H, W]
```

**修正**：
```python
# 正确：转换为灰度并使用单通道
if image.shape[2] == 3:
    image_gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
image_np = image_gray.astype(np.float32)
image_tensor = torch.from_numpy(image_np).unsqueeze(0).float()  # [1, H, W]
image_tensor = image_tensor.unsqueeze(1).repeat(3, 1, 1, 1)  # [3, 1, H, W]
```

### 3. 推理接口不匹配

**修正后使用与 function.py 一致的接口**：
```python
# 初始化状态
state = model.val_init_state(imgs_tensor=image_tensor)

# 添加框提示
model.add_new_bbox(state, fid=1, obj_id=3, bbox=wg_bbox, clear_old_points=False)
model.add_new_bbox(state, fid=1, obj_id=1, bbox=cg_bbox, clear_old_points=False)

# 前向传播
for out_fid, out_ids, out_logits in model.propagate_in_video(state, start_frame_idx=0):
    segs[out_fid] = {oid: out_logits[i] for i, oid in enumerate(out_ids)}
```

### 4. Frame ID 调整

**问题**：
- 之前使用 fid=0 获取中间帧的分割结果

**修正**：
```python
# 使用 fid=1（3帧序列的中间帧：0, 1, 2）
if 1 in segs:
    cg_logit = segs[1].get(1, torch.zeros((H, W), device=device))
    wg_logit = segs[1].get(3, torch.zeros((H, W), device=device))
```

## obj_id 映射说明

根据用户的 `function.py` 代码，obj_id 的映射关系如下：

| obj_id | 区域名称 | 颜色 | 说明 |
|--------|----------|------|------|
| 1      | CG (中央腺) | 绿色 | Central Gland |
| 3      | WG (全腺体) | 红色 | Whole Gland |
| -      | PZ (外周区) | 蓝色 | Peripheral Zone，通过 WG - CG 计算得出 |

## PZ 计算方式

PZ 不是直接预测的，而是通过数学计算得出：

```python
# 计算概率
cg_prob = torch.sigmoid(cg_logit)
wg_prob = torch.sigmoid(wg_logit)

# PZ = WG - CG（使用 ReLU 确保非负）
pz_prob = torch.relu(wg_prob - cg_prob)

# 二值化
pz_mask = (pz_prob > 0.5)
```

## 预期效果

修正后的脚本应该能够：

1. ✅ 正确加载用户的自定义训练模型
2. ✅ 使用正确的输入格式（灰度单通道）
3. ✅ 调用实际模型进行推理
4. ✅ 输出符合解剖结构的 mask（而非方形）
5. ✅ 支持 WG、CG、PZ 三种区域的分割

## 使用说明

### Mock 模式（默认）
如果模型文件不可用，脚本会自动降级到 Mock 模式，生成方形 mask 用于演示。

### 真实推理模式
设置环境变量 `USE_PYTHON_SEGMENTATION=true` 即可启用真实模型推理：

```bash
export USE_PYTHON_SEGMENTATION=true
```

### 自定义模型路径
可以通过环境变量指定自定义模型路径：

```bash
export MEDICAL_SAM2_CUSTOM_CHECKPOINT=/path/to/your/model.pth
```

或者通过 API 传递 `checkpoint` 参数。

## 文件修改清单

- ✅ `scripts/segment_medical.py` - 修正模型加载、输入格式、推理接口

## 测试建议

1. 首先在 Mock 模式下验证前端交互是否正常
2. 确认模型文件存在：`Seg-code-try2region-noise/work_dir/.../best_mean3d_model.pth`
3. 设置 `USE_PYTHON_SEGMENTATION=true` 启用真实推理
4. 测试不同场景：
   - 仅 WG 框
   - 仅 CG 框
   - WG + CG 框（应生成 PZ）

## 注意事项

1. **模型文件**：确保用户的自定义模型文件存在且可访问
2. **GPU/CPU**：当前配置使用 CPU 进行推理（web demo 环境限制），如需 GPU 可修改配置
3. **性能**：真实模型推理可能需要几秒钟时间
4. **错误处理**：如果模型加载失败，脚本会自动降级到 Mock 模式

## 后续优化建议

1. 添加模型缓存机制，避免重复加载
2. 实现异步推理，提升用户体验
3. 添加更详细的日志和错误信息
4. 支持更多输入格式（如 NIfTI）
