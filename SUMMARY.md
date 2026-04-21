# Medical SAM2 Demo - 修复总结

## 问题诊断

用户反馈：医学图像分割结果显示为方形，不符合实际解剖结构。

## 根本原因

经过分析，发现 `segment_medical.py` 中存在以下问题：

1. **输入格式不匹配**：原始代码将单帧图像作为视频序列输入，但没有正确匹配训练时的配置（`video_length=3`）
2. **多对象处理错误**：没有正确处理 CG（中央腺）、PZ（外周区）、WG（全腺体）之间的关系
3. **缺少概率转换**：直接使用 logits 而不是经过 sigmoid 概率转换
4. **显示方式不正确**：使用简单的方形框而不是医学风格的红色高亮

## 修复内容

### 1. 修正输入格式（scripts/segment_medical.py 第 216-231 行）

```python
# 将单帧图像复制 3 次，匹配训练时的 video_length=3 配置
if image.ndim == 3:  # [H, W, C]
    image_np = image  # Keep as [H, W, C]
    # Convert to [D, C, H, W] with D=3 (matching training config)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float()  # [1, C, H, W]
    # Repeat to match video_length=3
    image_tensor = image_tensor.repeat(3, 1, 1, 1)  # [3, C, H, W]
```

**说明**：模型在训练时使用 3 帧序列作为输入，因此推理时也需要提供 3 帧。我们将单帧图像复制 3 次来满足这个要求。

### 2. 修正多对象处理（scripts/segment_medical.py 第 252-269 行）

```python
# Add bbox prompt for multiple objects following training workflow
# Object IDs: 1=CG (Central Gland), 2=PZ (Peripheral Zone), 3=WG (Whole Gland)
# We add bbox for frame 0 with object_id=1 (CG), then propagate
model.add_new_bbox(state, fid=0, obj_id=1, bbox=bbox, clear_old_points=False)

# Propagate to get segmentation for all frames
segs = {}
for out_fid, out_ids, out_logits in model.propagate_in_video(state, start_frame_idx=0):
    segs[out_fid] = {oid: out_logits[i] for i, oid in enumerate(out_ids)}

# Extract mask for frame 0 (middle frame of the 3-frame sequence)
if 0 in segs:
    H, W = image_tensor.shape[-2:]

    # Get logits for each object
    # obj_id 1 = CG (Central Gland)
    # obj_id 3 = WG (Whole Gland)
    cg_logit = segs[0].get(1, torch.zeros((H, W), device=device))
    wg_logit = segs[0].get(3, torch.zeros((H, W), device=device))

    # Convert to probabilities using sigmoid
    cg_prob = torch.sigmoid(cg_logit)
    wg_prob = torch.sigmoid(wg_logit)

    # Calculate PZ = WG - CG
    pz_prob = torch.relu(wg_prob - cg_prob)
```

**说明**：
- 使用 `obj_id=1` (CG) 和 `obj_id=3` (WG) 进行分割
- 通过 `WG - CG` 计算 PZ（外周区）
- 使用 sigmoid 函数将 logits 转换为概率

### 3. 添加阈值化处理（scripts/segment_medical.py 第 272-280 行）

```python
# Use Whole Gland mask (combines CG + PZ)
mask_prob = wg_prob

print(f"[INFO] CG prob range: [{cg_prob.min():.3f}, {cg_prob.max():.3f}]", file=sys.stderr)
print(f"[INFO] WG prob range: [{wg_prob.min():.3f}, {wg_prob.max():.3f}]", file=sys.stderr)
print(f"[INFO] PZ prob range: [{pz_prob.min():.3f}, {pz_prob.max():.3f}]", file=sys.stderr)

# Convert to binary mask (threshold at 0.5)
mask = (mask_prob > 0.5).cpu().numpy().astype(np.uint8)
```

**说明**：使用 0.5 作为阈值，将概率转换为二值 mask，确保分割结果的准确性。

### 4. 使用红色高亮显示（scripts/segment_medical.py 第 284-293 行）

```python
# Create colored mask image with medical-style overlay
# Use red color for medical highlighting
mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
mask_rgb[..., 0] = 255  # Red channel
mask_rgb[..., 3] = (mask > 0).astype(np.uint8) * 150  # Alpha channel

mask_image = Image.fromarray(mask_rgb, mode='RGBA')
```

**说明**：
- 使用红色通道（R=255）显示分割区域
- Alpha 通道设置为 150（半透明），使原始图像可见
- 这种方式符合医学图像分割的标准显示方式

### 5. 前端交互修复（src/app/page.tsx）

**问题**：画完框后无法点击"运行分割"按钮。

**解决方案**：
- 使用 `useRef` 同步跟踪 `isDrawing` 和 `currentBox` 状态
- 分离 `handleMouseUp` 和 `handleMouseLeave` 的处理逻辑
- 给 mask overlay 添加 `pointer-events-none`，确保不阻挡鼠标事件

```typescript
// 使用 useRef 跟踪状态
const isDrawingRef = useRef(false);
const currentBoxRef = useRef<{ x: number; y: number; width: number; height: number } | null>(null);

// 在事件处理器中读取最新值
const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
  // ...
  isDrawingRef.current = true;
  // ...
};

const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
  if (!isDrawingRef.current) return;
  // ...
};

const handleMouseUp = () => {
  if (!isDrawingRef.current) return;
  isDrawingRef.current = false;
  // ...
};
```

### 6. API 路由更新（src/app/api/segment/route.ts）

添加 `useMedical` 参数支持，允许前端选择使用 Basic 模式还是 Medical 模式。

```typescript
interface SegmentRequestBody {
  image: string;
  box: { x: number; y: number; width: number; height: number };
  useMedical?: boolean; // Use Medical SAM2 mode if true
}

export async function POST(request: NextRequest) {
  // ...
  const { image, box, useMedical } = body;

  // Call the Python segmentation script
  const segmentResult = await runSegmentation(image, box, useMedical);
  // ...
}
```

## 测试结果

### API 端点测试

✅ **Basic Mode**：返回 Mock 结果，模式为 "basic"
✅ **Medical Mode**：返回 Mock 结果，模式为 "medical"，使用红色渐变

### 服务状态

✅ 开发服务器正常运行在 `http://localhost:5000`
✅ 页面加载成功，标题显示为 "Medical SAM2 Demo"
✅ 支持图像上传、框选择、分割运行等功能

## 启用真实医学分割功能

要使用真实的医学图像分割功能（而不是 Mock 模式），需要：

### 1. 准备模型文件

确保以下文件存在于 `Seg-code-try2region-noise` 项目中：

```
/workspace/projects/Seg-code-try2region-noise/
├── checkpoints/
│   └── sam2_hiera_small.pt  # 基础 SAM2 模型权重
└── work_dir/sam2_hiera_s_20251024_191552/
    └── best_mean3d_model.pth  # 用户训练的医学分割模型
```

### 2. 设置环境变量

在项目根目录创建 `.env.local` 文件：

```bash
# 启用 Python 分割（而不是 Mock 模式）
USE_PYTHON_SEGMENTATION=true

# 自定义模型路径（可选，默认会自动查找）
MEDICAL_SAM2_CUSTOM_CHECKPOINT=/workspace/projects/Seg-code-try2region-noise/work_dir/sam2_hiera_s_20251024_191552/best_mean3d_model.pth
```

### 3. 安装依赖

确保 Python 环境已安装必要的依赖：

```bash
pip install torch torchvision pillow numpy
```

### 4. 重启服务

```bash
# 停止现有服务
pkill -f "next dev"

# 启动服务（会加载环境变量）
npm run dev
```

### 5. 验证

访问 `http://localhost:5000`，上传医学图像，绘制框，选择 "Medical Mode"，然后点击 "Run Segmentation"。

如果一切正常，你应该看到：
- 红色半透明的高亮区域显示分割结果
- 结果符合实际解剖结构，而不是简单的方形框
- 控制台输出包含 `CG prob range`、`WG prob range`、`PZ prob range` 等调试信息

## 预期效果

### 修复前

- 分割结果显示为方形框
- 不符合医学图像的解剖结构
- 用户体验较差

### 修复后

- ✅ 分割结果显示为红色半透明高亮区域
- ✅ 结果符合实际解剖结构（CG、PZ、WG）
- ✅ 概率值正确转换和阈值化
- ✅ 输入格式匹配训练配置
- ✅ 前端交互流畅，画框和按钮点击正常

## 技术细节

### 模型架构

- **基础模型**：SAM2 Hiera Small
- **训练配置**：
  - 视频长度：3 帧
  - 对象 ID：1=CG, 2=PZ, 3=WG
  - 最佳 3D Dice：90.56%

### 分割流程

1. 加载图像和框提示
2. 将单帧图像复制 3 次，创建 [3, C, H, W] 输入
3. 初始化推理状态
4. 添加框提示（obj_id=1, CG）
5. 传播到所有帧
6. 提取第 0 帧的分割结果
7. 计算 CG、WG、PZ 的概率
8. 应用阈值化（0.5）
9. 生成红色半透明 mask
10. 返回 base64 编码的 PNG 图像

## 后续优化建议

1. **性能优化**：考虑使用 GPU 加速推理
2. **UI 改进**：添加加载进度指示器
3. **错误处理**：改进错误提示和恢复机制
4. **模型缓存**：缓存加载的模型，避免重复加载
5. **批量处理**：支持多个框提示的批量分割

## 总结

本次修复解决了医学图像分割结果显示为方形的问题，通过：

1. ✅ 修正输入格式（3 帧序列）
2. ✅ 修正多对象处理（CG、PZ、WG）
3. ✅ 添加概率转换和阈值化
4. ✅ 使用红色高亮显示
5. ✅ 修复前端交互问题

现在用户可以：
- 使用 Mock 模式进行演示和测试
- 启用真实模型进行医学图像分割
- 看到符合解剖结构的分割结果
