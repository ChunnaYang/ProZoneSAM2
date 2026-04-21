# 用户绘制的框作为 SAM 模型提示输入的完整流程说明

## 问题确认

**是的，现在用户的绘制的框会作为提示输入到您的 SAM 模型中！**

## 完整流程

### 1. 前端绘制框
用户在网页上绘制 WG 或 CG 类型的框：

```typescript
// 前端 (src/app/page.tsx)
interface Box {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  type: 'WG' | 'CG';  // ← 用户选择框的类型
}

// 用户绘制后存储在 boxes 数组中
const [boxes, setBoxes] = useState<Box[]>([]);
```

### 2. 点击"运行分割"按钮
前端调用 API 路由：

```typescript
// 前端发送请求
const response = await fetch('/api/segment', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    image: imageData,  // Base64 编码的图像
    boxes: boxes,      // ← 用户绘制的框数组
    useMedical: true   // 使用医学模型
  })
});
```

### 3. API 路由接收请求
```typescript
// src/app/api/segment/route.ts
export async function POST(request: NextRequest) {
  const body: SegmentRequestBody = await request.json();
  const { image, boxes, useMedical } = body;

  // 直接调用 segment_medical.py
  const segmentResult = await runSegmentation(image, boxes, useMedical);

  // ... 返回结果
}
```

### 4. Python 脚本接收并处理框
```python
# scripts/segment_medical.py
def main():
    input_data = json.loads(args.input_json)
    image_b64 = input_data.get("image")
    boxes = input_data.get("boxes")  # ← 接收用户绘制的框数组

    # 加载模型
    model = load_model()

    # 调用分割函数，传入用户绘制的框
    masks_dict, success = segment_image(image, boxes, model)
```

### 5. 框作为提示输入到 SAM 模型
```python
# scripts/segment_medical.py
def segment_image(image, boxes: list, model):
    # 解析用户绘制的框
    wg_boxes = [b for b in boxes if b.get('type') == 'WG']  # 全腺体框
    cg_boxes = [b for b in boxes if b.get('type') == 'CG']  # 中央腺框

    # 初始化推理状态
    state = model.val_init_state(imgs_tensor=image_tensor)

    # 🔑 关键步骤：将用户绘制的框作为提示输入到模型
    if has_wg:
        wg_box = wg_boxes[0]
        wg_bbox = torch.tensor([
            wg_box['x'],           # ← 用户绘制的 x 坐标
            wg_box['y'],           # ← 用户绘制的 y 坐标
            wg_box['x'] + wg_box['width'],   # ← 用户绘制的宽度
            wg_box['y'] + wg_box['height']   # ← 用户绘制的高度
        ]).unsqueeze(0).to(device)

        # 将 WG 框作为提示输入到 SAM 模型
        model.add_new_bbox(
            state,
            fid=1,          # 帧索引（中间帧）
            obj_id=3,       # 对象 ID = 3 表示 WG（全腺体）
            bbox=wg_bbox,   # ← 用户绘制的框
            clear_old_points=False
        )
        print(f"[INFO] Added WG bbox (obj_id=3): {wg_bbox.tolist()}")

    if has_cg:
        cg_box = cg_boxes[0]
        cg_bbox = torch.tensor([
            cg_box['x'],
            cg_box['y'],
            cg_box['x'] + cg_box['width'],
            cg_box['y'] + cg_box['height']
        ]).unsqueeze(0).to(device)

        # 将 CG 框作为提示输入到 SAM 模型
        model.add_new_bbox(
            state,
            fid=1,
            obj_id=1,       # 对象 ID = 1 表示 CG（中央腺）
            bbox=cg_bbox,   # ← 用户绘制的框
            clear_old_points=False
        )
        print(f"[INFO] Added CG bbox (obj_id=1): {cg_bbox.tolist()}")
```

### 6. 模型推理
```python
# 前向传播，模型使用用户提供的框作为提示
for out_fid, out_ids, out_logits in model.propagate_in_video(state, start_frame_idx=0):
    segs[out_fid] = {oid: out_logits[i] for i, oid in enumerate(out_ids)}
```

### 7. 提取分割结果
```python
# 从中间帧获取分割结果
if 1 in segs:
    # obj_id 1 = CG（中央腺）
    cg_logit = segs[1].get(1, torch.zeros((H, W), device=device))
    cg_prob = torch.sigmoid(cg_logit)

    # obj_id 3 = WG（全腺体）
    wg_logit = segs[1].get(3, torch.zeros((H, W), device=device))
    wg_prob = torch.sigmoid(wg_logit)

    # PZ（外周区）= WG - CG（数学计算得出）
    pz_prob = torch.relu(wg_prob - cg_prob)
```

### 8. 返回多个 Mask
```python
masks_dict = {
    'WG': wg_mask_base64,  # 全腺体（红色）
    'CG': cg_mask_base64,  # 中央腺（绿色）
    'PZ': pz_mask_base64   # 外周区（蓝色，WG - CG）
}
```

### 9. 前端显示结果
```typescript
// 前端接收并显示多个 mask
const masks = response.masks;

// 叠加显示 WG、CG、PZ 三个 mask
{masks.WG && <img src={masks.WG} className="absolute w-full h-full" />}
{masks.CG && <img src={masks.CG} className="absolute w-full h-full" />}
{masks.PZ && <img src={masks.PZ} className="absolute w-full h-full" />}
```

## 关键确认点

### ✅ 用户的框是否作为提示输入？
**是的！** 在 `scripts/segment_medical.py` 中，用户的框通过以下方式输入到模型：

```python
model.add_new_bbox(state, fid=1, obj_id=3, bbox=wg_bbox, ...)
model.add_new_bbox(state, fid=1, obj_id=1, bbox=cg_bbox, ...)
```

其中 `wg_bbox` 和 `cg_bbox` 就是用户绘制的框的坐标。

### ✅ 使用了哪个模型？
- **基础模型**：`Seg-code-try2region-noise/checkpoints/sam2_hiera_small.pt`
- **自定义权重**：`Seg-code-try2region-noise/work_dir/sam2_hiera_s_20251024_191552/best_mean3d_model.pth`

### ✅ obj_id 映射
| obj_id | 区域 | 颜色 | 说明 |
|--------|------|------|------|
| 1      | CG (中央腺) | 🟢 绿色 | 用户绘制的 CG 框作为提示 |
| 3      | WG (全腺体) | 🔴 红色 | 用户绘制的 WG 框作为提示 |
| -      | PZ (外周区) | 🔵 蓝色 | 通过 WG - CG 计算得出 |

## 文件修改总结

1. **`src/app/api/segment/route.ts`** - 修改为直接调用 `segment_medical.py`（而不是 `segment.py`）
2. **`scripts/segment_medical.py`** - 已修正为：
   - 使用用户的 `get_network()` 函数加载模型
   - 使用灰度单通道输入格式 `[3, 1, H, W]`
   - 将用户绘制的框作为提示输入到模型
   - 返回多个 mask（WG、CG、PZ）

## 测试建议

1. 确认环境变量 `USE_PYTHON_SEGMENTATION=true` 已设置
2. 在前端绘制 WG 框和/或 CG 框
3. 点击"运行分割"
4. 查看 stderr 输出，确认：
   - 模型加载成功
   - 框坐标正确传递
   - 模型推理成功
5. 查看分割结果，应该符合解剖结构（而非方形）

## 日志输出示例

```
[INFO] Using device: cpu
[INFO] Loading custom trained model from .../best_mean3d_model.pth
[INFO] Custom checkpoint from epoch 10
[INFO] Best mean 3D Dice: 0.85
[INFO] Custom weights loaded successfully!
[INFO] Input shape: torch.Size([3, 1, 512, 512])
[INFO] Added WG bbox (obj_id=3): [[100, 150, 400, 350]]
[INFO] Added CG bbox (obj_id=1): [[150, 200, 350, 300]]
[INFO] Segments extracted for frames: [0, 1, 2]
[INFO] CG prob range: [0.012, 0.987]
[INFO] WG prob range: [0.045, 0.992]
[INFO] PZ prob range: [0.000, 0.876]
```

## 注意事项

1. **多框支持**：脚本支持同时绘制 WG 和 CG 框
2. **PZ 计算**：PZ 不是直接预测的，而是通过 WG - CG 计算得出
3. **降级机制**：如果模型加载失败，会自动降级到 Mock 模式
4. **性能**：真实模型推理可能需要几秒钟时间
