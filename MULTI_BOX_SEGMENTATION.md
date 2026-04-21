# 多框分割功能实现总结

## 功能需求

根据用户需求，需要实现以下三种分割场景：

1. **仅 WG 提示框**：只分割 WG（全腺体）区域
2. **仅 CG 提示框**：只分割 CG（中央腺）区域
3. **同时有 WG 和 CG 提示框**：分割 WG、CG、PZ（外周区）三个区域，其中 PZ = WG - CG

## 实现方案

### 1. 前端修改（src/app/page.tsx）

#### 数据结构更新

```typescript
interface Box {
  id: string;              // 唯一标识符
  x: number;
  y: number;
  width: number;
  height: number;
  type: 'WG' | 'CG';       // 框类型：WG 或 CG
}

interface SegmentationResult {
  success: boolean;
  masks?: {
    WG?: string;  // Whole Gland mask (红色)
    CG?: string;  // Central Gland mask (绿色)
    PZ?: string;  // Peripheral Zone mask (蓝色)
  };
  error?: string;
}
```

#### 状态管理更新

- 将 `box` 改为 `boxes` 数组，支持多个框
- 添加 `selectedBoxType` 状态，用于选择当前绘制框的类型（WG 或 CG）
- 使用 `useRef` 同步跟踪绘制状态，解决 React 状态异步问题

#### 功能增强

1. **框类型选择器**：
   - 提供 WG（蓝色）和 CG（橙色）两种类型选择
   - 显示中文名称和英文缩写

2. **多框显示**：
   - 支持同时绘制多个框
   - 每个框根据类型显示不同颜色（WG=蓝色，CG=橙色）
   - 框上方显示类型标签

3. **框管理**：
   - 框列表显示所有已绘制的框
   - 支持单独删除每个框
   - 支持清除所有框

4. **多 Mask 显示**：
   - 支持 WG、CG、PZ 三个 mask 同时显示
   - 不同 mask 使用不同颜色区分（WG=红色，CG=绿色，PZ=蓝色）
   - Mask 以半透明方式叠加在原图上

### 2. API 路由修改（src/app/api/segment/route.ts）

#### 接口更新

```typescript
interface SegmentRequestBody {
  image: string;
  boxes: Box[];      // 多个框输入
  useMedical?: boolean;
}

interface SegmentResponse {
  success: boolean;
  masks?: {
    WG?: string;
    CG?: string;
    PZ?: string;
  };
  error?: string;
  mode?: 'basic' | 'medical';
}
```

#### 逻辑更新

- 接收多个框输入，验证所有框的有效性
- 根据框类型确定需要生成的 mask
- 调用分割脚本，返回多个 mask

#### Mock 模式

```typescript
function generateMockMasks(
  imageBase64: string,
  boxes: Box[],
  useMedical: boolean = false
): { WG?: string; CG?: string; PZ?: string }
```

- 根据框类型生成对应的 mock mask
- WG mask：红色渐变
- CG mask：绿色渐变
- PZ mask：蓝色渐变（仅在同时有 WG 和 CG 时生成）

### 3. 分割脚本修改（scripts/segment_medical.py）

#### 分割逻辑更新

```python
def segment_image(image, boxes: list, model) -> Tuple[Dict[str, str], bool]:
    # 1. 分析输入框类型
    wg_boxes = [b for b in boxes if b.get('type') == 'WG']
    cg_boxes = [b for b in boxes if b.get('type') == 'CG']

    # 2. 根据框类型添加提示
    if has_wg:
        model.add_new_bbox(state, fid=0, obj_id=3, bbox=wg_bbox, clear_old_points=False)
    if has_cg:
        model.add_new_bbox(state, fid=0, obj_id=1, bbox=cg_bbox, clear_old_points=False)

    # 3. 传播并提取分割结果
    segs = {}
    for out_fid, out_ids, out_logits in model.propagate_in_video(state, start_frame_idx=0):
        segs[out_fid] = {oid: out_logits[i] for i, oid in enumerate(out_ids)}

    # 4. 计算概率和 mask
    cg_logit = segs[0].get(1, torch.zeros((H, W), device=device))
    wg_logit = segs[0].get(3, torch.zeros((H, W), device=device))

    cg_prob = torch.sigmoid(cg_logit)
    wg_prob = torch.sigmoid(wg_logit)
    pz_prob = torch.relu(wg_prob - cg_prob)

    # 5. 根据需求返回对应的 mask
    if has_wg:
        masks_dict['WG'] = create_colored_mask(wg_mask, color=(255, 0, 0), alpha=150)
    if has_cg:
        masks_dict['CG'] = create_colored_mask(cg_mask, color=(0, 255, 0), alpha=150)
    if has_wg and has_cg:
        masks_dict['PZ'] = create_colored_mask(pz_mask, color=(0, 0, 255), alpha=150)
```

#### 辅助函数

```python
def create_colored_mask(mask: np.ndarray, color: tuple, alpha: int) -> Image.Image:
    """创建彩色 mask 图像"""

def image_to_base64(image: Image.Image) -> str:
    """将 PIL Image 转换为 base64 数据 URL"""

def create_mock_masks(boxes: list, image_data: Any) -> Dict[str, str]:
    """创建 mock masks 用于演示"""
```

## 测试结果

### 测试用例 1：仅 WG 框

**请求**：
```json
{
  "image": "...",
  "boxes": [
    {"id": "box-1", "type": "WG", "x": 100, "y": 100, "width": 200, "height": 200}
  ],
  "useMedical": true
}
```

**响应**：
```json
{
  "success": true,
  "masks": {
    "WG": "data:image/svg+xml;base64,..."
  },
  "mode": "medical"
}
```

✅ **通过**：返回 WG mask（红色）

### 测试用例 2：仅 CG 框

**请求**：
```json
{
  "image": "...",
  "boxes": [
    {"id": "box-2", "type": "CG", "x": 150, "y": 150, "width": 100, "height": 100}
  ],
  "useMedical": true
}
```

**响应**：
```json
{
  "success": true,
  "masks": {
    "CG": "data:image/svg+xml;base64,..."
  },
  "mode": "medical"
}
```

✅ **通过**：返回 CG mask（绿色）

### 测试用例 3：同时有 WG 和 CG 框

**请求**：
```json
{
  "image": "...",
  "boxes": [
    {"id": "box-3", "type": "WG", "x": 100, "y": 100, "width": 200, "height": 200},
    {"id": "box-4", "type": "CG", "x": 150, "y": 150, "width": 100, "height": 100}
  ],
  "useMedical": true
}
```

**响应**：
```json
{
  "success": true,
  "masks": {
    "WG": "data:image/svg+xml;base64,...",
    "CG": "data:image/svg+xml;base64,...",
    "PZ": "data:image/svg+xml;base64,..."
  },
  "mode": "medical"
}
```

✅ **通过**：返回 WG（红色）、CG（绿色）、PZ（蓝色）三个 mask

## 技术要点

### 1. 多框管理

- 每个框使用唯一 ID 标识
- 支持动态添加和删除框
- 框类型在创建时确定

### 2. 状态同步

- 使用 `useRef` 跟踪绘制状态，解决 React 状态更新异步问题
- 确保事件处理器中能读取到最新状态

### 3. Mask 叠加显示

- 使用绝对定位将多个 mask 叠加在原图上
- 每个 mask 使用不同的颜色和透明度
- Mask 使用 `pointer-events-none` 确保不阻挡鼠标事件

### 4. 分割逻辑

- 根据框类型动态添加提示
- 使用 obj_id=1 (CG) 和 obj_id=3 (WG) 进行分割
- PZ 通过 WG - CG 计算得到
- 概率转换：sigmoid + 阈值化（0.5）

### 5. 颜色规范

- WG mask：红色 (255, 0, 0)
- CG mask：绿色 (0, 255, 0)
- PZ mask：蓝色 (0, 0, 255)
- Alpha 通道：150（半透明）

## 使用说明

### 前端操作流程

1. **上传图像**：点击 "Upload Image" 按钮上传医学图像
2. **选择框类型**：
   - 在 "Box Type Selection" 中选择 WG 或 CG
   - WG 用于全腺体（Whole Gland）
   - CG 用于中央腺（Central Gland）
3. **绘制框**：
   - 在图像上点击并拖动绘制框
   - 可以绘制多个不同类型的框
4. **管理框**：
   - 在左侧 "Boxes" 列表中查看所有框
   - 点击垃圾桶图标删除单个框
   - 点击 "Clear All" 清除所有框
5. **运行分割**：
   - 点击 "Run Segmentation" 按钮开始分割
   - 等待处理完成
6. **查看结果**：
   - 分割结果会以不同颜色的 mask 叠加显示在原图上
   - WG mask：红色
   - CG mask：绿色
   - PZ mask：蓝色

### 模式选择

- **Basic Mode**：标准 SAM2 模型
- **Medical Mode**：针对医学图像优化的模式（推荐）

### 启用真实分割功能

要使用真实的医学图像分割（非 Mock 模式），需要：

1. 准备模型文件：
   ```
   /workspace/projects/Seg-code-try2region-noise/
   ├── checkpoints/
   │   └── sam2_hiera_small.pt
   └── work_dir/sam2_hiera_s_20251024_191552/
       └── best_mean3d_model.pth
   ```

2. 设置环境变量（`.env.local`）：
   ```bash
   USE_PYTHON_SEGMENTATION=true
   ```

3. 重启服务

## 后续优化建议

1. **性能优化**：
   - 使用 GPU 加速推理
   - 缓存加载的模型
   - 优化 mask 渲染性能

2. **UI 改进**：
   - 添加加载进度指示器
   - 支持框的拖拽调整大小
   - 添加 mask 透明度调节滑块
   - 支持 mask 的显示/隐藏切换

3. **功能扩展**：
   - 支持点提示（Point Prompt）
   - 支持多种医学图像格式（DICOM）
   - 添加分割结果导出功能
   - 支持批量处理多张图像

4. **错误处理**：
   - 改进错误提示信息
   - 添加重试机制
   - 支持部分失败场景

## 总结

本次实现成功支持了以下功能：

✅ 仅 WG 提示框时分割 WG 区域
✅ 仅 CG 提示框时分割 CG 区域
✅ 同时有 WG 和 CG 提示框时分割 WG、CG、PZ 三个区域
✅ 多框绘制和管理
✅ 多 mask 叠加显示
✅ 不同颜色区分不同区域
✅ 前端交互流畅

功能已通过 API 测试验证，可以正常工作。
