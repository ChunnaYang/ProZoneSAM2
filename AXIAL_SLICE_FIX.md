# 轴状位切片显示修复说明

## 问题描述
用户反馈上传nii.gz文件后显示的视图不正确，需要显示轴状位切片。

## 问题原因
原代码假设nii.gz文件的维度顺序是 (D, H, W)，但医学影像（特别是前列腺MRI）的常见格式是 (H, W, D)，其中：
- H = 高度（Height）
- W = 宽度（Width）
- D = 深度（Depth），代表轴状位切片数量

## 解决方案

### 1. 修复 `/api/load-volume` API
**修改内容：**
- 自动检测nii.gz文件的维度顺序
- 确保轴状位切片沿着D维度（通常是最后一个维度或最小维度）提取
- 修正切片提取方式：从 `data_norm[i]` 改为 `data_norm[:, :, i]`

**代码逻辑：**
```python
# 检测维度顺序
if shape[0] == min(shape):
    # 第一个维度是最小维度，可能需要转置
    data = np.transpose(data, (0, 1, 2))
elif shape[2] == min(shape) or shape[0] > shape[2]:
    # 第三个维度是切片维度，数据已经是(H, W, D)格式
    pass

# 确保数据格式为 (H, W, D)
H, W, D = data.shape

# 提取轴状位切片
for i in range(D):
    slice_img = data_norm[:, :, i]  # 沿D维度提取
```

### 2. 修复 `scripts/inference_3d.py`
**修改内容：**
- 更新 `load_nifti_from_base64()` 函数，确保返回 (H, W, D) 格式
- 更新 `segment_volume()` 函数，正确处理 (H, W, D) 格式的体积
- 更新 `create_mock_masks()` 函数，适配正确的维度顺序

**关键更改：**
```python
# segment_volume() 函数
# 原来: D, H, W = volume.shape
# 修改后: H, W, D = volume.shape

# 提取切片
# 原来: slice_img = volume_norm[slice_idx]
# 修改后: slice_img = volume_norm[:, :, slice_idx]
```

## 医学影像维度说明

### 前列腺MRI常见格式
对于前列腺MRI，典型的维度顺序是：
- **(H, W, D)**: 最常见
  - H: 高度，通常为256-512
  - W: 宽度，通常为256-512
  - D: 轴状位切片数量，通常为20-50

### 切片类型
1. **轴状位（Axial）**: 从头到脚的横切面，沿着Z轴（D维度）
2. **冠状位（Coronal）**: 从前到后的切面，沿着Y轴
3. **矢状位（Sagittal）**: 从左到右的切面，沿着X轴

### 轴状位切片显示
- 轴状位切片显示的是从头顶到脚底的横截面
- 在(H, W, D)格式中，每个切片是一个(H, W)的2D图像
- 切片索引从0到D-1，通常从下往上或从上往下排列

## 测试建议

1. **验证维度检测**
   - 上传一个nii.gz文件
   - 查看控制台输出的shape信息
   - 确认显示的是 (H, W, D) 格式

2. **验证切片显示**
   - 使用切片导航功能浏览所有切片
   - 确认显示的是轴状位（横截面）视图
   - 检查前列腺的解剖结构是否正确

3. **验证标注功能**
   - 在切片上绘制标注框
   - 运行分割
   - 确认分割结果与标注区域对应

## 后续改进

如果仍然遇到显示问题，可以考虑：
1. 让用户手动指定切片维度
2. 根据affine矩阵自动确定方向
3. 添加切片方向选择功能（轴状位、冠状位、矢状位）

## 文件变更清单

- ✅ `src/app/api/load-volume/route.ts` - 修复nii.gz加载逻辑
- ✅ `scripts/inference_3d.py` - 更新3D分割脚本
- ✅ 创建本文档说明修复内容
