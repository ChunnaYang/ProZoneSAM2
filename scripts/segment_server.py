#!/usr/bin/env python3
"""
Medical SAM2 Segmentation Server

这个脚本创建一个持久化的模型服务，只加载一次模型，
然后服务多个请求，大幅提升性能。

使用方法:
1. 启动服务器: python3 scripts/segment_server.py
2. 发送请求: curl -X POST http://localhost:8000/segment -d @data.json
"""

import sys
import json
import base64
import argparse
import os
import time
from io import BytesIO
from typing import Tuple, Dict, Any, Optional

# Try to import SAM2 packages
try:
    import torch
    import torch.nn as nn
    from PIL import Image
    import numpy as np
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from scipy import ndimage
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Error: Required packages not installed: {e}", file=sys.stderr)
    print("Please install: pip install torch pillow numpy fastapi uvicorn", file=sys.stderr)
    sys.exit(1)

# Try to import user's SAM2 project
MEDICAL_SAM2_AVAILABLE = False
model_cache = None

if TORCH_AVAILABLE:
    try:
        # Add the user's project to Python path
        sys.path.insert(0, '/workspace/projects/Seg-code-try2region-noise')

        # Temporarily set sys.argv to avoid parse_args() issues
        original_argv = sys.argv
        sys.argv = ['segment_medical.py']

        try:
            from sam2_train.build_sam import build_sam2_video_predictor
            MEDICAL_SAM2_AVAILABLE = True
        finally:
            sys.argv = original_argv
    except ImportError as e:
        print(f"Warning: Medical SAM2 not available: {e}", file=sys.stderr)

# Model configuration
PROJECT_DIR = '/workspace/projects/Seg-code-try2region-noise'
DEFAULT_MODEL_CONFIG = 'sam2_hiera_s'
DEFAULT_MODEL_CHECKPOINT = os.path.join(PROJECT_DIR, 'checkpoints/sam2_hiera_small.pt')
CUSTOM_MODEL_CHECKPOINT = os.getenv('MEDICAL_SAM2_CUSTOM_CHECKPOINT',
    os.path.join(PROJECT_DIR, 'work_dir/sam2_hiera_s_20251024_191552/best_mean3d_model.pth'))

# FastAPI app
app = FastAPI(title="Medical SAM2 Segmentation Server")

# Request/Response models
class Box(BaseModel):
    id: str
    x: float
    y: float
    width: float
    height: float
    type: str  # 'WG' or 'CG'

class SegmentRequest(BaseModel):
    image: str  # base64 data URL
    boxes: list[Box]
    use_medical: bool = True

class SegmentResponse(BaseModel):
    success: bool
    masks: dict[str, str]  # WG, CG, PZ
    mode: str = "medical"
    inference_time: float  # seconds


def load_model_once(checkpoint_path: str = None, use_custom: bool = True):
    """
    加载模型并缓存，只加载一次
    """
    global model_cache

    if model_cache is not None:
        return model_cache

    print(f"[INFO] Loading model (first time only)...", file=sys.stderr)
    start_time = time.time()

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {device}", file=sys.stderr)

        # Build model
        net = build_sam2_video_predictor(
            config_file=DEFAULT_MODEL_CONFIG,
            ckpt_path=DEFAULT_MODEL_CHECKPOINT,
            device=device,
            apply_postprocessing=False
        )

        # Load custom weights
        if use_custom:
            custom_checkpoint_path = checkpoint_path or CUSTOM_MODEL_CHECKPOINT

            if os.path.exists(custom_checkpoint_path):
                print(f"[INFO] Loading custom trained model from {custom_checkpoint_path}", file=sys.stderr)

                try:
                    custom_state_dict = torch.load(custom_checkpoint_path, map_location='cpu')

                    if 'model' in custom_state_dict:
                        model_weights = custom_state_dict['model']
                        epoch = custom_state_dict.get('epoch', 'unknown')
                        best_dice = custom_state_dict.get('best_mean3d_dice', 'N/A')
                        print(f"[INFO] Custom checkpoint from epoch {epoch}", file=sys.stderr)
                        print(f"[INFO] Best mean 3D Dice: {best_dice}", file=sys.stderr)
                    else:
                        model_weights = custom_state_dict

                    net.load_state_dict(model_weights, strict=False)
                    print("[INFO] Custom weights loaded successfully!", file=sys.stderr)

                except Exception as e:
                    print(f"[WARNING] Failed to load custom weights: {e}", file=sys.stderr)
            else:
                print(f"[INFO] Custom checkpoint not found at {custom_checkpoint_path}", file=sys.stderr)

        net.eval()
        model_cache = net

        load_time = time.time() - start_time
        print(f"[INFO] Model loaded in {load_time:.2f} seconds", file=sys.stderr)

        return net
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return None


def load_image_from_base64(base64_string: str):
    """Load image from base64 string."""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]

    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    return np.array(image)


def create_colored_mask(mask: np.ndarray, color: tuple = (255, 0, 0), alpha: int = 150) -> Image.Image:
    """Create a colored mask image from binary mask."""
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    mask_rgb[..., 0] = color[0]
    mask_rgb[..., 1] = color[1]
    mask_rgb[..., 2] = color[2]
    mask_rgb[..., 3] = (mask > 0).astype(np.uint8) * alpha
    return Image.fromarray(mask_rgb, mode='RGBA')


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 data URL."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{image_base64}"


def smooth_mask(mask_array: np.ndarray, min_size: int = 100, operation: str = 'both') -> np.ndarray:
    """
    对 mask 进行后处理，消除小的噪声区域和填充小空洞

    Args:
        mask_array: 输入的 mask 数组 (H, W)
        min_size: 最小区域大小，小于这个大小的区域会被移除
        operation: 形态学操作类型 ('opening', 'closing', 'both')

    Returns:
        平滑后的 mask 数组
    """
    # 确保是二值 mask
    if mask_array.max() > 1:
        mask_array = (mask_array > 0).astype(np.uint8)

    # 执行形态学操作
    if operation in ['opening', 'both']:
        # 开运算：先腐蚀后膨胀，消除小的噪声区域
        structure = ndimage.generate_binary_structure(2, 2)
        mask_array = ndimage.binary_opening(mask_array, structure=structure, iterations=1)

    if operation in ['closing', 'both']:
        # 闭运算：先膨胀后腐蚀，填充小的空洞
        structure = ndimage.generate_binary_structure(2, 2)
        mask_array = ndimage.binary_closing(mask_array, structure=structure, iterations=1)

    # 移除小的连通区域
    labeled, num_features = ndimage.label(mask_array)
    sizes = ndimage.sum(mask_array, labeled, range(num_features + 1))

    # 保留大于 min_size 的区域
    mask_smoothed = np.zeros_like(mask_array)
    for i in range(1, num_features + 1):
        if sizes[i] >= min_size:
            mask_smoothed[labeled == i] = 1

    return mask_smoothed


@app.post("/segment", response_model=SegmentResponse)
async def segment_endpoint(request: SegmentRequest):
    """
    分割端点
    """
    start_time = time.time()

    try:
        # Load model (cached)
        model = load_model_once()
        if model is None:
            raise HTTPException(status_code=500, detail="Model not available")

        # Load image
        image = load_image_from_base64(request.image)

        # Determine which boxes are present
        wg_boxes = [b for b in request.boxes if b.type == 'WG']
        cg_boxes = [b for b in request.boxes if b.type == 'CG']

        if not wg_boxes and not cg_boxes:
            raise HTTPException(status_code=400, detail="No valid boxes found")

        # Convert image to tensor
        if image.ndim == 3 and image.shape[2] == 3:
            image_gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        elif image.ndim == 2:
            image_gray = image
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        if image_gray.max() > 1.0:
            image_gray = image_gray / 255.0

        # Resize to 1024x1024 (model's expected input size)
        # Note: The model is trained on 1024x1024, changing this will cause errors
        target_size = 1024
        image_pil = Image.fromarray((image_gray * 255).astype(np.uint8))
        image_pil = image_pil.resize((target_size, target_size), Image.LANCZOS)
        image_np = np.array(image_pil).astype(np.float32) / 255.0

        # Convert to [3, 3, H, W] tensor
        image_tensor = torch.from_numpy(image_np).unsqueeze(0).float()
        image_tensor = image_tensor.unsqueeze(1).repeat(3, 3, 1, 1)

        # Get device
        device = next(model.parameters()).device
        image_tensor = image_tensor.to(device)

        # Initialize inference state
        state = model.val_init_state(
            imgs_tensor=image_tensor,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True
        )

        # Calculate scale factor
        orig_H, orig_W = image.shape[0], image.shape[1]
        scale_factor = target_size / max(orig_H, orig_W)

        # Add bbox prompts
        if wg_boxes:
            wg_box = wg_boxes[0]
            wg_bbox = torch.tensor([
                wg_box.x * scale_factor,
                wg_box.y * scale_factor,
                (wg_box.x + wg_box.width) * scale_factor,
                (wg_box.y + wg_box.height) * scale_factor
            ]).unsqueeze(0).to(device)
            model.add_new_bbox(state, frame_idx=1, obj_id=3, bbox=wg_bbox, clear_old_points=False)

        if cg_boxes:
            cg_box = cg_boxes[0]
            cg_bbox = torch.tensor([
                cg_box.x * scale_factor,
                cg_box.y * scale_factor,
                (cg_box.x + cg_box.width) * scale_factor,
                (cg_box.y + cg_box.height) * scale_factor
            ]).unsqueeze(0).to(device)
            model.add_new_bbox(state, frame_idx=1, obj_id=1, bbox=cg_bbox, clear_old_points=False)

        # Propagate
        segs = {}
        for out_fid, out_ids, out_logits in model.propagate_in_video(state, start_frame_idx=0):
            segs[out_fid] = {oid: out_logits[i] for i, oid in enumerate(out_ids)}

        # Extract masks
        if 1 in segs:
            H, W = image_tensor.shape[-2:]
            cg_logit = segs[1].get(1, torch.zeros((H, W), device=device))
            wg_logit = segs[1].get(3, torch.zeros((H, W), device=device))

            # Convert to probabilities
            cg_prob = torch.sigmoid(cg_logit)
            wg_prob = torch.sigmoid(wg_logit)
            pz_prob = torch.relu(wg_prob - cg_prob)

            # Resize masks back to original size
            from torch.nn.functional import interpolate

            def expand_for_interpolation(x):
                if x.ndim == 2:
                    return x.unsqueeze(0).unsqueeze(0)
                elif x.ndim == 3:
                    return x.unsqueeze(0)
                else:
                    return x.unsqueeze(1)

            wg_prob_resized = interpolate(
                expand_for_interpolation(wg_prob),
                size=(orig_H, orig_W),
                mode='bilinear',
                align_corners=False
            ).squeeze()

            cg_prob_resized = interpolate(
                expand_for_interpolation(cg_prob),
                size=(orig_H, orig_W),
                mode='bilinear',
                align_corners=False
            ).squeeze()

            pz_prob_resized = interpolate(
                expand_for_interpolation(pz_prob),
                size=(orig_H, orig_W),
                mode='bilinear',
                align_corners=False
            ).squeeze()

            # Create masks
            masks_dict = {}

            if wg_boxes:
                wg_mask = (wg_prob_resized > 0.5).cpu().numpy().astype(np.uint8)

                # Apply post-processing to smooth the mask
                wg_mask_smoothed = smooth_mask(wg_mask, min_size=100, operation='both')

                wg_mask_image = create_colored_mask(wg_mask_smoothed, color=(255, 0, 0), alpha=150)
                wg_mask_base64 = image_to_base64(wg_mask_image)
                masks_dict['WG'] = wg_mask_base64

            if cg_boxes:
                cg_mask = (cg_prob_resized > 0.5).cpu().numpy().astype(np.uint8)

                # Apply post-processing to smooth the mask
                cg_mask_smoothed = smooth_mask(cg_mask, min_size=100, operation='both')

                cg_mask_image = create_colored_mask(cg_mask_smoothed, color=(0, 255, 0), alpha=150)
                cg_mask_base64 = image_to_base64(cg_mask_image)
                masks_dict['CG'] = cg_mask_base64

            if wg_boxes and cg_boxes:
                pz_mask = (pz_prob_resized > 0.5).cpu().numpy().astype(np.uint8)

                # Apply post-processing to smooth the mask
                pz_mask_smoothed = smooth_mask(pz_mask, min_size=100, operation='both')

                pz_mask_image = create_colored_mask(pz_mask_smoothed, color=(0, 0, 255), alpha=150)
                pz_mask_base64 = image_to_base64(pz_mask_image)
                masks_dict['PZ'] = pz_mask_base64

            inference_time = time.time() - start_time
            return SegmentResponse(
                success=True,
                masks=masks_dict,
                mode="medical",
                inference_time=round(inference_time, 2)
            )
        else:
            raise HTTPException(status_code=500, detail="Segmentation failed")

    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "ok", "model_loaded": model_cache is not None}


def main():
    parser = argparse.ArgumentParser(description="Medical SAM2 Segmentation Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()

    print("="*60)
    print("Medical SAM2 Segmentation Server")
    print("="*60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("="*60)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
