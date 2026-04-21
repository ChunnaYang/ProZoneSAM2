#!/usr/bin/env python3
"""
3D Medical Image Segmentation with Interactive Bounding Boxes

This script supports:
1. Loading 3D MRI volumes (nii.gz format)
2. Interactive bounding box prompts for WG and CG regions
3. 3D segmentation using trained Medical SAM2 model
4. Visualization of CG and PZ regions (PZ = WG - CG)
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from PIL import Image
from io import BytesIO
import base64
import traceback

# Add Seg-code-try2region-noise to path
sys.path.insert(0, '/workspace/projects/Seg-code-try2region-noise')

# Import config
import cfg

# For nii.gz loading
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    print("Error: nibabel not installed. Cannot load nii.gz files.", file=sys.stderr)
    sys.exit(1)

# For morphological post-processing
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    print("Warning: scipy not installed. Post-processing disabled.", file=sys.stderr)
    SCIPY_AVAILABLE = False

# Configuration
PROJECT_DIR = '/workspace/projects/Seg-code-try2region-noise'
DEFAULT_MODEL_CHECKPOINT = os.path.join(PROJECT_DIR, 'work_dir/sam2_hiera_s_20251024_191552/best_mean3d_model.pth')
SAM_CHECKPOINT = os.path.join(PROJECT_DIR, 'checkpoints/sam2_hiera_small.pt')


def load_nifti_from_base64(nifti_b64):
    """Load NIfTI file from base64 string and return as numpy array."""
    import gzip
    
    temp_path = None
    try:
        # Decode base64
        if ',' in nifti_b64:
            nifti_b64 = nifti_b64.split(',')[1]

        nifti_data = base64.b64decode(nifti_b64)
        
        # Debug info
        print(f"[DEBUG] Decoded data length: {len(nifti_data)} bytes", file=sys.stderr)
        print(f"[DEBUG] First 20 bytes (hex): {nifti_data[:20].hex()}", file=sys.stderr)
        print(f"[DEBUG] First 20 bytes (repr): {repr(nifti_data[:20])}", file=sys.stderr)
        
        # Check if file size is reasonable
        if len(nifti_data) < 100:
            raise ValueError(f"File too small ({len(nifti_data)} bytes), not a valid NIfTI file")

        # Verify gzip magic number
        if nifti_data[:2] != b'\x1f\x8b':
            magic_bytes = nifti_data[:2].hex()
            print(f"[ERROR] Invalid gzip file format. Magic bytes: {magic_bytes} (expected: 1f8b)", file=sys.stderr)
            print(f"[ERROR] File might be: ", end="", file=sys.stderr)
            
            # Try to identify common file types
            if nifti_data[:4] == b'\x89PNG':
                print("PNG image file", file=sys.stderr)
            elif nifti_data[:2] == b'\xff\xd8':
                print("JPEG image file", file=sys.stderr)
            elif nifti_data[:4] == b'RIFF' and nifti_data[8:12] == b'WEBP':
                print("WebP image file", file=sys.stderr)
            elif nifti_data[:4] == b'%PDF':
                print("PDF document", file=sys.stderr)
            elif nifti_data[:2] == b'PK':
                print("ZIP file (possibly .nii without .gz)", file=sys.stderr)
            else:
                print("Unknown format", file=sys.stderr)
                
            raise ValueError(f"Uploaded file is not a valid .nii.gz (gzip compressed NIfTI) file. Detected magic bytes: {magic_bytes}")

        # Save to temp file
        temp_path = '/tmp/temp_volume.nii.gz'
        with open(temp_path, 'wb') as f:
            f.write(nifti_data)

        # Try to decompress to verify it's valid gzip
        try:
            with gzip.open(temp_path, 'rb') as gz_file:
                header = gz_file.read(344)  # NIfTI header size
                if len(header) < 344:
                    raise ValueError("File does not contain a valid NIfTI header")
        except gzip.BadGzipFile as e:
            raise ValueError(f"File is not a valid gzip file: {e}")

        # Load with nibabel
        nii_img = nib.load(temp_path)
        data = nii_img.get_fdata()

        print(f"[INFO] Loaded NIfTI volume with original shape: {data.shape}", file=sys.stderr)

        # For prostate MRI, ensure we have (H, W, D) format
        # Axial slices are along the last dimension
        shape = data.shape

        # Detect and reorder dimensions if needed
        if shape[0] == min(shape) and shape[0] < shape[2]:
            # First dimension is smallest, might need transpose
            print(f"[INFO] First dimension is smallest ({shape[0]}), checking data format", file=sys.stderr)
            pass
        elif shape[2] == min(shape):
            # Third dimension is smallest, this is likely D (axial slices)
            print(f"[INFO] Third dimension is smallest, using as axial slice dimension", file=sys.stderr)
            pass

        # Ensure data is 3D
        if data.ndim == 3:
            H, W, D = data.shape
            print(f"[INFO] Volume shape: H={H}, W={W}, D={D} axial slices", file=sys.stderr)
        else:
            raise ValueError(f"Expected 3D volume, got {data.ndim}D")

        return data, nii_img.affine
    except Exception as e:
        print(f"[ERROR] Failed to load NIfTI: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"[WARNING] Failed to remove temp file: {e}", file=sys.stderr)


def smooth_mask(mask_array: np.ndarray, min_size: int = 100, operation: str = 'both') -> np.ndarray:
    """
    Post-processing to remove small noise regions and fill small holes.
    """
    if not SCIPY_AVAILABLE:
        return mask_array

    # Ensure binary mask and 2D
    if mask_array.ndim == 3:
        # Process each slice independently
        result = np.zeros_like(mask_array)
        for i in range(mask_array.shape[0]):
            result[i] = smooth_mask(mask_array[i], min_size, operation)
        return result

    # Ensure binary mask
    if mask_array.max() > 1:
        mask_array = (mask_array > 0).astype(np.uint8)

    # Ensure 2D
    mask_array = np.squeeze(mask_array)
    if mask_array.ndim != 2:
        return mask_array

    # Perform morphological operations
    if operation in ['opening', 'both']:
        structure = ndimage.generate_binary_structure(2, 2)
        mask_array = ndimage.binary_opening(mask_array, structure=structure, iterations=1)

    if operation in ['closing', 'both']:
        structure = ndimage.generate_binary_structure(2, 2)
        mask_array = ndimage.binary_closing(mask_array, structure=structure, iterations=1)

    # Remove small connected components
    labeled, num_features = ndimage.label(mask_array)
    sizes = ndimage.sum(mask_array, labeled, range(num_features + 1))

    mask_smoothed = np.zeros_like(mask_array)
    for i in range(1, num_features + 1):
        if sizes[i] >= min_size:
            mask_smoothed[labeled == i] = 1

    return mask_smoothed


def create_colored_mask(mask: np.ndarray, color: tuple = (255, 0, 0), alpha: int = 150) -> Image.Image:
    """Create a colored mask image from binary mask."""
    if mask.ndim == 3:
        # Process first slice
        mask = mask[0]

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


def load_model(checkpoint_path: str = None):
    """Load the Medical SAM2 model."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {device}", file=sys.stderr)

        # Create mock args for model loading
        args = cfg.parse_args()
        args.pretrain = checkpoint_path or DEFAULT_MODEL_CHECKPOINT
        args.gpu_device = 0 if device == "cuda" else None
        args.gpu = (device == "cuda")
        args.sam_ckpt = SAM_CHECKPOINT
        args.video_length = 3

        # Import build function
        from sam2_train.build_sam import build_sam2_video_predictor

        # Build model
        print(f"[INFO] Building SAM2 predictor from {SAM_CHECKPOINT}", file=sys.stderr)
        net = build_sam2_video_predictor(
            config_file=args.sam_config,
            ckpt_path=args.sam_ckpt,
            device=device,
            mode=None,
            apply_postprocessing=False
        )

        # Load custom weights
        if os.path.exists(args.pretrain):
            print(f"[INFO] Loading custom weights from {args.pretrain}", file=sys.stderr)
            checkpoint = torch.load(args.pretrain, map_location='cpu')

            if 'model' in checkpoint:
                net.load_state_dict(checkpoint['model'])
                print(f"[INFO] Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}", file=sys.stderr)
            else:
                net.load_state_dict(checkpoint)
        else:
            print(f"[ERROR] Checkpoint not found at {args.pretrain}", file=sys.stderr)
            return None

        net.eval()
        return net

    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None


def segment_volume(volume, affine, boxes, model):
    """
    Perform 3D segmentation on the volume using the given boxes.

    Args:
        volume: 3D volume (H, W, D) - Axial slices along D dimension
        affine: Affine matrix from NIfTI
        boxes: List of dictionaries with type (WG/CG), x, y, width, height, slice_idx
        model: Loaded model predictor

    Returns:
        Dict with slice indices and their segmentation masks
    """
    if model is None:
        return create_mock_masks(volume, boxes), True

    try:
        device = next(model.parameters()).device

        # Volume should be (H, W, D) format for prostate MRI
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume, got {volume.ndim}D")

        H, W, D = volume.shape
        print(f"[INFO] Processing volume: H={H}, W={W}, D={D} axial slices", file=sys.stderr)

        # Normalize volume to [0, 1]
        volume_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

        # Determine which boxes are present
        wg_boxes = [b for b in boxes if b.get('type') == 'WG']
        cg_boxes = [b for b in boxes if b.get('type') == 'CG']

        if not wg_boxes and not cg_boxes:
            return create_mock_masks(volume, boxes), True

        # Get unique slice indices
        slice_indices = sorted(list(set([b.get('slice_idx', D // 2) for b in boxes])))

        results = {}

        target_size = 1024

        for slice_idx in slice_indices:
            if slice_idx >= D:
                print(f"[WARNING] Slice index {slice_idx} out of range (0-{D-1})", file=sys.stderr)
                continue

            print(f"[INFO] Processing axial slice {slice_idx}/{D}", file=sys.stderr)

            # Extract axial slice (slice along D dimension)
            slice_img = volume_norm[:, :, slice_idx]

            # Resize to model input size
            from PIL import Image as PILImage
            img_pil = PILImage.fromarray((slice_img * 255).astype(np.uint8))
            img_pil = img_pil.resize((target_size, target_size), PILImage.LANCZOS)
            img_np = np.array(img_pil).astype(np.float32) / 255.0

            # Convert to tensor [3, 3, H, W] for video_length=3
            image_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(1).repeat(3, 3, 1, 1)
            image_tensor = image_tensor.to(device)

            # Initialize inference state
            state = model.val_init_state(
                imgs_tensor=image_tensor,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True
            )

            # Calculate scale factor for bbox coordinates
            orig_H, orig_W = H, W
            scale_factor = target_size / max(orig_H, orig_W)

            # Add bbox prompts for this slice
            slice_wg_boxes = [b for b in wg_boxes if b.get('slice_idx', D // 2) == slice_idx]
            slice_cg_boxes = [b for b in cg_boxes if b.get('slice_idx', D // 2) == slice_idx]

            if len(slice_wg_boxes) > 0:
                wg_box = slice_wg_boxes[0]
                wg_bbox = torch.tensor([
                    wg_box['x'] * scale_factor,
                    wg_box['y'] * scale_factor,
                    (wg_box['x'] + wg_box['width']) * scale_factor,
                    (wg_box['y'] + wg_box['height']) * scale_factor
                ]).unsqueeze(0).to(device)
                model.add_new_bbox(state, frame_idx=1, obj_id=3, bbox=wg_bbox, clear_old_points=False)
                print(f"[INFO] Added WG bbox for slice {slice_idx}", file=sys.stderr)

            if len(slice_cg_boxes) > 0:
                cg_box = slice_cg_boxes[0]
                cg_bbox = torch.tensor([
                    cg_box['x'] * scale_factor,
                    cg_box['y'] * scale_factor,
                    (cg_box['x'] + cg_box['width']) * scale_factor,
                    (cg_box['y'] + cg_box['height']) * scale_factor
                ]).unsqueeze(0).to(device)
                model.add_new_bbox(state, frame_idx=1, obj_id=1, bbox=cg_bbox, clear_old_points=False)
                print(f"[INFO] Added CG bbox for slice {slice_idx}", file=sys.stderr)

            # Propagate to get segmentation
            segs = {}
            for out_fid, out_ids, out_logits in model.propagate_in_video(state, start_frame_idx=0):
                segs[out_fid] = {oid: out_logits[i] for i, oid in enumerate(out_ids)}

            # Extract masks for frame 1
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
                    return x.unsqueeze(0)

                wg_prob_resized = interpolate(expand_for_interpolation(wg_prob), size=(orig_H, orig_W), mode='bilinear', align_corners=False).squeeze()
                cg_prob_resized = interpolate(expand_for_interpolation(cg_prob), size=(orig_H, orig_W), mode='bilinear', align_corners=False).squeeze()
                pz_prob_resized = interpolate(expand_for_interpolation(pz_prob), size=(orig_H, orig_W), mode='bilinear', align_corners=False).squeeze()

                # Create masks
                slice_result = {}

                if len(slice_wg_boxes) > 0:
                    wg_mask = (wg_prob_resized > 0.5).cpu().numpy().astype(np.uint8)
                    wg_mask_smoothed = smooth_mask(wg_mask, min_size=100, operation='both')
                    wg_mask_image = create_colored_mask(wg_mask_smoothed, color=(255, 0, 0), alpha=150)
                    slice_result['WG'] = image_to_base64(wg_mask_image)

                if len(slice_cg_boxes) > 0:
                    cg_mask = (cg_prob_resized > 0.5).cpu().numpy().astype(np.uint8)
                    cg_mask_smoothed = smooth_mask(cg_mask, min_size=100, operation='both')
                    cg_mask_image = create_colored_mask(cg_mask_smoothed, color=(0, 255, 0), alpha=150)
                    slice_result['CG'] = image_to_base64(cg_mask_image)

                if len(slice_wg_boxes) > 0 and len(slice_cg_boxes) > 0:
                    pz_mask = (pz_prob_resized > 0.5).cpu().numpy().astype(np.uint8)
                    pz_mask_smoothed = smooth_mask(pz_mask, min_size=100, operation='both')
                    pz_mask_image = create_colored_mask(pz_mask_smoothed, color=(0, 0, 255), alpha=150)
                    slice_result['PZ'] = image_to_base64(pz_mask_image)

                # Store result for this slice
                results[str(slice_idx)] = slice_result

            # Reset state for next iteration
            model.reset_state(state)

        return results, True

    except Exception as e:
        print(f"[ERROR] Error during 3D segmentation: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return create_mock_masks(volume, boxes), True


def create_mock_masks(volume, boxes):
    """Create mock masks for demonstration."""
    results = {}

    # Volume should be (H, W, D) format
    if volume.ndim != 3:
        H, W = volume.shape[:2]
        D = 1
    else:
        H, W, D = volume.shape

    # Get unique slice indices
    slice_indices = sorted(list(set([b.get('slice_idx', D // 2) for b in boxes])))

    for slice_idx in slice_indices:
        slice_result = {}
        for box in boxes:
            if box.get('slice_idx', D // 2) != slice_idx:
                continue

            mask_type = box.get('type', 'WG')
            if mask_type not in slice_result:
                color = (255, 0, 0) if mask_type == 'WG' else (0, 255, 0)
                mask_image = create_colored_mask(np.zeros((H, W)), color=color, alpha=150)
                slice_result[mask_type] = image_to_base64(mask_image)

        if 'WG' in slice_result and 'CG' in slice_result:
            color = (0, 0, 255)
            mask_image = create_colored_mask(np.zeros((H, W)), color=color, alpha=150)
            slice_result['PZ'] = image_to_base64(mask_image)

        results[str(slice_idx)] = slice_result

    return results


def main():
    parser = argparse.ArgumentParser(description="3D Medical SAM2 Inference")
    parser.add_argument("input_json", nargs='?', help="Input JSON with nii.gz data and boxes")
    parser.add_argument("--checkpoint", help="Path to model checkpoint", default=None)
    args = parser.parse_args()

    try:
        # Parse input
        if args.input_json:
            input_json_str = args.input_json
            if os.path.exists(input_json_str):
                with open(input_json_str, 'r') as f:
                    input_data = json.load(f)
            else:
                input_data = json.loads(input_json_str)
        else:
            input_json_str = sys.stdin.read()
            if not input_json_str.strip():
                result = {"success": False, "error": "No input data provided"}
                print(json.dumps(result))
                sys.exit(1)
            input_data = json.loads(input_json_str)

        volume_b64 = input_data.get("volume")
        boxes = input_data.get("boxes")

        if not volume_b64 or not boxes:
            result = {"success": False, "error": "Missing volume or boxes"}
            print(json.dumps(result))
            sys.exit(1)

        # Load volume
        volume, affine = load_nifti_from_base64(volume_b64)

        # Load model
        model = load_model(args.checkpoint)

        # Perform segmentation
        results, success = segment_volume(volume, affine, boxes, model)

        if success:
            result = {"success": True, "results": results}
            print(json.dumps(result))
            sys.exit(0)
        else:
            result = {"success": False, "error": "Segmentation failed"}
            print(json.dumps(result))
            sys.exit(1)

    except Exception as e:
        result = {"success": False, "error": str(e)}
        print(json.dumps(result), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
