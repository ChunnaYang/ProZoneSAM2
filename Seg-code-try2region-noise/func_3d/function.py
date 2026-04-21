import torch
import torch.nn as nn
from tqdm import tqdm
from monai.losses import DiceLoss
from func_3d.utils import dice_3d
import cfg
from conf import settings
import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
import numpy as np
import nibabel as nib
from monai.losses import DiceLoss, FocalLoss
args = cfg.parse_args()
GPUdevice = torch.device('cuda', args.gpu_device)

criterion_BCE = torch.nn.BCEWithLogitsLoss()
dice_func = DiceLoss(sigmoid=True)


# ===========================================================
# 🧩 辅助函数
# ===========================================================
def ensure_shape_4d(x):
    """确保 tensor 为 [1,1,H,W]"""
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:
        x = x.unsqueeze(1)
    return x


def normalize_img(img_np):
    """归一化灰度图像并处理多通道输入"""
    if img_np.ndim == 3:  # (C,H,W)
        img_np = np.mean(img_np, axis=0)
    return (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)


def get_slice(vol, fid):
    """兼容 2D / 3D 输出张量，返回 (H,W) 切片"""
    if vol.ndim == 5:  # [B, C, D, H, W]
        return vol[0, 0, fid, :, :].cpu()
    elif vol.ndim == 4:  # [B, C, H, W]
        return vol[0, 0, :, :].cpu()
    else:
        raise ValueError(f"Unexpected shape {vol.shape}")

def add_bbox_noise(bbox, noise_range=50, image_size=1024):
    """为 GT bbox 添加 0~noise_range 像素随机扰动"""
    if bbox.ndim == 1:
        bbox = bbox.clone()
    else:
        bbox = bbox[0].clone()

    noise = torch.randint(-noise_range, noise_range + 1, (4,), device=bbox.device)
    bbox_noisy = bbox + noise
    # 保证合法边界
    bbox_noisy[0::2] = bbox_noisy[0::2].clamp(0, image_size - 1)  # x1,x2
    bbox_noisy[1::2] = bbox_noisy[1::2].clamp(0, image_size - 1)  # y1,y2
    bbox_noisy[2] = max(bbox_noisy[2], bbox_noisy[0] + 1)
    bbox_noisy[3] = max(bbox_noisy[3], bbox_noisy[1] + 1)
    return bbox_noisy
# ===========================================================

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=1/21, focal_weight=20/21):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        # ✅ 你的 MONAI 版本 FocalLoss 不支持 sigmoid 参数
        self.dice_loss = DiceLoss(sigmoid=True, to_onehot_y=True)
        self.focal_loss = FocalLoss(to_onehot_y=True, gamma=2.0)

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        return self.dice_weight * dice + self.focal_weight * focal

# ===========================================================
# 🧩 TRAIN (使用 3D Dice 监督)
# ===========================================================
def train_sam(args, net: nn.Module, optimizer1, optimizer2, train_loader, epoch):
    net.train()
    epoch_loss = 0
    video_length = args.video_length
    prompt, prompt_freq = args.prompt, args.prompt_freq
    vis_enabled = getattr(args, "train_vis", False)
    os.makedirs("train_vis", exist_ok=True)

    with tqdm(total=len(train_loader), desc=f"[Train Epoch {epoch}]", unit='batch') as pbar:
        for batch_idx, pack in enumerate(train_loader):
            torch.cuda.empty_cache()
            imgs_tensor = pack["image"].squeeze(0).float().to(GPUdevice)
            mask_dict = pack["label"]

            if prompt == "bbox":
                bbox_dict = pack["bbox"]
            elif prompt == "click":
                pt_dict, point_labels_dict = pack["pt"], pack["p_label"]

            state = net.train_init_state(imgs_tensor=imgs_tensor)
            prompt_frames = list(range(0, video_length, prompt_freq))

            obj_list = []
            for fid in prompt_frames:
                obj_list += list(mask_dict[fid].keys())
            obj_list = list(set(obj_list))
            if not obj_list:
                continue

            with torch.cuda.amp.autocast():
                # ---- 添加提示 ----
                for fid in prompt_frames:
                    for obj_id in obj_list:
                        try:
                            if prompt == "bbox":
                                bbox = bbox_dict[fid][obj_id]
                                if args.det_source == "gt" and args.noise > 0:
                                    bbox = add_bbox_noise(bbox, noise_range=args.noise, image_size=args.image_size)
                                net.train_add_new_bbox(state, fid, obj_id, bbox.to(GPUdevice), clear_old_points=False)
                            elif prompt == "click":
                                points = pt_dict[fid][obj_id].to(GPUdevice)
                                labels = point_labels_dict[fid][obj_id].to(GPUdevice)
                                net.train_add_new_points(state, fid, obj_id, points, labels, clear_old_points=False)
                        except KeyError:
                            net.train_add_new_mask(
                                state, fid, obj_id, torch.zeros(imgs_tensor.shape[2:]).to(GPUdevice)
                            )

                # ---- 前向传播 ----
                segs = {}
                for out_fid, out_ids, out_logits in net.train_propagate_in_video(state, start_frame_idx=0):
                    segs[out_fid] = {oid: out_logits[i] for i, oid in enumerate(out_ids)}

                # ===== 聚合成 3D =====
                wg_preds, cg_preds, pz_preds = [], [], []
                wg_gts, cg_gts, pz_gts = [], [], []

                for fid in range(video_length):
                    H, W = imgs_tensor.shape[-2:]
                    wg_logit = ensure_shape_4d(segs[fid].get(3, torch.zeros((H, W), device=GPUdevice)))
                    cg_logit = ensure_shape_4d(segs[fid].get(1, torch.zeros((H, W), device=GPUdevice)))

                    wg_prob = torch.sigmoid(wg_logit)
                    cg_prob = torch.sigmoid(cg_logit)
                    pz_pred = torch.relu(wg_prob - cg_prob)

                    cg_gt = ensure_shape_4d(mask_dict[fid].get(1, torch.zeros((H, W))).to(GPUdevice)).float()
                    pz_gt = ensure_shape_4d(mask_dict[fid].get(2, torch.zeros((H, W))).to(GPUdevice)).float()
                    wg_gt = torch.clamp(cg_gt + pz_gt, 0, 1).float()

                    wg_preds.append(wg_prob.unsqueeze(2))
                    cg_preds.append(cg_prob.unsqueeze(2))
                    pz_preds.append(pz_pred.unsqueeze(2))
                    wg_gts.append(wg_gt.unsqueeze(2))
                    cg_gts.append(cg_gt.unsqueeze(2))
                    pz_gts.append(pz_gt.unsqueeze(2))

                wg_vol_pred = torch.cat(wg_preds, dim=2)
                cg_vol_pred = torch.cat(cg_preds, dim=2)
                pz_vol_pred = torch.cat(pz_preds, dim=2)
                wg_vol_gt = torch.cat(wg_gts, dim=2)
                cg_vol_gt = torch.cat(cg_gts, dim=2)
                pz_vol_gt = torch.cat(pz_gts, dim=2)

                # === 3D 损失 ===
                #criterion = CombinedLoss(dice_weight=1/21, focal_weight=20/21).to(GPUdevice)

                #loss1 = criterion(wg_vol_pred, wg_vol_gt)
                #loss2 = criterion(cg_vol_pred, cg_vol_gt)
                #loss3 = criterion(pz_vol_pred, pz_vol_gt)
                loss1 = criterion_BCE(wg_vol_pred, wg_vol_gt) + (1 - dice_3d(wg_vol_pred, wg_vol_gt))
                loss2 = criterion_BCE(cg_vol_pred, cg_vol_gt) + (1 - dice_3d(cg_vol_pred, cg_vol_gt))
                loss3 = criterion_BCE(pz_vol_pred, pz_vol_gt) + (1 - dice_3d(pz_vol_pred, pz_vol_gt))
                loss4 = torch.mean(torch.relu(cg_vol_pred - wg_vol_pred))
                loss5 = criterion_BCE(wg_vol_pred, wg_vol_gt)
                loss6 = criterion_BCE(cg_vol_pred, cg_vol_gt)
                #total_loss = loss1 + loss2 + 2 * loss3 +  2 * loss4
                #total_loss = loss5 + loss6
                #total_loss = loss1 + loss2 + 2 * loss4
                total_loss = loss1 + loss2
                # ---- 可视化单帧2D ----rii
                if vis_enabled and batch_idx == 0 and epoch % 1 == 0:
                    fid = min(video_length // 2, wg_vol_pred.shape[2] - 1)
                    img_np = imgs_tensor[fid].cpu().squeeze().numpy()
                    img_norm = normalize_img(img_np)

                    plt.figure(figsize=(12, 5))
                    plt.subplot(1, 2, 1)
                    plt.imshow(img_norm, cmap="gray")
                    plt.imshow((get_slice(wg_vol_pred, fid) > 0.5), cmap="Reds", alpha=0.4)
                    plt.title("WG Prediction"); plt.axis("off")

                    plt.subplot(1, 2, 2)
                    plt.imshow(img_norm, cmap="gray")
                    plt.imshow((get_slice(cg_vol_pred, fid) > 0.5), cmap="Blues", alpha=0.4)
                    plt.imshow((get_slice(pz_vol_pred, fid) > 0.5), cmap="Greens", alpha=0.4)
                    plt.title("CG + PZ Prediction"); plt.axis("off")

                    plt.suptitle(f"Train Visualization | Epoch {epoch}")
                    plt.tight_layout()
                    plt.savefig(f"train_vis/epoch{epoch:03d}.png", dpi=150, bbox_inches="tight")
                    plt.close()

                optimizer1.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer1.step()
                epoch_loss += total_loss.item()

            net.reset_state(state)
            pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})
            pbar.update()

    return epoch_loss / len(train_loader), 0, 0


# ===========================================================
# 🧪 VALIDATION (3D Dice + 同步2D可视化)
# ===========================================================
def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
    net.eval()
    n_val = len(val_loader)
    vis_enabled = getattr(args, "vis", False)
    if vis_enabled:
        print(f"[VAL] 2D Visualization enabled (epoch {epoch})")

    wg_dice_sum = cg_dice_sum = pz_dice_sum = 0
    case_count = 0

    with tqdm(total=n_val, desc=f"[Val Epoch {epoch}]", unit="batch", leave=False) as pbar:
        for pack in val_loader:
            imgs_tensor = pack["image"].squeeze(0).float().to(GPUdevice)
            mask_dict = pack["label"]
            case_count += 1
            case_name = f"case_{case_count:03d}"
            # === Debug: 打印 GT mask 的 obj_id 和均值 ===
            #print(f"\n[DEBUG_GT] {case_name}")
            #for fid in range(len(mask_dict)):
                #print(f"  fid={fid}, mask_dict keys: {list(mask_dict[fid].keys())}")
                #for k, v in mask_dict[fid].items():
                    #print(f"    obj_id={k} | mask mean={v.float().mean().item():.4f}")
                # 只打印前几帧避免太多输出
                #if fid >= 2:
                    #print("  ... (仅展示前3帧)")
                    #break

            if args.prompt == "bbox":
                bbox_dict = pack["bbox"]
            elif args.prompt == "click":
                pt_dict, point_labels_dict = pack["pt"], pack["p_label"]

            frame_id = list(range(imgs_tensor.size(0)))
            state = net.val_init_state(imgs_tensor=imgs_tensor)
            prompt_frames = list(range(0, len(frame_id), args.prompt_freq))
            obj_list = []
            for fid in frame_id:
                obj_list += list(mask_dict[fid].keys())
            obj_list = list(set(obj_list))
            if not obj_list:
                continue

            with torch.no_grad():
                for fid in prompt_frames:
                    for obj_id in obj_list:
                        try:
                            if args.prompt == "bbox":
                                bbox = bbox_dict[fid][obj_id]
                                if args.det_source == "gt" and args.noise > 0:
                                    bbox = add_bbox_noise(bbox, noise_range=args.noise, image_size=args.image_size)
                                net.add_new_bbox(state, fid, obj_id, bbox.to(GPUdevice), clear_old_points=False)
                            elif args.prompt == "click":
                                points = pt_dict[fid][obj_id].to(GPUdevice)
                                labels = point_labels_dict[fid][obj_id].to(GPUdevice)
                                net.add_new_points(state, fid, obj_id, points, labels, clear_old_points=False)
                        except KeyError:
                            net.add_new_mask(
                                state, fid, obj_id, torch.zeros(imgs_tensor.shape[2:]).to(GPUdevice)
                            )

                segs = {}
                for out_fid, out_ids, out_logits in net.propagate_in_video(state, start_frame_idx=0):
                    segs[out_fid] = {oid: out_logits[i] for i, oid in enumerate(out_ids)}

                wg_preds, cg_preds, pz_preds, wg_gts, cg_gts, pz_gts = [], [], [], [], [], []

                for fid in frame_id:
                    H, W = imgs_tensor.shape[-2:]
                    wg_logit = ensure_shape_4d(segs[fid].get(3, torch.zeros((H, W), device=GPUdevice)))
                    cg_logit = ensure_shape_4d(segs[fid].get(1, torch.zeros((H, W), device=GPUdevice)))
                    #print(f"\n[DEBUG] fid={fid} segs keys:", segs[fid].keys())
                    #for k, v in segs[fid].items():
                        #print(f"   obj_id={k} | pred mean={torch.sigmoid(v).mean().item():.4f}")

                    wg_pred = torch.sigmoid(wg_logit)
                    cg_pred = torch.sigmoid(cg_logit)
                    pz_pred = torch.relu(wg_pred - cg_pred)

                    cg_gt = ensure_shape_4d(mask_dict[fid].get(1, torch.zeros((H, W))).to(GPUdevice)).float()
                    pz_gt = ensure_shape_4d(mask_dict[fid].get(2, torch.zeros((H, W))).to(GPUdevice)).float()
                    wg_gt = torch.clamp(cg_gt + pz_gt, 0, 1).float()

                    wg_preds.append(wg_pred.unsqueeze(2))
                    cg_preds.append(cg_pred.unsqueeze(2))
                    pz_preds.append(pz_pred.unsqueeze(2))
                    wg_gts.append(wg_gt.unsqueeze(2))
                    cg_gts.append(cg_gt.unsqueeze(2))
                    pz_gts.append(pz_gt.unsqueeze(2))
                
                wg_vol_pred = torch.cat(wg_preds, dim=2)
                cg_vol_pred = torch.cat(cg_preds, dim=2)
                pz_vol_pred = torch.cat(pz_preds, dim=2)
                wg_vol_gt = torch.cat(wg_gts, dim=2)
                cg_vol_gt = torch.cat(cg_gts, dim=2)
                pz_vol_gt = torch.cat(pz_gts, dim=2)
                #print(f"[DEBUG] Shapes: WG_pred={wg_vol_pred.shape}, CG_pred={cg_vol_pred.shape}, PZ_pred={pz_vol_pred.shape}")
                #print(f"[DEBUG] Means: WG_pred={wg_vol_pred.mean().item():.4f}, CG_pred={cg_vol_pred.mean().item():.4f}, PZ_pred={pz_vol_pred.mean().item():.4f}")
                #print(f"[DEBUG] GT means: WG_gt={wg_vol_gt.mean().item():.4f}, CG_gt={cg_vol_gt.mean().item():.4f}, PZ_gt={pz_vol_gt.mean().item():.4f}")

                wg_dice = dice_3d(wg_vol_pred, wg_vol_gt)
                cg_dice = dice_3d(cg_vol_pred, cg_vol_gt)
                pz_dice = dice_3d(pz_vol_pred, pz_vol_gt)
                wg_dice_sum += wg_dice
                cg_dice_sum += cg_dice
                pz_dice_sum += pz_dice

                # ---- 可视化2D ----
                if vis_enabled:
                    save_root = f"vis_results/epoch_{epoch:03d}"
                    os.makedirs(save_root, exist_ok=True)
                    fid = min(len(frame_id) // 2, wg_vol_pred.shape[2] - 1)
                    img_np = imgs_tensor[fid].cpu().squeeze().numpy()
                    img_norm = normalize_img(img_np)

                    plt.figure(figsize=(12, 5))
                    plt.subplot(1, 2, 1)
                    plt.imshow(img_norm, cmap="gray")
                    plt.imshow((get_slice(wg_vol_pred, fid) > 0.5), cmap="Reds", alpha=0.4)
                    plt.title(f"{case_name} | WG"); plt.axis("off")

                    plt.subplot(1, 2, 2)
                    plt.imshow(img_norm, cmap="gray")
                    plt.imshow((get_slice(cg_vol_pred, fid) > 0.5), cmap="Blues", alpha=0.4)
                    plt.imshow((get_slice(pz_vol_pred, fid) > 0.5), cmap="Greens", alpha=0.4)
                    plt.title(f"{case_name} | CG + PZ"); plt.axis("off")

                    plt.tight_layout()
                    plt.savefig(os.path.join(save_root, f"{case_name}.png"), dpi=200, bbox_inches="tight")
                    plt.close()

                    # 可选：保存 NIfTI 体积
                    wg_vol = (wg_vol_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
                    cg_vol = (cg_vol_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
                    pz_vol = (pz_vol_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
                    #nib.save(nib.Nifti1Image(wg_vol, affine=np.eye(4)),
                             #os.path.join(save_root, f"{case_name}_WG.nii.gz"))
                    #nib.save(nib.Nifti1Image(cg_vol + 2 * pz_vol, affine=np.eye(4)),
                             #os.path.join(save_root, f"{case_name}_CGPZ.nii.gz"))

            net.reset_state(state)
            pbar.update()

    wg_dice_mean = wg_dice_sum / max(case_count, 1)
    cg_dice_mean = cg_dice_sum / max(case_count, 1)
    pz_dice_mean = pz_dice_sum / max(case_count, 1)
    avg_3d = (wg_dice_mean + cg_dice_mean + pz_dice_mean) / 3

    print(f"[VAL] Epoch {epoch}")
    print(f"  3D Dice => WG={wg_dice_mean:.4f} | CG={cg_dice_mean:.4f} | PZ={pz_dice_mean:.4f} | Mean={avg_3d:.4f}")

    return avg_3d, (wg_dice_mean, cg_dice_mean, pz_dice_mean), {
        "WG_3D": wg_dice_mean, "CG_3D": cg_dice_mean, "PZ_3D": pz_dice_mean
    }


def test_sam(args, test_loader, net: nn.Module, save_root="test_results_noise", testset_name="test"):
    """
    测试阶段：
    - 计算 3D Dice / IoU / HD95
    - 保存每位患者的结果至 CSV
    - 输出可视化图和 .nii.gz 分割体积（方向与GT对齐）
    - 自动添加平均结果行（Mean）
    - 每个患者的每个有标注切片生成4列对比图（GT/PRED）
    """
    import pandas as pd
    from func_3d.utils import dice_3d, iou_3d, hd95_3d
    import os
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import zoom

    net.eval()
    os.makedirs(save_root, exist_ok=True)
    subset_dir = os.path.join(save_root, testset_name)
    os.makedirs(subset_dir, exist_ok=True)

    results = []
    with tqdm(total=len(test_loader), desc=f"[TEST | {testset_name}]", unit="case") as pbar:
        for pack in test_loader:
            imgs_tensor = pack["image"].squeeze(0).float().to(GPUdevice)
            mask_dict = pack["label"]

            case_meta = pack["image_meta_dict"]["filename_or_obj"]
            if isinstance(case_meta, (list, tuple)):
                case_meta = case_meta[0]
            case_name = os.path.basename(str(case_meta))
            case_name = os.path.splitext(case_name)[0]

            # === 获取提示 ===
            if args.prompt == "bbox":
                bbox_dict = pack["bbox"]
            elif args.prompt == "click":
                pt_dict, point_labels_dict = pack["pt"], pack["p_label"]

            frame_id = list(range(imgs_tensor.size(0)))
            state = net.val_init_state(imgs_tensor=imgs_tensor)
            prompt_frames = list(range(0, len(frame_id), args.prompt_freq))
            obj_list = []
            for fid in frame_id:
                obj_list += list(mask_dict[fid].keys())
            obj_list = list(set(obj_list))
            if not obj_list:
                continue

            with torch.no_grad():
                for fid in prompt_frames:
                    for obj_id in obj_list:
                        try:
                            if args.prompt == "bbox":
                                bbox = bbox_dict[fid][obj_id]
                                if args.det_source == "gt" and args.noise > 0:
                                    bbox = add_bbox_noise(bbox, noise_range=args.noise, image_size=args.image_size)
                                net.add_new_bbox(state, fid, obj_id, bbox.to(GPUdevice), clear_old_points=False)
                            elif args.prompt == "click":
                                points = pt_dict[fid][obj_id].to(GPUdevice)
                                labels = point_labels_dict[fid][obj_id].to(GPUdevice)
                                net.add_new_points(state, fid, obj_id, points, labels, clear_old_points=False)
                        except KeyError:
                            net.add_new_mask(state, fid, obj_id, torch.zeros(imgs_tensor.shape[2:]).to(GPUdevice))

                segs = {}
                for out_fid, out_ids, out_logits in net.propagate_in_video(state, start_frame_idx=0):
                    segs[out_fid] = {oid: out_logits[i] for i, oid in enumerate(out_ids)}

                wg_preds, cg_preds, pz_preds, wg_gts, cg_gts, pz_gts = [], [], [], [], [], []
                for fid in frame_id:
                    H, W = imgs_tensor.shape[-2:]
                    wg_logit = ensure_shape_4d(segs[fid].get(3, torch.zeros((H, W), device=GPUdevice)))
                    cg_logit = ensure_shape_4d(segs[fid].get(1, torch.zeros((H, W), device=GPUdevice)))
                    wg_pred = torch.sigmoid(wg_logit)
                    cg_pred = torch.sigmoid(cg_logit)
                    pz_pred = torch.relu(wg_pred - cg_pred)

                    cg_gt = ensure_shape_4d(mask_dict[fid].get(1, torch.zeros((H, W))).to(GPUdevice)).float()
                    pz_gt = ensure_shape_4d(mask_dict[fid].get(2, torch.zeros((H, W))).to(GPUdevice)).float()
                    wg_gt = torch.clamp(cg_gt + pz_gt, 0, 1).float()

                    wg_preds.append(wg_pred.unsqueeze(2))
                    cg_preds.append(cg_pred.unsqueeze(2))
                    pz_preds.append(pz_pred.unsqueeze(2))
                    wg_gts.append(wg_gt.unsqueeze(2))
                    cg_gts.append(cg_gt.unsqueeze(2))
                    pz_gts.append(pz_gt.unsqueeze(2))

                wg_vol_pred = torch.cat(wg_preds, dim=2)
                cg_vol_pred = torch.cat(cg_preds, dim=2)
                pz_vol_pred = torch.cat(pz_preds, dim=2)
                wg_vol_gt = torch.cat(wg_gts, dim=2)
                cg_vol_gt = torch.cat(cg_gts, dim=2)
                pz_vol_gt = torch.cat(pz_gts, dim=2)

                # === 指标计算 ===
                wg_dice = dice_3d(wg_vol_pred, wg_vol_gt)
                cg_dice = dice_3d(cg_vol_pred, cg_vol_gt)
                pz_dice = dice_3d(pz_vol_pred, pz_vol_gt)
                wg_iou = iou_3d(wg_vol_pred, wg_vol_gt)
                cg_iou = iou_3d(cg_vol_pred, cg_vol_gt)
                pz_iou = iou_3d(pz_vol_pred, pz_vol_gt)
                wg_hd = hd95_3d(wg_vol_pred, wg_vol_gt)
                cg_hd = hd95_3d(cg_vol_pred, cg_vol_gt)
                pz_hd = hd95_3d(pz_vol_pred, pz_vol_gt)

                results.append({
                    "Case": case_name,
                    "WG_Dice": wg_dice, "CG_Dice": cg_dice, "PZ_Dice": pz_dice,
                    "WG_IoU": wg_iou, "CG_IoU": cg_iou, "PZ_IoU": pz_iou,
                    "WG_HD95": wg_hd, "CG_HD95": cg_hd, "PZ_HD95": pz_hd
                })

                # === 载入 affine ===
                affine, header, ref_shape = np.eye(4), None, wg_vol_pred.squeeze().cpu().numpy().shape
                try:
                    gt_path = os.path.join(args.data_path, testset_name, "mr_image", case_name + ".nii.gz")
                    if not os.path.exists(gt_path):
                        gt_path = os.path.join(args.data_path, "test", "mr_image", case_name + ".nii.gz")
                    if os.path.exists(gt_path):
                        ref_nii = nib.load(gt_path)
                        affine, header, ref_shape = ref_nii.affine, ref_nii.header, ref_nii.shape
                        print(f"[INFO] Using affine from {gt_path}")
                except Exception as e:
                    print(f"[ERROR] Failed to load affine for {case_name}: {e}")

                wg_np = np.transpose((wg_vol_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8), (2, 0, 1))
                cg_np = np.transpose((cg_vol_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8), (2, 0, 1))
                pz_np = np.transpose((pz_vol_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8), (2, 0, 1))

                # === 保存方向对齐的 NIfTI ===
                #nib.save(nib.Nifti1Image(wg_np, affine=affine, header=header),
                         #os.path.join(subset_dir, f"{case_name}_WG.nii.gz"))
                #nib.save(nib.Nifti1Image(cg_np + 2 * pz_np, affine=affine, header=header),
                         #os.path.join(subset_dir, f"{case_name}_CGPZ.nii.gz"))

                # === 每个切片生成4列对比图 ===
                vis_dir = os.path.join(subset_dir, case_name + "_slices")
                os.makedirs(vis_dir, exist_ok=True)
                img_vol = imgs_tensor.cpu().numpy()
                if img_vol.ndim == 4:
                    img_vol = img_vol[:, 0, :, :]  # [D,1,H,W] → [D,H,W]
                elif img_vol.ndim == 3:
                    pass
                else:
                    raise ValueError(f"Unexpected img_vol shape {img_vol.shape}")

                for fid in range(img_vol.shape[0]):
                    wg_gt_2d = wg_vol_gt[0, 0, fid].cpu().numpy()
                    wg_pred_2d = wg_vol_pred[0, 0, fid].cpu().numpy()
                    cg_gt_2d = cg_vol_gt[0, 0, fid].cpu().numpy()
                    pz_gt_2d = pz_vol_gt[0, 0, fid].cpu().numpy()
                    cg_pred_2d = cg_vol_pred[0, 0, fid].cpu().numpy()
                    pz_pred_2d = pz_vol_pred[0, 0, fid].cpu().numpy()

                    if np.sum(wg_gt_2d) == 0 and np.sum(cg_gt_2d) == 0 and np.sum(pz_gt_2d) == 0:
                        continue

                    img_np = img_vol[fid]
                    # === 顺时针旋转90° ===
                    img_np = np.rot90(img_np, k=-1)
                    wg_gt_2d = np.rot90(wg_gt_2d, k=-1)
                    wg_pred_2d = np.rot90(wg_pred_2d, k=-1)
                    cg_gt_2d = np.rot90(cg_gt_2d, k=-1)
                    pz_gt_2d = np.rot90(pz_gt_2d, k=-1)
                    cg_pred_2d = np.rot90(cg_pred_2d, k=-1)
                    pz_pred_2d = np.rot90(pz_pred_2d, k=-1)

                    # === 提升灰度对比度 ===
                    img_norm = (img_np - np.percentile(img_np, 1)) / (np.percentile(img_np, 99) - np.percentile(img_np, 1) + 1e-8)
                    img_norm = np.clip(img_norm, 0, 1)

                    # === Helper: 轻度亮度增强 ===
                    def auto_contrast(img, clip_percent=1.5):
                        lo = np.percentile(img, clip_percent)
                        hi = np.percentile(img, 100 - clip_percent)
                        img_adj = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
                        return img_adj

                    # === Helper: 叠加函数 ===
                    #def overlay(ax, base, mask=None, colors=None, alpha=0.5, title=None):
                        #base_adj = auto_contrast(base, clip_percent=0.5)  # 提升亮度 & 对比度
                        #ax.imshow(base_adj, cmap="gray")
                        #if mask is not None:
                            #for m, c in zip(mask, colors):
                                #if np.any(m):
                                    #colored = np.zeros((*m.shape, 4))
                                    #colored[..., :3] = plt.cm.get_cmap(c)(1.0)[:3]
                                    #colored[..., 3] = (m > 0).astype(float) * alpha
                                    #ax.imshow(colored)
                        #ax.set_title(title)
                        #ax.axis("off")
                    def overlay(ax, base, mask=None, colors=None, alpha=0.4, title=None):
                            """
                            显示灰度原图 + 高亮半透明mask叠加
                            colors: e.g. ["#0077ff", "#00ff00", "#ff0000"]
                            """
                            base_adj = auto_contrast(base, clip_percent=0.3)
                            ax.imshow(base_adj, cmap="gray")

                            if mask is not None:
                                for m, c in zip(mask, colors):
                                    if np.any(m):
                                        # 自定义高亮色 (避免matplotlib colormap发灰)
                                        color_rgb = np.array(matplotlib.colors.to_rgb(c))
                                        rgba = np.zeros((*m.shape, 4))
                                        rgba[..., :3] = color_rgb
                                        rgba[..., 3] = (m > 0).astype(float) * alpha
                                        ax.imshow(rgba)
                            ax.set_title(title)
                            ax.axis("off")    

                    # === 绘制四列对比图 ===
                    #fig, axs = plt.subplots(1, 4, figsize=(16, 4))
                    #fig, axs = plt.subplots(1, 5, figsize=(20, 4))
                    #overlay(axs[0], img_np, title="Raw")  # 不传 mask，就是纯灰度原图
                    #overlay(axs[1], img_np, [wg_gt_2d], ["#0000ff"], 0.5, "GT WG")  # 蓝色
                    #overlay(axs[2], img_np, [wg_pred_2d > 0.5], ["#0000ff"], 0.5, "Pred WG")
                    #overlay(axs[3], img_np, [cg_gt_2d, pz_gt_2d], ["#ff0000", "#00ff00"], 0.5, "GT CG+PZ")  # 红+绿
                    #overlay(axs[4], img_np, [cg_pred_2d > 0.5, pz_pred_2d > 0.5], ["#ff0000", "#00ff00"], 0.5, "Pred CG+PZ")
                    
                    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
                    axs = axs.flatten()  # ✅ 把二维转为一维，方便 axs[0]、axs[1] 这样索引

                    overlay(axs[0], img_np, title="Raw")  # 不传 mask，就是纯灰度原图
                    overlay(axs[1], img_np, [wg_gt_2d], ["#0000ff"], 0.5, "GT WG")  # 蓝色
                    overlay(axs[2], img_np, [wg_pred_2d > 0.5], ["#0000ff"], 0.5, "Pred WG")
                    overlay(axs[3], img_np, [cg_gt_2d], ["#ff0000"], 0.5, "GT CG")  # 红
                    overlay(axs[4], img_np, [cg_pred_2d > 0.5], ["#ff0000"], 0.5, "Pred CG")
                    overlay(axs[5], img_np, [pz_gt_2d], ["#00ff00"], 0.5, "GT PZ")  # 绿色
                    overlay(axs[6], img_np, [pz_pred_2d > 0.5], ["#00ff00"], 0.5, "Pred PZ") 
                    overlay(axs[7], img_np, [cg_gt_2d, pz_gt_2d], ["#ff0000", "#00ff00"], 0.5, "GT CG+PZ")     
                    overlay(axs[8], img_np, [cg_pred_2d > 0.5, pz_pred_2d > 0.5], ["#ff0000", "#00ff00"], 0.5, "Pred CG+PZ")   
                    #overlay(axs[0], img_np, [wg_gt_2d], ["Blues"], 0.6, "GT WG")
                    #overlay(axs[1], img_np, [wg_pred_2d > 0.5], ["Blues"], 0.6, "Pred WG")
                    #overlay(axs[2], img_np, [cg_gt_2d, pz_gt_2d], ["Reds", "Greens"], 0.6, "GT CG+PZ")
                    #overlay(axs[3], img_np, [cg_pred_2d > 0.5, pz_pred_2d > 0.5], ["Reds", "Greens"], 0.6, "Pred CG+PZ")

                    plt.tight_layout()
                    plt.savefig(os.path.join(vis_dir, f"slice_{fid:03d}.png"), dpi=150, bbox_inches="tight")
                    plt.close()



            net.reset_state(state)
            pbar.update()

    # === 保存 CSV 并添加平均值行 ===
    df = pd.DataFrame(results)
    if len(df) > 0:
        mean_row = {"Case": "Mean"}
        for col in df.columns[1:]:
            mean_row[col] = np.nanmean(df[col].astype(float))
        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    csv_path = os.path.join(save_root, f"test_metrics_{testset_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Subset {testset_name} results saved to: {csv_path}")
    print(df.tail(1))
    return df
