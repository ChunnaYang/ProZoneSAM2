"""Utility functions for training and evaluation.
    Yunli Qi
"""

import logging
import os
import random
import sys
import time
from datetime import datetime

import dateutil.tz
import numpy as np
import torch
from torch.autograd import Function
from scipy.spatial.distance import directed_hausdorff
from medpy.metric.binary import hd95
import cfg

args = cfg.parse_args()
device = torch.device('cuda', args.gpu_device)

def get_network(args, net, use_gpu=True, gpu_device = 0, distribution = True):
    """ return given network
    """

    if net == 'sam2':
        from sam2_train.build_sam import build_sam2_video_predictor

        sam2_checkpoint = args.sam_ckpt
        model_cfg = args.sam_config

        net = build_sam2_video_predictor(config_file=model_cfg, ckpt_path=sam2_checkpoint, mode=None)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if use_gpu:
        net = net.to(device=gpu_device)

    return net

def create_logger(log_dir, phase='train'):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger

def set_log_dir(root_dir, exp_name):
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)

    # set log path
    exp_path = os.path.join(root_dir, exp_name)
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    prefix = exp_path + '_' + timestamp
    os.makedirs(prefix)
    path_dict['prefix'] = prefix

    # set checkpoint path
    ckpt_path = os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path)
    path_dict['ckpt_path'] = ckpt_path

    log_path = os.path.join(prefix, 'Log')
    os.makedirs(log_path)
    path_dict['log_path'] = log_path

    # set sample image path for fid calculation
    sample_path = os.path.join(prefix, 'Samples')
    os.makedirs(sample_path)
    path_dict['sample_path'] = sample_path

    return path_dict

def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))

def random_click(mask, point_labels = 1, seed=None):
    # check if all masks are black
    max_label = max(set(mask.flatten()))
    if max_label == 0:
        point_labels = max_label
    indices = np.argwhere(mask == max_label) 
    if seed is not None:
        rand_instance = random.Random(seed)
        rand_num = rand_instance.randint(0, len(indices) - 1)
    else:
        rand_num = random.randint(0, len(indices) - 1)
    output_index_1 = indices[rand_num][0]
    output_index_0 = indices[rand_num][1]
    return point_labels, np.array([output_index_0, output_index_1])

def random_perturb_bbox(bbox_gt, img, shift=5):
    H, W = img.shape
    bbox = bbox_gt.clone()
    dx_min = random.randint(-shift, shift)
    dy_min = random.randint(-shift, shift)
    dx_max = random.randint(-shift, shift)
    dy_max = random.randint(-shift, shift)
    bbox[1] += dx_min  
    bbox[0] += dy_min  
    bbox[3] += dx_max  
    bbox[2] += dy_max  
    bbox[1] = min(bbox[1], bbox[3] - 1e-2)
    bbox[0] = min(bbox[0], bbox[2] - 1e-2)
    bbox[3] = max(bbox[3], bbox[1] + 1e-2)
    bbox[2] = max(bbox[2], bbox[0] + 1e-2)
    bbox[1] = max(0.0, min(bbox[1], W - 1e-2))
    bbox[0] = max(0.0, min(bbox[0], H - 1e-2))
    bbox[3] = max(0.0, min(bbox[3], W - 1e-2))
    bbox[2] = max(0.0, min(bbox[2], H - 1e-2))
    return bbox

def generate_bbox(mask, variation=0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    if len(mask.shape) != 2:
        current_shape = mask.shape
        raise ValueError(f"Mask shape is not 2D, but {current_shape}")
    max_label = max(set(mask.flatten()))
    if max_label == 0:
        return np.array([np.nan, np.nan, np.nan, np.nan])
    indices = np.argwhere(mask == max_label) 
    x0 = np.min(indices[:, 0]); x1 = np.max(indices[:, 0])
    y0 = np.min(indices[:, 1]); y1 = np.max(indices[:, 1])
    w = x1 - x0; h = y1 - y0
    mid_x = (x0 + x1) / 2; mid_y = (y0 + y1) / 2
    if variation > 0:
        num_rand = np.random.randn(2) * variation
        w *= 1 + num_rand[0]; h *= 1 + num_rand[1]
        x1 = mid_x + w / 2; x0 = mid_x - w / 2
        y1 = mid_y + h / 2; y0 = mid_y - h / 2
    return np.array([y0, x0, y1, x1])

# ======================= 修复 TRE NaN 的类 =======================
class MaskMetrics:
    def __init__(self):
        pass

    def target_registration_error(self, mask1, mask2):
        """计算 TRE（质心距离），如果无前景则返回 0"""
        if mask1.sum() == 0 or mask2.sum() == 0:
            return 0.0
        centroid1 = np.mean(np.argwhere(mask1), axis=0)
        centroid2 = np.mean(np.argwhere(mask2), axis=0)
        return np.linalg.norm(centroid1 - centroid2)

    def dice_coefficient(self, mask1, mask2):
        """计算 Dice 系数"""
        intersection = np.logical_and(mask1, mask2).sum()
        denominator = mask1.sum() + mask2.sum()
        if denominator == 0:
            return 1.0
        return 2. * intersection / denominator
    # ===============================================================
    # 🧮 计算 3D Dice 系数（体素级）
    # ===============================================================
    

    def __call__(self, mask1, mask2):
        return {
            "tre": self.target_registration_error(mask1, mask2),
            "dice": self.dice_coefficient(mask1, mask2)
        }
# ===============================================================
# 🧮 计算 3D Dice 系数（体素级）
# ===============================================================
def dice_3d(pred_volume, gt_volume, smooth=1e-6):
    """
    计算 3D Dice 系数 (voxel-level)
    参数:
        pred_volume: np.ndarray 或 torch.Tensor, 形状 (D, H, W)
        gt_volume:   np.ndarray 或 torch.Tensor, 形状 (D, H, W)
    返回:
        float: Dice 系数
    """
    import numpy as np
    import torch

    # 转换为 numpy 数组
    if isinstance(pred_volume, torch.Tensor):
        pred_volume = pred_volume.detach().cpu().numpy()
    if isinstance(gt_volume, torch.Tensor):
        gt_volume = gt_volume.detach().cpu().numpy()

    # 二值化
    pred_bin = (pred_volume > 0.5).astype(np.uint8)
    gt_bin = (gt_volume > 0.5).astype(np.uint8)

    # 计算交并
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    total = pred_bin.sum() + gt_bin.sum()
    if total == 0:
        return 1.0  # 全空返回1.0
    dice = (2.0 * intersection + smooth) / (total + smooth)
    return float(dice)

# ===============================================================
def iou_3d(pred_volume, gt_volume, smooth=1e-6):
    """计算 3D IoU"""
    if isinstance(pred_volume, torch.Tensor):
        pred_volume = pred_volume.detach().cpu().numpy()
    if isinstance(gt_volume, torch.Tensor):
        gt_volume = gt_volume.detach().cpu().numpy()
    pred_bin = (pred_volume > 0.5).astype(np.uint8)
    gt_bin = (gt_volume > 0.5).astype(np.uint8)
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    if union == 0:
        return 1.0
    return (inter + smooth) / (union + smooth)

def hd95_3d(pred_volume, gt_volume):
    """计算 3D Hausdorff Distance (95th percentile)"""
    if isinstance(pred_volume, torch.Tensor):
        pred_volume = pred_volume.detach().cpu().numpy()
    if isinstance(gt_volume, torch.Tensor):
        gt_volume = gt_volume.detach().cpu().numpy()

    pred_bin = (pred_volume > 0.5).astype(np.uint8)
    gt_bin = (gt_volume > 0.5).astype(np.uint8)

    if np.sum(pred_bin) == 0 or np.sum(gt_bin) == 0:
        return np.nan  # 空体积跳过

    try:
        return float(hd95(pred_bin, gt_bin))
    except Exception:
        return np.nan
##=============================================================
def eval_seg(pred,true_mask_p,threshold):
    b, c, h, w = pred.size()
    if c == 2:
        iou_d, iou_c, disc_dice, cup_dice = 0,0,0,0
        for th in threshold:
            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')
            cup_pred = vpred_cpu[:,1,:,:].numpy().astype('int32')
            disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
            cup_mask = gt_vmask_p [:, 1, :, :].squeeze(1).cpu().numpy().astype('int32')
            iou_d += iou(disc_pred,disc_mask)
            iou_c += iou(cup_pred,cup_mask)
            disc_dice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
            cup_dice += dice_coeff(vpred[:,1,:,:], gt_vmask_p[:,1,:,:]).item()
        return iou_d / len(threshold), iou_c / len(threshold), disc_dice / len(threshold), cup_dice / len(threshold)
    elif c > 2: 
        ious = [0] * c; dices = [0] * c
        for th in threshold:
            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            for i in range(0, c):
                pred = vpred_cpu[:,i,:,:].numpy().astype('int32')
                mask = gt_vmask_p[:,i,:,:].squeeze(1).cpu().numpy().astype('int32')
                ious[i] += iou(pred,mask)
                dices[i] += dice_coeff(vpred[:,i,:,:], gt_vmask_p[:,i,:,:]).item()
        return tuple(np.array(ious + dices) / len(threshold)) 
    else:
        eiou, edice = 0,0
        for th in threshold:
            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')
            disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
            eiou += iou(disc_pred,disc_mask)
            edice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
        return eiou / len(threshold), edice / len(threshold)

def iou(outputs: np.array, labels: np.array):
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    return ((intersection + SMOOTH) / (union + SMOOTH)).mean()

def dice_coeff(input, target):
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device = input.device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()
    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])
    return s / (i + 1)

class DiceCoeff(Function):
    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps
        return (2 * self.inter.float() + eps) / self.union.float()

    def backward(self, grad_output):
        input, target = self.saved_variables
        grad_input = grad_target = None
        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None
        return grad_input, grad_target
    
# ✅ 新增组合损失函数（Dice + BCE）
def dice_bce_loss(pred, gt, bce_func):
    """
    pred: raw logits (B,H,W) or (1,H,W)
    gt:   binary ground truth (same size)
    bce_func: 已定义的 BCEWithLogitsLoss
    """
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
    bce = bce_func(pred, gt)
    dice_val = dice_coeff(torch.sigmoid(pred), gt).item()
    dice = 1 - dice_val
    return bce + dice