#!/usr/bin/env python3
""" Train network using PyTorch
    Modified for WG/CG/PZ segmentation
    Author: YMT
"""

import os
import time
import datetime
import torch
import torch.optim as optim
import swanlab
import cfg
from func_3d import function
from conf import settings
from func_3d.utils import get_network
from func_3d.dataset import get_dataloader
import shutil


def main():
    args = cfg.parse_args()
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'work_dir/{args.sam_config}_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy('cfg.py', os.path.join(save_dir, f'cfg_{timestamp}.py'))

        # ==================== 初始化网络 ====================
    GPUdevice = torch.device('cuda', args.gpu_device)
    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)

    if torch.cuda.device_count() > 1:
        print(f"⚡ Using {torch.cuda.device_count()} GPUs (DataParallel)")
        net = torch.nn.DataParallel(net)

    # ====================================================
    # 🧩 参数统计 + 打印 & 写入 SwanLab
    # ====================================================
    model = net.module if hasattr(net, 'module') else net
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_ratio = trainable_params / total_params * 100
    param_memory_mb = trainable_params * 4 / 1024 / 1024  # 假设float32, 每参数4字节

    print("\n==================== 🧩 Trainable Parameter Summary ====================")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio:      {trainable_ratio:.4f}%")
    print(f"Approx. trainable param memory: {param_memory_mb:.2f} MB")
    print("--------------------------------------------------------------------------")
    print(f"{'Layer Name':60s} | {'Train?':^7s} | {'Param #':>12s}")
    print("-" * 85)
    for name, param in model.named_parameters():
        print(f"{name:60s} | {'✅' if param.requires_grad else '❌':^7s} | {param.numel():12,}")
    print("=" * 85 + "\n")

    # ====================================================
    # 💾 把参数信息写入 SwanLab config
    # ====================================================
    param_info = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio(%)": round(trainable_ratio, 4),
        "trainable_memory(MB)": round(param_memory_mb, 2),
    }

    # 先加载数据前写入
    # （此时 SwanLab 还未 init）
    # 我们将在 SwanLab init 处加入 param_info

    # 优化器
    optimizer1 = optim.Adam(list(model.sam_mask_decoder.parameters()), lr=1e-4, betas=(0.9, 0.999))
    optimizer2 = optim.Adam(list(model.memory_encoder.parameters()), lr=1e-8, betas=(0.9, 0.999))
    # =========================================================
    # 🔍 检查优化器实际包含的参数数量
    # =========================================================
    opt1_params = sum(p.numel() for p in optimizer1.param_groups[0]['params'])
    opt2_params = sum(p.numel() for p in optimizer2.param_groups[0]['params'])
    total_trainable = opt1_params + opt2_params

    print("\n==================== ⚙️ Optimizer Parameter Summary ====================")
    print(f"Optimizer1 (sam_mask_decoder): {opt1_params:,} params, lr=1e-4")
    print(f"Optimizer2 (memory_encoder):   {opt2_params:,} params, lr=1e-8")
    print(f"Total parameters being updated: {total_trainable:,}")
    print("========================================================================\n")
    
    # 同步写入 SwanLab config
    swanlab.init(
        project="MedSAM2_WG_CG_PZ",
        name=f"exp_{timestamp}",
        config={
            "epochs": settings.EPOCH,
            "lr_decoder": 1e-4,
            "lr_memory": 1e-8,
            "architecture": "sam2",
            "decoder_params": opt1_params,
            "memory_params": opt2_params,
            "total_trainable_params": total_trainable,
        }
    )

    # 加载数据
    nice_train_loader, nice_val_loader = get_dataloader(args)

    # Swanlab 初始化
    swanlab.init(
        project="MedSAM2_WG_CG_PZ",
        name=f"exp_{timestamp}",
        config={"epochs": settings.EPOCH, "lr": 1e-4, "architecture": "sam2"}
    )

    best_dice = 0.0

    # ==================== 训练循环 ====================
    for epoch in range(settings.EPOCH):
        # ---- Train ----
        net.train()
        time_start = time.time()
        train_loss, _, _ = function.train_sam(args, net, optimizer1, optimizer2, nice_train_loader, epoch)
        time_end = time.time()
        print(f"Epoch {epoch:03d} | Train Loss={train_loss:.4f} | Time={time_end - time_start:.2f}s")

        # ---- Validation ----
        if epoch % args.val_freq == 0 or epoch == settings.EPOCH - 1:
            net.eval()
            val_dice, (wg_2d, cg_2d, pz_2d), region_metrics = function.validation_sam(args, nice_val_loader, epoch, net)

            # 如果验证函数中未计算3D dice，则 region_metrics 为空，此处可按需添加3D dice汇总逻辑
            # 假设 region_metrics 包含：
            # region_metrics = {"WG_3D": val_wg_3d, "CG_3D": val_cg_3d, "PZ_3D": val_pz_3d}

            mean_dice_2d = (wg_2d + cg_2d + pz_2d) / 3.0
            wg_3d = region_metrics.get("WG_3D", 0.0)
            cg_3d = region_metrics.get("CG_3D", 0.0)
            pz_3d = region_metrics.get("PZ_3D", 0.0)
            mean_dice_3d = (wg_3d + cg_3d + pz_3d) / 3.0 if wg_3d + cg_3d + pz_3d > 0 else 0.0

            #print("------------------------------------------------------")
            #print(f"VAL Epoch {epoch:03d} Results:")
            #print(f"   2D Dice => WG={wg_2d:.4f}, CG={cg_2d:.4f}, PZ={pz_2d:.4f}, Mean={mean_dice_2d:.4f}")
            if mean_dice_3d > 0:
                print(f"   3D Dice => WG={wg_3d:.4f}, CG={cg_3d:.4f}, PZ={pz_3d:.4f}, Mean={mean_dice_3d:.4f}")
            print("------------------------------------------------------")

            # 保存 best model
            # ===========================================================
            # 🌟 保存两类最优模型：Mean_3D 和 PZ_3D
            # ===========================================================
            # 初始化两个最优指标
            if epoch == 0:
                best_mean3d_dice = 0.0
                best_pz3d_dice = 0.0

            # ---- 更新平均3D最优 ----
            if mean_dice_3d > locals().get('best_mean3d_dice', 0):
                best_mean3d_dice = mean_dice_3d
                torch.save({
                    'model': net.state_dict(),  # 保存 state_dict 更轻量
                    'best_mean3d_dice': best_mean3d_dice,
                    'epoch': epoch
                }, os.path.join(save_dir, 'best_mean3d_model.pth'))
                print(f"✅ Best Mean3D model updated @ epoch {epoch} | Mean 3D Dice={best_mean3d_dice:.4f}")

            # ---- 更新 PZ_3D 最优 ----
            if pz_3d > locals().get('best_pz3d_dice', 0):
                best_pz3d_dice = pz_3d
                torch.save({
                    'model': net.state_dict(),
                    'best_pz3d_dice': best_pz3d_dice,
                    'epoch': epoch
                }, os.path.join(save_dir, 'best_PZ3D_model.pth'))
                print(f"✅ Best PZ3D model updated @ epoch {epoch} | PZ 3D Dice={best_pz3d_dice:.4f}")

            #if val_dice > best_dice:
                #best_dice = val_dice
                #torch.save({'model': net, 'best_dice': best_dice, 'epoch': epoch},
                           #os.path.join(save_dir, 'best_model.pth'))
                #print(f"✅ Best model updated @ epoch {epoch} | Mean Dice={best_dice:.4f}")

            # ---- 日志记录 ----
            swanlab.log({
                "Train_Loss": train_loss,
                #"Mean_2D_Dice": mean_dice_2d,
                #"WG_2D_Dice": wg_2d,
                #"CG_2D_Dice": cg_2d,
                #"PZ_2D_Dice": pz_2d,
                "Mean_3D_Dice": mean_dice_3d,
                "WG_3D_Dice": wg_3d,
                "CG_3D_Dice": cg_3d,
                "PZ_3D_Dice": pz_3d,
            }, step=epoch)

        # ---- 保存当前模型 ----
        torch.save({'model': net, 'best_dice': best_dice, 'epoch': epoch},
                   os.path.join(save_dir, 'latest_epoch.pth'))


if __name__ == '__main__':
    main()
