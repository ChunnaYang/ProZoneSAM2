import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import nibabel as nib
from func_3d.utils import generate_bbox, random_perturb_bbox
from mmdet.apis import DetInferencer
from mmengine.registry import init_default_scope

# ======================================================
# 🔹 PROSTATE Dataset with dual bbox sources (GT or DINO)
# ======================================================
class PROSTATE(Dataset):
    """
    前列腺 MR 图像数据集
    支持两种检测框来源：
    - det_source='gt': 使用 GT 掩码生成检测框（监督验证用）
    - det_source='dino': 使用 Grounding DINO 自动检测框（推理用）
    """
    def __init__(self,
                 args,
                 data_path,
                 transform=None,
                 transform_msk=None,
                 mode='train',
                 prompt='bbox',
                 seed=None,
                 variation=0,
                 det_source='gt',             # ✅ 新增：检测框来源选择
                 dino_cfg=None,
                 dino_ckpt=None,
                 device='cuda'):

        self.name_list = os.listdir(os.path.join(data_path, mode, 'mr_image'))
        if mode == 'train':
            self.name_list = self.name_list[:args.data_num]

        # 基础配置
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.seed = seed
        self.variation = variation
        self.noise = args.noise
        self.det_source = det_source
        self.device = device

        # Grounding DINO 模式下加载检测器
        if det_source == 'dino':
            init_default_scope('mmdet')
            self.detector = DetInferencer(
                model=dino_cfg,
                weights=dino_ckpt,
                device=device,
                show_progress=False
            )
            print(f"✅ 已加载 Grounding DINO 检测模型: {dino_ckpt}")
        else:
            self.detector = None

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, self.mode, 'mr_image', name)
        mask_path = os.path.join(self.data_path, self.mode, 'mr_onelabel', name)

        # 读取图像和掩码
        mask = nib.load(mask_path).get_fdata()
        image = nib.load(img_path).get_fdata()

        # 提取非空切片
        nonzero_slices = [i for i in range(mask.shape[2]) if np.any(mask[:, :, i] != 0)]
        start = nonzero_slices[0] if nonzero_slices else 0
        data_seg_3d = mask[:, :, nonzero_slices] if nonzero_slices else np.zeros((*mask.shape[:2], 0))
        num_frame = data_seg_3d.shape[2]

        # 图像 resize
        imgs = torch.zeros(num_frame, 3, self.img_size, self.img_size)
        for i in range(num_frame):
            img = image[:, :, nonzero_slices[i]]
            img = (255.0 * (img - img.min()) / (img.max() - img.min() + 1e-8)).astype(np.uint8)
            img = Image.fromarray(img).convert('RGB').resize((self.img_size, self.img_size))
            imgs[i] = torch.tensor(np.array(img)).permute(2, 0, 1)

        # ==================================================
        # ✅ 检测框生成逻辑
        # ==================================================
        bbox_dict = {}
        if self.det_source == 'gt':   # 使用 GT mask 生成 bbox
            for i in range(num_frame):
                mask_slice = data_seg_3d[:, :, i]
                bbox_dict[i] = {}
                for obj_id in np.unique(mask_slice):
                    if obj_id == 0:
                        continue
                    ys, xs = np.where(mask_slice == obj_id)
                    if len(xs) > 0 and len(ys) > 0:
                        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                        bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
                        bbox_dict[i][int(obj_id)] = bbox
            bbox_source = "GT"
        else:   # 使用 Grounding DINO 检测结果
            bbox_dict = {}
            for i in range(num_frame):
                slice_img = imgs[i].permute(1, 2, 0).numpy().astype(np.uint8)
                bbox_dict[i] = {}
                for region, text_prompt in {"CG": "CG", "PZ": "PZ"}.items():
                    result = self.detector({'img': slice_img, 'text': text_prompt})
                    if len(result['predictions'][0]['bboxes']) == 0:
                        continue
                    bbox = np.array(result['predictions'][0]['bboxes'][0], dtype=np.float32)
                    ann_obj_id = 1 if region == "CG" else 2
                    bbox_dict[i][ann_obj_id] = bbox
            bbox_source = "DINO"

        # ==================================================
        image_meta_dict = {'filename_or_obj': name}

        return {
            'image': imgs,
            'mask': mask,
            'bbox': bbox_dict,
            'path': img_path,
            'name': name,
            'bbox_source': bbox_source,
            'image_meta_dict': image_meta_dict
        }


# ======================================================
# 🔹 独立 Grounding DINO 加载接口（供推理脚本直接调用）
# ======================================================
def load_detection_model(model_cfg, ckpt_path, device='cuda'):
    init_default_scope('mmdet')
    detector = DetInferencer(
        model=model_cfg,
        weights=ckpt_path,
        device=device,
        show_progress=False
    )
    print(f"✅ 已加载 Grounding DINO 模型: {ckpt_path}")
    return detector