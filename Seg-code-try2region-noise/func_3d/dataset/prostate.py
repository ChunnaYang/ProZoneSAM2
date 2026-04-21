""" Dataloader for PROSTATE dataset (WG, CG, PZ)
    Modified by YMT
"""
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import nibabel as nib
from func_3d.utils import random_click, generate_bbox, random_perturb_bbox

class PROSTATE(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None,
                 mode='train', prompt='bbox', seed=None, variation=0):

        self.name_list = os.listdir(os.path.join(data_path, mode, 'mr_image'))
        if mode == 'train':
            self.name_list = self.name_list[:args.data_num]

        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.seed = seed
        self.variation = variation
        self.noise = args.noise
        self.video_length = args.video_length if mode == 'train' else None

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, self.mode, 'mr_image', name)
        mask_path = os.path.join(self.data_path, self.mode, 'mr_onelabel', name)

        img_nii = nib.load(img_path)
        mask_nii = nib.load(mask_path)
        image = img_nii.get_fdata()
        mask = mask_nii.get_fdata()

        newsize = (self.img_size, self.img_size)
        nonzero_slices = [i for i in range(mask.shape[2]) if np.any(mask[:, :, i] != 0)]
        if len(nonzero_slices) == 0:
            nonzero_slices = [0]
        num_frame = len(nonzero_slices)
        video_length = self.video_length or num_frame

        img_tensor = torch.zeros(video_length, 3, self.img_size, self.img_size)
        mask_dict, bbox_dict = {}, {}

        for i in range(video_length):
            slice_idx = nonzero_slices[i % num_frame]
            img = image[:, :, slice_idx]
            img = (255 * (img - img.min()) / (img.max() - img.min() + 1e-8)).astype(np.uint8)
            img = Image.fromarray(img).convert('RGB').resize(newsize)
            img_tensor[i] = torch.tensor(np.array(img)).permute(2, 0, 1)

            mask_slice = mask[:, :, slice_idx]
            obj_mask_dict, obj_bbox_dict = {}, {}

            # CG=1, PZ=2, WG=(1 or 2)=3
            for obj in [1, 2]:
                if np.sum(mask_slice == obj) > 0:
                    obj_mask = Image.fromarray((mask_slice == obj).astype(np.uint8)).resize(newsize)
                    obj_mask = torch.tensor(np.array(obj_mask)).unsqueeze(0).int()
                    obj_mask_dict[obj] = obj_mask
                    obj_bbox_dict[obj] = generate_bbox(np.array(obj_mask.squeeze(0)),
                                                       variation=self.variation, seed=self.seed)
                    if self.noise != 0:
                        obj_bbox_dict[obj] = random_perturb_bbox(
                            torch.tensor(obj_bbox_dict[obj]), obj_mask.squeeze(0), self.noise)

            wg_mask = ((mask_slice == 1) | (mask_slice == 2)).astype(np.uint8)
            wg_mask = Image.fromarray(wg_mask).resize(newsize)
            wg_mask = torch.tensor(np.array(wg_mask)).unsqueeze(0).int()
            obj_mask_dict[3] = wg_mask
            obj_bbox_dict[3] = generate_bbox(np.array(wg_mask.squeeze(0)),
                                             variation=self.variation, seed=self.seed)

            mask_dict[i] = obj_mask_dict
            bbox_dict[i] = obj_bbox_dict

        return {'image': img_tensor, 'label': mask_dict, 'bbox': bbox_dict,
                'image_meta_dict': {'filename_or_obj': name}}
