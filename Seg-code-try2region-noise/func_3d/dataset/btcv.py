""" Dataloader for the BTCV dataset
    Yunli Qi
"""
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from func_3d.utils import random_click, generate_bbox


class BTCV(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', seed=None, variation=0):

        # Set the data list for training
        self.name_list = os.listdir(os.path.join(data_path, mode, 'image')) # 视频数（3D数据个数）image0001 image0002 ...
        
        # Set the basic information of the dataset
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self.seed = seed
        self.variation = variation
        if mode == 'Training':
            self.video_length = args.video_length
        else:
            self.video_length = None

    def __len__(self):
        return len(self.name_list)   # 24

    def __getitem__(self, index):
        point_label = 1
        newsize = (self.img_size, self.img_size)

        """Get the images"""
        name = self.name_list[index]  # img0001
        img_path = os.path.join(self.data_path, self.mode, 'image', name)  # 读取一个3D数据（视频）
        mask_path = os.path.join(self.data_path, self.mode, 'mask', name)
        
        data_seg_3d_shape = np.load(mask_path + '/0.npy').shape  # label.shape (512,512) (128,128)
        num_frame = len(os.listdir(mask_path)) #.npy文件个数 记录mask信息 120
        data_seg_3d = np.zeros(data_seg_3d_shape + (num_frame,)) # (512,512,88) (128,128,120)
        for i in range(num_frame):
            data_seg_3d[..., i] = np.load(os.path.join(mask_path, f'{i}.npy')) #读取完整的3D mask
        for i in range(data_seg_3d.shape[-1]):   #用for循环用于找到第一个包含非零元素的帧从88维度方向
            if np.sum(data_seg_3d[..., i]) > 0:
                data_seg_3d = data_seg_3d[..., i:]
                break
        starting_frame_nonzero = i  # 第一个包含非零元素的帧
        for j in reversed(range(data_seg_3d.shape[-1])):
            if np.sum(data_seg_3d[..., j]) > 0:
                data_seg_3d = data_seg_3d[..., :j+1]
                break
        num_frame = data_seg_3d.shape[-1]  # 经过上述裁剪后，数据中剩余的帧数。
        if self.video_length is None: # 确定视频长度
            video_length = int(num_frame / 4)
        else:
            video_length = self.video_length
        if num_frame > video_length and self.mode == 'Training': 
        # 如果剩余的帧数大于指定的视频长度，并且当前模式是 Training（训练模式），则从所有可用帧中随机选择一个起始帧。这确保了训练时数据的多样性。
        # 如果剩余帧数不超过视频长度，或者当前模式不是 Training，则从第一帧（索引 0）开始。
            starting_frame = np.random.randint(0, num_frame - video_length + 1)
        else:
            starting_frame = 0
        img_tensor = torch.zeros(video_length, 3, self.img_size, self.img_size) # 存储图像数据
        mask_dict = {}
        point_label_dict = {}
        pt_dict = {}
        bbox_dict = {}
        
        # 开始处理视频段
        for frame_index in range(starting_frame, starting_frame + video_length):
            img = Image.open(os.path.join(img_path, f'{frame_index + starting_frame_nonzero}.jpg')).convert('RGB')
            mask = data_seg_3d[..., frame_index]
            # mask = np.rot90(mask)
            obj_list = np.unique(mask[mask > 0]) # obj_list为mask中的label个数
            diff_obj_mask_dict = {}
            if self.prompt == 'bbox':
                diff_obj_bbox_dict = {}
            elif self.prompt == 'click':
                diff_obj_pt_dict = {}
                diff_obj_point_label_dict = {}
            else:
                raise ValueError('Prompt not recognized')
            for obj in obj_list:
                obj_mask = mask == obj
                # if self.transform_msk:
                obj_mask = Image.fromarray(obj_mask)
                obj_mask = obj_mask.resize(newsize)
                obj_mask = torch.tensor(np.array(obj_mask)).unsqueeze(0).int()
                    # obj_mask = self.transform_msk(obj_mask).int()
                diff_obj_mask_dict[obj] = obj_mask

                if self.prompt == 'click':
                    diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = random_click(np.array(obj_mask.squeeze(0)), point_label, seed=None)
                if self.prompt == 'bbox':
                    diff_obj_bbox_dict[obj] = generate_bbox(np.array(obj_mask.squeeze(0)), variation=self.variation, seed=self.seed)
            # if self.transform:
                # state = torch.get_rng_state()
                # img = self.transform(img)
                # torch.set_rng_state(state)
            img = img.resize(newsize)
            img = torch.tensor(np.array(img)).permute(2, 0, 1)
            # print(img.shape) #torch.Size([3,1024,1024])
            # print(img_tensor[frame_index - starting_frame, :, :, :].shape) #torch.Size([3,1024,1024])
            img_tensor[frame_index - starting_frame, :, :, :] = img
            mask_dict[frame_index - starting_frame] = diff_obj_mask_dict
            if self.prompt == 'bbox':
                bbox_dict[frame_index - starting_frame] = diff_obj_bbox_dict
            elif self.prompt == 'click':
                pt_dict[frame_index - starting_frame] = diff_obj_pt_dict
                point_label_dict[frame_index - starting_frame] = diff_obj_point_label_dict


        image_meta_dict = {'filename_or_obj':name}
        if self.prompt == 'bbox':
            return {
                'image':img_tensor,
                'label': mask_dict,
                'bbox': bbox_dict,
                'image_meta_dict':image_meta_dict,
            }
        elif self.prompt == 'click':
            return {
                'image':img_tensor,
                'label': mask_dict,
                'p_label':point_label_dict,
                'pt':pt_dict,
                'image_meta_dict':image_meta_dict,
            }