""" Dataloader for the BTCV dataset
    Yunli Qi
"""
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import nibabel as nib
from func_3d.utils import random_click, generate_bbox
from scipy.ndimage import zoom

class PROSTATE(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', seed=None, variation=0):

        # Set the data list for training
        self.name_list = os.listdir(os.path.join(data_path, mode, 'mr_image')) # 视频数（3D数据个数）image0001 image0002 ...
        
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
        img_path = os.path.join(self.data_path, self.mode, 'mr_image', name)  # 读取一个3D数据（视频）
        mask_path = os.path.join(self.data_path, self.mode, 'mr_onelabel', name)
        num_frame = nib.load(mask_path).get_fdata().shape[0]
        data_seg_3d = np.zeros(nib.load(mask_path).get_fdata().shape) # (512,512,88)

        # 加载 .nii.gz 文件并获取数据
        mask = nib.load(mask_path).get_fdata()
        print(mask.shape)
        data_seg_3d = mask
        image = nib.load(img_path).get_fdata()[:,:,:,0]

        # 找到第一个非零的第一维切片
        for i in range(mask.shape[0]):  # mask.shape[0] 是第一维的大小
            if np.sum(mask[i,...]) > 0:  # 检查第 i 个切片是否包含非零元素
                data_seg_3d = mask[i:,...]  # 从第 i 个切片开始截取
                break
        starting_frame_nonzero = i  # 记录第一个包含非零元素的切片索引
        # 找到最后一个非零的第一维切片
        for j in reversed(range(mask.shape[0])):  # 从最后一维开始反向查找
            if np.sum(mask[j,...]) > 0:  # 检查第 j 个切片是否包含非零元素
                data_seg_3d = mask[:j+1,...]  # 截取到第 j 个切片为止
                break

        # 更新剩余的切片数（沿着第一维）
        num_frame = data_seg_3d.shape[0]

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

        for frame_index in range(starting_frame, starting_frame + video_length):
            print('----------------------------------')
            print('starting_frame:',starting_frame)
            print('starting_frame_nonzero:',starting_frame_nonzero)
            print('num_frame:',num_frame)
            print('frame_index:',frame_index)
            print('frame_index + starting_frame_nonzero:',frame_index + starting_frame_nonzero)
            print(name)
            img = image[frame_index + starting_frame_nonzero,...]
            
            img = Image.fromarray(img).convert('RGB')  # 转为 RGB 图像
            # img = np.stack([img] * 3, axis=-1)  # 重复通道来生成 RGB 图像 (H, W, 3)
            mask = data_seg_3d[frame_index,...]
            # mask = np.rot90(mask)
            obj_list = np.unique(mask[mask > 0])
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
                diff_obj_mask_dict[obj] = obj_mask

                if self.prompt == 'click':
                    diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = random_click(np.array(obj_mask.squeeze(0)), point_label, seed=None)
                if self.prompt == 'bbox':
                    diff_obj_bbox_dict[obj] = generate_bbox(np.array(obj_mask.squeeze(0)), variation=self.variation, seed=self.seed)
            img = img.resize(newsize)
            # img = resize_data(img, (3,1024,1024))
            # img = img.astype(np.uint8)  # 或 np.float32（取决于你想要的效果）
            # img = Image.fromarray(img).resize(newsize)
            img = img.resize(newsize)
            img = torch.tensor(np.array(img)).permute(2, 0, 1)
            # print(img.shape) #torch.Size([3, 1024, 1024])
            # print(img_tensor[frame_index - starting_frame,:, :, :].shape) #torch.Size([3,1024,1024])
            # print(img_tensor.shape)
            img_tensor[frame_index - starting_frame,:, :, :] = img
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