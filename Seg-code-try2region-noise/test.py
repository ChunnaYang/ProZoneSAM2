import os
import torch
import cfg
from func_3d import function
from func_3d.utils import get_network
from func_3d.dataset import get_dataloader

def main():
    args = cfg.parse_args()
    GPUdevice = torch.device('cuda', args.gpu_device)

    print(f"🚀 Loading pretrained model from {args.pretrain}")
    net = get_network(args, args.net, use_gpu=True, gpu_device=GPUdevice)
    checkpoint = torch.load(args.pretrain, map_location=GPUdevice)
    net.load_state_dict(checkpoint['model'])
    net.eval()

    # 加载所有指定测试集
    test_loaders = get_dataloader(args, mode="test")

    os.makedirs("test_results_noise30", exist_ok=True)

    # 遍历测试集并调用 test_sam
    for subset_name, loader in test_loaders.items():
        print(f"\n🧪 Evaluating subset: {subset_name}")
        function.test_sam(args, loader, net, save_root="test_results_noise30", testset_name=subset_name)

if __name__ == '__main__':
    main()
