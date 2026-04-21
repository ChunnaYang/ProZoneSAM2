from .btcv import BTCV
from .amos import AMOS
from .prostate import PROSTATE
from torch.utils.data import DataLoader
import os


def get_dataloader(args, mode="train"):
    """
    mode:
        'train'：返回 (train_loader, val_loader)
        'test' ：根据 args.testset 返回对应测试集的 DataLoader
    """

    if args.dataset == 'prostate':
        if mode == "train":
            # ✅ 训练 + 验证
            train_dataset = PROSTATE(args, args.data_path, None, None, mode='train', prompt=args.prompt)
            val_dataset   = PROSTATE(args, args.data_path, None, None, mode='val', prompt=args.prompt)
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
            val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
            return train_loader, val_loader

        elif mode == "test":
            # ✅ 按用户指定加载
            testsets = [s.strip() for s in args.testset.split(',')]
            loaders = {}

            for subset in testsets:
                subset_dir = os.path.join(args.data_path, subset)
                if not os.path.exists(subset_dir):
                    print(f"⚠️  Subset '{subset}' not found, skipped.")
                    continue

                print(f"✅ Loading test subset: {subset}")
                dataset = PROSTATE(args, args.data_path, None, None, mode=subset, prompt=args.prompt)
                loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
                loaders[subset] = loader

            if not loaders:
                raise FileNotFoundError("❌ No valid test subsets found!")

            return loaders  # 返回字典 {subset_name: loader}

    else:
        raise ValueError(f"Dataset {args.dataset} not supported!")
