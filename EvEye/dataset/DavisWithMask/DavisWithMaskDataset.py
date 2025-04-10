import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
from pathlib import Path
from natsort import natsorted
import albumentations as A
from EvEye.utils.tonic.functional.ToFrameStack import to_frame_stack_numpy
from EvEye.utils.cache.MemmapCacheStructedEvents import *
from EvEye.utils.visualization.visualization import *
from EvEye.utils.tonic.functional.CutMaxCount import cut_max_count


class DavisWithMaskDataset(Dataset):
    def __init__(
        self,
        root_path,
        split,
        default_resolution=(256, 256),
        events_interpolation="causal_linear",
        mode="event",
    ) -> None:
        super(DavisWithMaskDataset, self).__init__()
        self.root_path = Path(root_path)
        self.split = split
        self.mode = mode
        self.events_interpolation = events_interpolation
        self.height, self.width = default_resolution

        self.data_path = self.root_path / self.split / "data"
        self.label_path = self.root_path / self.split / "label"
        self.images = natsorted(list(self.data_path.rglob("*.png")))
        self.masks = natsorted(list(self.label_path.rglob("*.png")))
        assert len(self.images) == len(self.masks)
        self.nums = len(self.images)


    def get_nums(self):
        num_frames_list = []

        ellipses_list = load_cached_structed_ellipses(self.ellipse_path)
        for ellipses in ellipses_list:
            num_frames_list.append(len(ellipses))
        total_frames = sum(num_frames_list)
        return total_frames

    def get_transform(self):
        return A.Compose([A.Resize(height=self.height, width=self.width)])

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self.masks[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask[mask == 255] = 1.0

        transform = self.get_transform()
        transformed = transform(image=image, mask=mask)
        image = transformed["image"]
        image = np.expand_dims(image, axis=0)
        mask = transformed["mask"]
        if np.sum(mask > 0) < 2:
            close = 1
        else:
            close = 0
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        ret = {"image": image, "mask": mask, "close": close}

        return ret


def main():
    base_path = Path("/mnt/data2T/junyuan/Datasets/RGBUNetDataset")

    dataset = DavisWithMaskDataset(
        root_path=base_path,
        split="train",
        default_resolution=(256, 256),
        events_interpolation="causal_linear",
        mode="rgb",
    )
    data = dataset[0]
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for i, (x, y) in enumerate(dataloader):  # 遍历数据加载器
        print(f'Batch {i+1}:')  # 打印批次号
        print(f'Input shape: {x.shape}')  # 打印特征数据的形状
        print(f'Output shape: {y.shape}')
        print()

    print(f"Total number of samples in dataset: {len(dataset)}")
    data, mask = dataset[0]  # 使用索引方式获取数据，更符合习惯
    print(f"Data type: {type(data)}, Mask type: {type(mask)}")
    print(f"Data shape: {data.shape}, Mask shape: {mask.shape}")
    print(f"Max value in data: {data.max()}, Max value in mask: {mask.max()}")

    # 归一化数据以便于保存图片
    # data = (data - data.min()) / (data.max() - data.min())
    # save_image(data, "data.png")
    # save_image(mask, "mask.png")

    # 输出mask查看其数值分布
    print(mask)


if __name__ == "__main__":
    main()
