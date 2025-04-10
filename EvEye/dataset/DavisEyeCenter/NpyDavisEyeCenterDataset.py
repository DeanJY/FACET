import numpy as np
import torch
import time

from torch.utils.data import Dataset
from pathlib import Path
from EvEye.utils.dvs_common_utils.processor.NumpyEventFrameRandomAffine import (
    NumpyEventFrameRandomAffine,
)

# from EvEye.utils.dvs_common_utils.processor.TorchEventFrameRandomAffine import (
#     TorchEventFrameRandomAffine,
# )


class NpyDavisEyeCenterDataset(Dataset):
    def __init__(
        self,
        root_path: Path | str,
        split="train",  # 'train' or 'val'
        spatial_affine=True,
        temporal_flip=True,
        device="cuda:0",
    ):
        assert split in ["train", "val"], "Invalid split."

        self.root_path = Path(root_path)
        self.split = split
        self.spatial_affine = spatial_affine
        self.temporal_flip = temporal_flip
        self.data_path = self.root_path / self.split / "np_data"
        self.label_path = self.root_path / self.split / "np_label"
        self.close_path = self.root_path / self.split / "np_close"
        self.device = device

    def __len__(self):
        return len(list(self.data_path.glob("*.npy")))

    def __getitem__(self, index):
        # start_time = time.time()
        data = np.load(self.data_path / f"{index}.npy")
        label = np.load(self.label_path / f"{index}.npy")
        close = np.load(self.close_path / f"{index}.npy")
        # data = torch.from_numpy(data).to(self.device)
        # label = torch.from_numpy(label).to(self.device)
        # close = torch.from_numpy(close).to(self.device)
        # load_time = time.time()
        # print(f"Load time: {load_time - start_time}")
        if self.split == "train":
            augment = NumpyEventFrameRandomAffine()
            # augment = TorchEventFrameRandomAffine()
            data = augment.transform_event_frame(data)
            label = augment.transform_label(label)
            if self.temporal_flip and np.random.rand() > 0.5:
                data, label, close = augment.temporal_flip(data, label, close)
        # transform_time = time.time()
        # print(f"Transform time: {transform_time - load_time}")
        data = torch.from_numpy(data).to(torch.float32)
        label = torch.from_numpy(label).to(torch.float32)
        close = torch.from_numpy(close).to(torch.float32)

        return data, label, close


def main():
    from torch.utils.data import DataLoader
    import time

    dataset = NpyDavisEyeCenterDataset(
        root_path="/mnt/data2T/junyuan/eye-tracking/testDataset",
        split="train",
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    for i, (x, y, z) in enumerate(dataloader):
        print(f"Batch {i+1}:")
        print(f"Data shape: {x.shape}")
        print(f"Data dtype: {x.dtype}")
        print(f"Label shape: {y.shape}")
        print(f"Label dtype: {y.dtype}")
        print(f"Close shape: {z.shape}")
        print(f"Close dtype: {z.dtype}")
        # print(f'Input data: {x}')
        # print(f'Output data: {y}')
        print()


if __name__ == "__main__":
    main()
