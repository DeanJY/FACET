import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from EvEye.utils.visualization.visualization import *

import albumentations as A
from albumentations.pytorch import ToTensorV2


class TestDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path | str,
        transform: bool = True,
        height: int = None,
        width: int = None,
        **kwargs,
    ) -> None:
        """
        TestDataset constructor

        Args:
            dataset_path (Path | str): root path to dataset
            transform (bool, optional): whether to apply transforms. Defaults to False.
            height (int): The height to resize images to.
            width (int): The width to resize images to.
        Returns:
            None
        """
        self._data_path = Path(dataset_path)
        self._transform = transform
        self._height = height
        self._width = width
        self._extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
        self._images = sorted(
            [
                file
                for extention in self._extensions
                for file in list(self._data_path.rglob(extention))
            ]
        )

    def _get_transform(self):
        return A.Compose(
            [
                A.Resize(height=240, width=346),
                A.Normalize(
                    mean=[0.0],
                    std=[1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        image_path = self._images[index]
        image_name = image_path.stem
        origin_image = load_image(str(image_path), "grayscale")[0]
        image = origin_image.copy()
        if self._transform:
            transform = self._get_transform()
            transformed = transform(image=image)
            image = transformed["image"]
        else:
            image = torch.from_numpy(image).unsqueeze(0).float()
        return image, origin_image, image_name


def main():
    dataset = TestDataset(
        dataset_path="/mnt/data2T/junyuan/eye-tracking/datasets/Data_davis_labelled_with_mask/test/data"
    )
    image, origin_image, image_name = dataset[0]
    image = image.numpy().transpose(1, 2, 0)
    print(image.shape, image.dtype, type(image))
    print(image.shape, image.dtype, type(image))
    print(type(image_name), image_name)
    save_image(origin_image, "origin_image.png", BGR2RGB=True)
    save_image(image, "image.png", BGR2RGB=True)


if __name__ == "__main__":
    main()
