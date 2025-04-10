import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from typing import Tuple, List
from EvEye.utils.processor.HDF5Processor import HDF5Processor


def process_and_save_images(
    data: np.ndarray,
    label: np.ndarray,
    base_path: str,
    file_base: str,
    start_index: int = 0,
) -> None:
    data_path: str = os.path.join(base_path, "data")
    label_path: str = os.path.join(base_path, "label")
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)

    total_images: int = data.shape[-1]
    with tqdm(total=total_images, desc=f"Processing images for {file_base}") as pbar:
        for i in range(data.shape[-1]):
            data_image: np.ndarray = data[..., i]
            label_image: np.ndarray = label[..., i]
            data_image_path: str = os.path.join(
                data_path, f"{file_base}_{i + start_index}.png"
            )
            label_image_path: str = os.path.join(
                label_path, f"{file_base}_{i + start_index}.png"
            )
            plt.imsave(data_image_path, data_image, cmap="gray")
            plt.imsave(label_image_path, label_image, cmap="gray")
            pbar.update(1)


def h5_png_split(h5_file_path: str) -> None:
    h5_files: List[str] = [f for f in os.listdir(h5_file_path) if f.endswith(".h5")]

    for h5 in h5_files:
        filepath: str = os.path.join(h5_file_path, h5)
        file_base: str = os.path.splitext(h5)[0]

        with HDF5Processor(filepath) as processor:
            data: np.ndarray = processor.read_data("data").transpose(1, 0, 2)
            label: np.ndarray = processor.read_data("label").transpose(1, 0, 2)

        # Process and save training images
        process_and_save_images(
            data,
            label,
            h5_file_path,
            file_base,
        )


def main():
    h5_png_split(
        "/mnt/data2T/junyuan/eye-tracking/datasets/Data_davis_labelled_with_mask/right",
    )
    print("Processing complete. Files have been saved.")


if __name__ == "__main__":
    main()
