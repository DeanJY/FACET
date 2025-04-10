from pathlib import Path
import numpy as np
import torch
import os
from natsort import natsorted
from tqdm import tqdm

from EvEye.utils.scripts.CacheFrameStack import load_memmap
from EvEye.utils.dvs_common_utils.representation.TorchFrameStack import (
    TorchFrameStack,
)
from EvEye.utils.scripts.CacheFrameStack import *


class NpyCacheDavisEyeCenterDataset:
    def __init__(
        self,
        root_path: Path | str,
        split="train",
        time_window=40000,
        frames_per_segment=50,
        spatial_downsample=(2, 2),
        events_interpolation="bilinear",  # 'bilinear', 'nearest', 'causal_linear'
    ):
        assert time_window == 40000
        self.root_path = Path(root_path)
        self.split = split
        self.time_window = time_window
        self.frames_per_segment = frames_per_segment
        self.time_window_per_segment = time_window * frames_per_segment
        self.spatial_downsample = spatial_downsample
        self.events_interpolation = events_interpolation

        self.events, self.labels = [], []
        self.num_frames_list, self.num_segments_list = [], []

        self.data_base_path: Path = self.root_path / self.split / "cached_data"
        self.label_base_path: Path = self.root_path / self.split / "cached_label"
        self.data_paths: list = natsorted(
            self.data_base_path.glob("events_batch_*.memmap")
        )
        self.label_paths: list = natsorted(
            self.label_base_path.glob("labels_batch_*.memmap")
        )
        self.data_info_paths: list = natsorted(
            self.data_base_path.glob("events_info_batch_*.txt")
        )
        self.label_info_paths: list = natsorted(
            self.label_base_path.glob("labels_info_batch_*.txt")
        )
        self.data_indices_paths: list = natsorted(
            self.data_base_path.glob("events_indices_batch_*.memmap")
        )
        self.label_indices_paths: list = natsorted(
            self.label_base_path.glob("labels_indices_batch_*.memmap")
        )
        self.data_indices_info_paths = natsorted(
            self.data_base_path.glob("events_indices_info_batch_*.txt")
        )
        self.label_indices_info_paths = natsorted(
            self.label_base_path.glob("labels_indices_info_batch_*.txt")
        )
        for (
            data_path,
            label_path,
            data_info_path,
            label_info_path,
            data_indices_path,
            label_indices_path,
            data_indices_info_path,
            label_indices_info_path,
        ) in tqdm(
            zip(
                self.data_paths,
                self.label_paths,
                self.data_info_paths,
                self.label_info_paths,
                self.data_indices_paths,
                self.label_indices_paths,
                self.data_indices_info_paths,
                self.label_indices_info_paths,
            ),
            total=len(self.data_paths),
            desc="Loading data...",
        ):
            events = load_memmap(data_path, data_info_path)
            events_indices = load_memmap(data_indices_path, data_indices_info_path)
            labels = load_memmap(label_path, label_info_path)
            labels_indices = load_memmap(label_indices_path, label_indices_info_path)
            for indice in events_indices:
                event = events[:, indice[0] : indice[1]]
                self.events.append(event)

            for indice in labels_indices:
                num_frames = indice[1] - indice[0]
                self.num_frames_list.append(num_frames)
                self.num_segments_list.append(num_frames // frames_per_segment)
                label = labels[:, indice[0] : indice[1]]
                self.labels.append(label)
        self.total_segments = sum(self.num_segments_list)

    def get_index(self, file_lens, index):
        file_lens_cumsum = np.cumsum(np.array(file_lens))
        file_id = np.searchsorted(file_lens_cumsum, index, side="right")
        sample_id = index - file_lens_cumsum[file_id - 1] if file_id > 0 else index

        return file_id, sample_id

    def __len__(self):
        return self.total_segments

    def __getitem__(self, index):
        file_id, segment_id = self.get_index(self.num_segments_list, index)
        event, label = self.events[file_id], self.labels[file_id]
        start_time = (
            label[0][0] + segment_id * self.time_window * self.frames_per_segment
        )
        end_time = start_time + self.time_window * self.frames_per_segment

        start_event_id = np.searchsorted(event[3], start_time, side="left")
        end_event_id = np.searchsorted(event[3], end_time, side="left")
        event_segment = event[:, start_event_id:end_event_id]
        event_segment = np.array(event_segment)
        event_segment[-1] -= start_time
        num_frames = self.frames_per_segment
        event_segment = torch.from_numpy(event_segment)
        # print(event_segment.shape)
        event_frame = TorchFrameStack(
            events=event_segment,
            size=(
                260 // self.spatial_downsample[0],
                346 // self.spatial_downsample[1],
            ),
            num_frames=num_frames,
            spatial_downsample=self.spatial_downsample,
            temporal_downsample=self.time_window,
            mode=self.events_interpolation,
        )
        event_frame = event_frame.moveaxis(0, 1)
        event_frame = event_frame.numpy()

        start_label_id = segment_id * self.frames_per_segment
        end_label_id = start_label_id + self.frames_per_segment
        label_segment = label[:, start_label_id:end_label_id]
        label_x = (label_segment[1] / 2).round()
        label_y = (label_segment[2] / 2).round()
        label_coord = np.vstack([label_x, label_y])

        closeness = 1 - np.array(label_segment[3])

        return event_frame, label_coord, closeness


def save_npy(base_path, dataset):
    base_path = Path(base_path)
    output_data_path = base_path / "np_data"
    output_label_path = base_path / "np_label"
    output_close_path = base_path / "np_close"
    os.makedirs(output_data_path, exist_ok=True)
    os.makedirs(output_label_path, exist_ok=True)
    os.makedirs(output_close_path, exist_ok=True)
    for i in tqdm(range(len(dataset)), desc="Saving data"):
        # 获取数据
        data, label, close = dataset[i]

        # 分别保存 data, label, close
        np.save(f"{output_data_path}/{i}.npy", data)
        np.save(f"{output_label_path}/{i}.npy", label)
        np.save(f"{output_close_path}/{i}.npy", close)


def main():
    train_dataset = NpyCacheDavisEyeCenterDataset(
        root_path="/mnt/data2T/junyuan/eye-tracking/datasets/DavisEyeCenterDataset",
        split="train",
        time_window=40000,
        frames_per_segment=50,
        spatial_downsample=(2, 2),
        events_interpolation="bilinear",  # 'bilinear', 'nearest', 'causal_linear'
    )
    val_dataset = NpyCacheDavisEyeCenterDataset(
        root_path="/mnt/data2T/junyuan/eye-tracking/datasets/DavisEyeCenterDataset",
        split="val",
        time_window=40000,
        frames_per_segment=50,
        spatial_downsample=(2, 2),
        events_interpolation="bilinear",  # 'bilinear', 'nearest', 'causal_linear'
    )

    output_train_path = (
        "/mnt/data8T/junyuan/eye-tracking/datasets/DavisEyeCenterDataset/train"
    )
    output_val_path = (
        "/mnt/data8T/junyuan/eye-tracking/datasets/DavisEyeCenterDataset/val"
    )

    save_npy(output_train_path, train_dataset)
    save_npy(output_val_path, val_dataset)


if __name__ == "__main__":
    main()
