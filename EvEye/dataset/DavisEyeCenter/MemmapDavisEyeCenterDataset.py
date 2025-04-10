import os
import cv2
import torch
import tonic
import numpy as np
import albumentations as A

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from tonic import slicers, transforms, functional
from EvEye.utils.tonic.functional.ToFrameStack import to_frame_stack_numpy
from EvEye.utils.tonic.slicers.SliceEventsAtIndices import slice_events_at_timepoints
from EvEye.utils.tonic.slicers.SliceWithTimestampAndCount import (
    slice_events_by_timestamp_and_count,
)
from EvEye.utils.dvs_common_utils.processor.NumpyEventFrameRandomAffine import (
    NumpyEventFrameRandomAffine,
)
from EvEye.utils.cache.MemmapCacheStructedEvents import (
    load_cached_structed_events,
    load_cached_structed_labels,
    get_nums,
)
from EvEye.utils.visualization.visualization import visualize, save_image


class DataAugmentation:
    def __init__(self, split, spatial_transform=True):
        self.split = split
        self.spatial_transform = spatial_transform
        self.transforms = self.get_transforms()
        self.replay = None  # 用于存储第一次应用的增强操作

    def get_transforms(self):
        if self.split == "train":
            if self.spatial_transform:
                return A.ReplayCompose(
                    [
                        A.ShiftScaleRotate(
                            shift_limit=0.2,
                            scale_limit=0.2,
                            rotate_limit=15,
                            interpolation=cv2.INTER_LINEAR,
                            border_mode=cv2.BORDER_CONSTANT,
                            p=1,
                        ),
                        A.HorizontalFlip(p=0.5),
                    ],
                    keypoint_params=A.KeypointParams(
                        format="xy", remove_invisible=False, angle_in_degrees=True
                    ),
                )
            else:
                return A.ReplayCompose(
                    [],
                    keypoint_params=A.KeypointParams(format="xy"),
                )
        elif self.split == "val" or self.split == "test":
            return A.ReplayCompose(
                [],
                keypoint_params=A.KeypointParams(format="xy"),
            )

    def apply_transforms(self, image, keypoints):
        if self.replay is None:
            # 第一次应用增强操作并记录
            augmented = self.transforms(
                image=image,
                keypoints=keypoints,
            )
            self.replay = augmented["replay"]
        else:
            # 重放记录的增强操作
            augmented = A.ReplayCompose.replay(
                self.replay,
                image=image,
                keypoints=keypoints,
            )
        return augmented["image"], augmented["keypoints"]

    def process_frames_and_labels(self, event_frames, labels):
        for i in range(labels.shape[0]):
            frame = np.transpose(event_frames[i], (1, 2, 0))
            label = [tuple(labels[i])]
            frame, label = self.apply_transforms(frame, label)
            event_frames[i] = np.transpose(frame, (2, 0, 1))
            labels[i] = label[0]
            # self.show_transforms(event_frames[i], labels[i], i)

    def show_transforms(self, event_frame, label, i):
        event_frame_vis = visualize(event_frame)
        x, y = label
        output_path = "/mnt/data2T/junyuan/eye-tracking/Transforms"
        os.makedirs(output_path, exist_ok=True)
        cv2.circle(event_frame_vis, (int(x), int(y)), 5, (0, 255, 0), -1)
        save_image(
            event_frame_vis,
            f"{output_path}/event_frame_{i}.png",
        )


class MemmapDavisEyeCenterDataset(Dataset):
    def __init__(
        self,
        root_path: Path | str,
        split="train",  # 'train', 'val' or 'test'
        time_window=40000,  # us
        frames_per_segment=50,
        sensor_size=(346, 260, 2),
        events_interpolation="causal_linear",  # 'bilinear', 'nearest', 'causal_linear'
        fixed_count: int = None,
        spatial_downsaple=True,  # downsample factor is 0.5
        saptial_transform=True,
        temporal_transform=True,
    ):
        assert split in ["train", "val"], "Invalid split."
        self.root_path = Path(root_path)
        self.split = split
        self.data_path = self.root_path / self.split / "cached_data"
        self.labels_path = self.root_path / self.split / "cached_label"
        self.time_window = time_window
        self.frames_per_segment = frames_per_segment
        self.sensor_size = sensor_size
        self.events_interpolation = events_interpolation
        self.fixed_count = fixed_count
        self.spatial_downsaple = spatial_downsaple
        self.saptial_transform = saptial_transform
        self.temporal_transform = temporal_transform

        self.events, self.labels = [], []
        self.num_frames_list, self.num_segments_list, self.total_segments = get_nums(
            self.labels_path, frames_per_segment=self.frames_per_segment
        )

    def get_index(self, file_lens, index):
        file_lens_cumsum = np.cumsum(np.array(file_lens))
        file_id = np.searchsorted(file_lens_cumsum, index, side="right")
        sample_id = index - file_lens_cumsum[file_id - 1] if file_id > 0 else index

        return file_id, sample_id

    def temporal_flip(self, event_frame: np.array, label: np.array, closes: np.array):
        assert event_frame.ndim == 4
        assert label.ndim == 2 and label.shape[1] == 2
        assert closes.ndim == 1 and closes.shape[0] == label.shape[0]

        flipped_event_frame = np.flip(event_frame, axis=(0, 1))
        flipped_label = np.flip(label, axis=0)
        flipped_closes = np.flip(closes, axis=0)

        return flipped_event_frame.copy(), flipped_label.copy(), flipped_closes.copy()

    def __len__(self):
        return self.total_segments

    def __getitem__(self, index):
        augmenter = DataAugmentation(self.split, self.saptial_transform)

        self.events = load_cached_structed_events(self.data_path)
        self.labels = load_cached_structed_labels(self.labels_path)
        file_id, segment_id = self.get_index(self.num_segments_list, index)
        event, label = self.events[file_id], self.labels[file_id]

        start_event_time = segment_id * self.frames_per_segment * self.time_window
        end_event_time = (segment_id + 1) * self.frames_per_segment * self.time_window
        event_segment = slice_events_at_timepoints(
            event, start_event_time, end_event_time
        )

        start_label_id = segment_id * self.frames_per_segment
        end_label_id = (segment_id + 1) * self.frames_per_segment
        label_segment = label[start_label_id:end_label_id]

        if self.spatial_downsaple:
            spatial_downsaple_transform = transforms.Downsample(spatial_factor=0.5)
            event_segment = spatial_downsaple_transform(event_segment)
            label_segment = spatial_downsaple_transform(label_segment)
            sensor_size = (
                int(self.sensor_size[0] * 0.5),
                int(self.sensor_size[1] * 0.5),
                int(self.sensor_size[2]),
            )
        else:
            sensor_size = self.sensor_size

        if self.fixed_count is None:
            event_frames = to_frame_stack_numpy(
                event_segment,
                sensor_size,
                self.frames_per_segment,
                self.events_interpolation,
                start_event_time,
                end_event_time,
            )

        elif self.fixed_count is not None:
            event_frames_list = []
            for index in range(self.frames_per_segment):
                segment_start_time = start_event_time + self.time_window * index
                segment_end_time = start_event_time + self.time_window * (index + 1)
                fixed_count_events = slice_events_by_timestamp_and_count(
                    event_segment, segment_end_time, self.fixed_count, forward=False
                )
                event_frame = to_frame_stack_numpy(
                    fixed_count_events,
                    sensor_size,
                    1,
                    self.events_interpolation,
                    segment_start_time,
                    segment_end_time,
                )
                event_frames_list.append(event_frame)
            event_frames = np.concatenate(event_frames_list, axis=0)

        labels = np.stack(
            [
                label_segment["x"].astype("float32"),
                label_segment["y"].astype("float32"),
            ],
            axis=1,
        )

        closes = label_segment["close"].astype("float32")

        augmenter.process_frames_and_labels(event_frames, labels)

        labels[:, 0] /= sensor_size[0]
        labels[:, 1] /= sensor_size[1]

        if self.split == "train" and self.temporal_transform and np.random.rand() > 0.5:
            event_frames, labels, closes = self.temporal_flip(
                event_frames, labels, closes
            )

        event_frames = torch.from_numpy(
            np.transpose(event_frames, (1, 0, 2, 3)).astype("float32")
        )
        labels = torch.from_numpy(np.transpose(labels, (1, 0)).astype("float32"))
        closes = torch.from_numpy((1 - closes).astype("float32"))

        return event_frames, labels, closes


def main():
    from torch.utils.data import DataLoader
    from EvEye.model.DavisEyeCenter.TennSt import TennSt
    import time

    dataset = MemmapDavisEyeCenterDataset(
        # root_path="/mnt/data2T/junyuan/eye-tracking/testDataset",
        root_path="/mnt/data2T/junyuan/eye-tracking/datasets/MemmapDavisEyeCenterDataset",
        split="train",
        time_window=40000,
        frames_per_segment=50,
        sensor_size=(346, 260, 2),
        events_interpolation="causal_linear",
        # fixed_count=5000,
        spatial_downsaple=False,
        saptial_transform=True,
        temporal_transform=True,
    )

    model = TennSt(
        channels=[2, 8, 16, 32, 48, 64, 80, 96, 112, 128, 256],
        t_kernel_size=5,
        n_depthwise_layers=4,
        detector_head=True,
        detector_depthwise=True,
        full_conv3d=False,
        norms="mixed",
    )

    dataset[120]
    print(f"Total segments: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    for i, (x, y, z) in enumerate(dataloader):
        print(f"Batch {i+1}:")
        # print(f"Input shape: {x.shape}")
        # output = model(x)
        # print(f"Output shape: {output.shape}")
        print(f"Data shape: {x.shape}")
        print(f"Data dtype: {x.dtype}")
        print(f"Label shape: {y.shape}")
        print(f"Label dtype: {y.dtype}")
        print(f"Close shape: {z.shape}")
        print(f"Close dtype: {z.dtype}")
        # print(x, y, z)
        print()

    # output = model(dataset[0][0].unsqueeze(0))
    # print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()
