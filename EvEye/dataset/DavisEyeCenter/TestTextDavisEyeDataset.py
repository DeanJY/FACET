import numpy as np
import torch

from pathlib import Path
from torch.utils.data import Dataset
from tonic import transforms
from natsort import natsorted
from EvEye.utils.processor.TxtProcessor import TxtProcessor
from EvEye.utils.tonic.functional.ToFrameStack import to_frame_stack_numpy
from EvEye.utils.tonic.slicers.SliceEventsAtIndices import slice_events_at_timepoints
from EvEye.utils.tonic.slicers.SliceWithTimestampAndCount import (
    slice_events_by_timestamp_and_count,
)


class TestTextDavisEyeDataset(Dataset):
    def __init__(
        self,
        txt_path: Path | str,
        label_path: Path | str,
        split="test",  # 'test'
        time_window=40000,  # us
        frames_per_segment=50,
        sensor_size=(346, 260, 2),
        events_interpolation="causal_linear",  # 'bilinear', 'nearest', 'causal_linear'
        spatial_factor=0.5,
        fixed_count=None,
    ):
        self.events = TxtProcessor(txt_path).load_events_from_txt()
        self.labels = TxtProcessor(label_path).load_labels_from_txt()
        self.events["t"] -= self.labels["t"][0] - time_window
        self.labels["t"] -= self.labels["t"][0] - time_window
        self.split = split
        self.time_window = time_window
        self.frames_per_segment = frames_per_segment
        self.sensor_size = sensor_size
        self.events_interpolation = events_interpolation
        self.spatial_factor = spatial_factor
        self.fixed_count = fixed_count
        self.total_segments = self.labels.shape[0] // self.frames_per_segment
        self.downsaple_transform = transforms.Downsample(
            spatial_factor=self.spatial_factor
        )
        self.sensor_size = (
            int(self.sensor_size[0] * self.spatial_factor),
            int(self.sensor_size[1] * self.spatial_factor),
            int(self.sensor_size[2]),
        )

    def __len__(self):
        return 1

    def __getitem__(self, index):
        start_event_time = 0
        end_event_time = self.labels["t"][-1]
        event_segment = slice_events_at_timepoints(
            self.events, start_event_time, end_event_time
        )

        if self.fixed_count is None:
            event_segment_downsampled = self.downsaple_transform(event_segment)
            event_frames_downsampled = to_frame_stack_numpy(
                event_segment_downsampled,
                self.sensor_size,
                self.labels.shape[0],
                self.events_interpolation,
                start_event_time,
                end_event_time,
            )

        elif self.fixed_count is not None:
            event_frames_list = []
            for index in range(self.labels.shape[0]):
                segment_start_time = start_event_time + self.time_window * index
                segment_end_time = start_event_time + self.time_window * (index + 1)
                event_segment_raw = slice_events_by_timestamp_and_count(
                    event_segment, segment_end_time, self.fixed_count, forward=False
                )
                event_segment_downsampled = self.downsaple_transform(event_segment_raw)
                event_frame = to_frame_stack_numpy(
                    event_segment_downsampled,
                    self.sensor_size,
                    1,
                    self.events_interpolation,
                    segment_start_time,
                    segment_end_time,
                )
                event_frames_list.append(event_frame)
            event_frames_downsampled = np.concatenate(event_frames_list, axis=0)

        event_frames_downsampled = torch.from_numpy(
            np.transpose(event_frames_downsampled, (1, 0, 2, 3)).astype("float32")
        )
        event_frames_downsampled = torch.unsqueeze(event_frames_downsampled, 0)

        event_frames = to_frame_stack_numpy(
            event_segment,
            (346, 260, 2),
            self.labels.shape[0],
            self.events_interpolation,
            start_event_time,
            end_event_time,
        )

        return event_frames_downsampled, event_frames


def main():
    from torch.utils.data import DataLoader

    txt_path = "/mnt/data2T/junyuan/eye-tracking/datasets/DavisEyeCenterDataset/train/data/user1_left_session_1_0_1_events.txt"
    label_path = "/mnt/data2T/junyuan/eye-tracking/datasets/DavisEyeCenterDataset/train/label/user1_left_session_1_0_1_centers.txt"
    dataset = TestTextDavisEyeDataset(
        txt_path=txt_path,
        label_path=label_path,
        split="test",
        time_window=40000,
        sensor_size=(346, 260, 2),
        events_interpolation="causal_linear",
        spatial_factor=0.5,
        fixed_count=5000,
    )

    print(
        f"shape: {dataset[0][0].shape}",
        f"dtype: {dataset[0][0].dtype}",
        f"max: {dataset[0][0].max()}",
        f"min: {dataset[0][0].min()}",
    )


if __name__ == "__main__":
    main()
