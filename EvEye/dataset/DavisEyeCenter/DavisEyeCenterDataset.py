from pathlib import Path
import numpy as np
import torch
from natsort import natsorted
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from EvEye.utils.dvs_common_utils.processor.EventRandomAffine import (
    EventRandomAffine,
    rand_range,
)
from EvEye.utils.dvs_common_utils.representation.TorchFrameStack import (
    TorchFrameStack,
)
from EvEye.utils.processor.TxtProcessor import TxtProcessor
from EvEye.utils.cache.MemmapCacheStructedEvents import load_memmap


class DavisEyeCenterDataset(Dataset):
    def __init__(
        self,
        root_path: Path | str,
        split="train",  # 'train', 'val' or 'test'
        time_window=40000,  # us
        frames_per_segment=50,
        spatial_downsample=(2, 2),
        events_interpolation="bilinear",  # 'bilinear', 'nearest', 'causal_linear'
        spatial_affine=True,
        temporal_shift=True,
        temporal_flip=True,
        temporal_scale=True,
        cache=False,
    ):
        assert time_window == 40000
        self.root_path = Path(root_path)
        self.split = split
        self.time_window = time_window
        self.frames_per_segment = frames_per_segment
        self.time_window_per_segment = time_window * frames_per_segment
        self.spatial_downsample = spatial_downsample
        self.events_interpolation = events_interpolation

        self.spatial_affine = spatial_affine
        self.temporal_shift = temporal_shift
        self.temporal_flip = temporal_flip
        self.temporal_scale = temporal_scale

        self.cache = cache

        self.events, self.labels = [], []
        self.num_frames_list, self.num_segments_list = [], []

        # Get data and label paths
        assert self.split in ["train", "val", "test"], "Invalid mode."
        if self.cache:
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
                labels_indices = load_memmap(
                    label_indices_path, label_indices_info_path
                )
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

        else:
            if self.split == "test":
                self.data_base_path: Path = self.root_path / self.split / "data"
                self.data_paths: list = natsorted(self.data_base_path.glob("*.txt"))
                for data_path in tqdm(
                    self.data_paths,
                    total=len(self.data_paths),
                    desc="Loading test data...",
                ):
                    assert data_path.is_file(), "Invalid data file."
                    event = TxtProcessor(data_path).load_events_from_txt()
                    event["t"] = event["t"] - event["t"].min()
                    num_frames = event["t"].max() // time_window
                    self.num_frames_list.append(num_frames)
                    label = torch.zeros((4, num_frames), dtype=torch.float32)
                    event = np.stack(
                        [
                            event["p"].astype("float32"),
                            event["x"].astype("float32"),
                            event["y"].astype("float32"),
                            event["t"].astype("float32"),
                        ],
                        axis=0,
                    )
                    event = torch.from_numpy(event)  # (4, N), 4 means (p, x, y, t)
                    self.events.append(event)
                    self.labels.append(label)

            elif self.split in ["train", "val"]:
                self.data_base_path: Path = self.root_path / self.split / "data"
                self.label_base_path: Path = self.root_path / self.split / "label"
                self.data_paths: list = natsorted(self.data_base_path.glob("*.txt"))
                self.label_paths: list = natsorted(self.label_base_path.glob("*.txt"))
                for data_path, label_path in tqdm(
                    zip(self.data_paths, self.label_paths),
                    total=len(self.data_paths),
                    desc="Loading data...",
                ):
                    assert (
                        data_path.is_file() and label_path.is_file()
                    ), "Invalid data or label file."
                    assert (
                        data_path.stem.split("_")[:6] == label_path.stem.split("_")[:6]
                    ), "Data and label mismatch."
                    event, label = (
                        TxtProcessor(data_path).load_events_from_txt(),
                        TxtProcessor(label_path).load_labels_from_txt(),
                    )

                    # Get the slice of the event data and label we want
                    num_frames = label.shape[0]
                    self.num_frames_list.append(num_frames)
                    self.num_segments_list.append(num_frames // frames_per_segment)

                    start_time = label["t"][0]
                    final_time = start_time + num_frames * time_window
                    start_index = np.searchsorted(event["t"], start_time, "left")
                    final_index = np.searchsorted(event["t"], final_time, "left")
                    event = event[start_index:final_index]

                    event["t"] -= start_time
                    label["t"] -= start_time

                    # Transform the data and label into torch tensors
                    label = np.stack(
                        [
                            label["t"].astype("float32"),
                            label["x"].astype("float32"),
                            label["y"].astype("float32"),
                            label["close"].astype("float32"),
                        ],
                        axis=0,
                    )
                    label = torch.from_numpy(label)  # (4, N), 4 means (t, x, y, close)
                    event = np.stack(
                        [
                            event["p"].astype("float32"),
                            event["x"].astype("float32"),
                            event["y"].astype("float32"),
                            event["t"].astype("float32"),
                        ],
                        axis=0,
                    )
                    event = torch.from_numpy(event)  # (4, N), 4 means (p, x, y, t)

                    self.events.append(event)
                    self.labels.append(label)

                self.total_segments = sum(self.num_segments_list)

    def __len__(self):
        if self.split in ["train", "val"]:
            return self.total_segments
        elif self.split == "test":
            return len(self.events)
        else:
            raise ValueError("Invalid mode. Must be train, val or test.")

    def __getitem__(self, index):
        if self.split in ["train", "val"]:
            # spatial affine transformation
            augment_flag = (self.split == "train") and self.spatial_affine
            self.augment = EventRandomAffine((260, 346), augment_flag=augment_flag)

            file_id, segment_id = self.get_index(self.num_segments_list, index)
            event, label = self.events[file_id], self.labels[file_id]
            # event, label = torch.from_numpy(event), torch.from_numpy(label)
            if event.shape == torch.Size([4, 0]):
                print(
                    f"Invalid file_id:{file_id}, segment_id:{segment_id}, index:{index}."
                )
                with open("Invalid_file_id_segment_id_index.txt", "a") as f:
                    f.write(
                        f"Invalid file_id:{file_id}, segment_id:{segment_id}, index:{index}.\n"
                    )
                return self.__getitem__(index + 1)

            start_time = (
                label[0][0] + segment_id * self.time_window * self.frames_per_segment
            )
            end_time = start_time + self.time_window * self.frames_per_segment

            # temporal shift
            max_offset = round(self.time_window_per_segment * 0.1)
            if (
                self.split == "train"
                and self.temporal_shift
                and start_time >= max_offset
            ):
                offset = np.random.rand() * max_offset
                start_time -= offset
                end_time -= offset
            else:
                offset = 0

            # temporal scaling
            num_frames = self.num_frames_list[file_id]
            event = event.clone()
            if (
                self.split == "train"
                and self.temporal_scale
                and end_time < (num_frames * self.time_window * 0.8)
            ):
                scale_factor = float(rand_range(0.8, 1.2))
                event[-1] *= scale_factor
            else:
                scale_factor = 1

            start_id = torch.searchsorted(event[-1], start_time, side="left")
            end_id = torch.searchsorted(event[-1], end_time, side="left")

            event_segment = event[:, start_id.item() : end_id.item()]
            event_segment[-1] -= start_time

            # label interpolation
            start_label_id = segment_id * self.frames_per_segment
            end_label_id = (segment_id + 1) * self.frames_per_segment
            label_numpy = label.cpu().numpy()
            arange = np.arange(0, num_frames)
            label_offset = offset / self.time_window
            interp_range = np.linspace(
                (start_label_id - label_offset) / scale_factor,
                (end_label_id - label_offset - 1) / scale_factor,
                self.frames_per_segment,
            )
            x_interp = np.interp(interp_range, arange, label_numpy[1, :])
            y_interp = np.interp(interp_range, arange, label_numpy[2, :])
            closeness = label_numpy[:, start_label_id:end_label_id][-1]
            label_segment = torch.tensor(
                np.stack([x_interp, y_interp, closeness], axis=1)
            ).type_as(label)

            # TorchFrameStack
            event, center, close = self.augment(event_segment, label_segment)
            num_frames = self.frames_per_segment
            event = TorchFrameStack(
                events=event,
                size=(
                    260 // self.spatial_downsample[0],
                    346 // self.spatial_downsample[1],
                ),
                num_frames=num_frames,
                spatial_downsample=self.spatial_downsample,
                temporal_downsample=self.time_window,
                mode=self.events_interpolation,
            )

            # time and polarity flip
            if self.split == "train" and self.temporal_flip and np.random.rand() > 0.5:
                event = event.flip(0).flip(1)
                center = center.flip(-1)
                close = close.flip(-1)

            event = event.moveaxis(0, 1).to(torch.float32)
            center = center.to(torch.float32)
            close = 1 - close.to(torch.float32)

            return event, center, close

        elif self.split == "test":
            event, label = self.events[index], self.labels[index]
            augment_flag = (self.split == "train") and self.spatial_affine
            self.augment = EventRandomAffine((260, 346), augment_flag=augment_flag)
            event, center, close = self.augment(event, label)
            num_frames = self.num_frames_list[index]
            event = TorchFrameStack(
                events=event,
                size=(
                    260 // self.spatial_downsample[0],
                    346 // self.spatial_downsample[1],
                ),
                num_frames=num_frames,
                spatial_downsample=self.spatial_downsample,
                temporal_downsample=self.time_window,
                mode=self.events_interpolation,
            )

            event = event.moveaxis(0, 1).to(torch.float32)
            center = center.to(torch.float32)
            close = 1 - close.to(torch.float32)

            return event, center, close

        else:
            raise ValueError("Invalid mode. Must be train, val or test.")

    def get_index(self, file_lens, index):
        file_lens_cumsum = np.cumsum(np.array(file_lens))
        file_id = np.searchsorted(file_lens_cumsum, index, side="right")
        sample_id = index - file_lens_cumsum[file_id - 1] if file_id > 0 else index

        return file_id, sample_id


def main():
    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader
    from EvEye.model.DavisEyeCenter.TennSt import TennSt

    dataset = DavisEyeCenterDataset(
        root_path="/mnt/data2T/junyuan/eye-tracking/testDataset",
        split="train",
        time_window=40000,
        frames_per_segment=50,
        spatial_downsample=(2, 2),
        events_interpolation="causal_linear",
        spatial_affine=True,
        temporal_flip=True,
        temporal_scale=True,
        temporal_shift=True,
        cache=False,
    )
    print(len(dataset))
    dataset[101]
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    print(len(dataset))
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
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # model = TennSt(
    #     channels=[2, 8, 16, 32, 48, 64, 80, 96, 112, 128, 256],
    #     t_kernel_size=5,
    #     n_depthwise_layers=4,
    #     detector_head=True,
    #     detector_depthwise=True,
    #     full_conv3d=False,
    #     norms="mixed",
    # )
    # model = model.cuda()
    # for event, center, close in dataloader:
    #     print(event.shape, center.shape, close.shape)
    #     output = model(event)
    #     print(output.shape)
    #     break


if __name__ == "__main__":
    main()
