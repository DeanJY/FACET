import time
from pathlib import Path
import numpy as np
from natsort import natsorted
from tqdm import tqdm

from EvEye.utils.processor.TxtProcessor import TxtProcessor


def merge_arrays(arrays):
    if arrays[0].ndim == 1:
        total_length = sum(arr.shape[0] for arr in arrays)
        merged_array = np.zeros((total_length,), dtype=arrays[0].dtype)
        indices = []
        current_index = 0
        for arr in tqdm(arrays, desc="Merging arrays"):
            end_index = current_index + arr.shape[0]
            merged_array[current_index:end_index] = arr
            indices.append((current_index, end_index))
            current_index = end_index
        return merged_array, indices
    else:
        total_length = sum(arr.shape[1] for arr in arrays)
        merged_array = np.zeros((arrays[0].shape[0], total_length))
        indices = []
        current_index = 0
        for arr in tqdm(arrays, desc="Merging arrays"):
            end_index = current_index + arr.shape[1]
            merged_array[:, current_index:end_index] = arr
            indices.append((current_index, end_index))
            current_index = end_index
        return merged_array, indices


def create_memmap(data, data_file, info_file):
    mmap = np.memmap(data_file, dtype=data.dtype, mode="w+", shape=data.shape)
    mmap[:] = data
    mmap.flush()
    with open(info_file, "w") as f:
        f.write(f"Data shape: {data.shape}\n")
        f.write(f"Data dtype: {data.dtype}\n")
    return mmap


def load_memmap(data_file, info_file):
    with open(info_file, "r") as f:
        lines = f.readlines()
        shape_line = lines[0].strip()
        dtype_line = lines[1].strip()
        shape_str = shape_line.split(": ")[1]
        shape = tuple(map(int, shape_str.strip("()").split(",")))
        dtype_str = dtype_line.split(": ")[1]
        dtype = np.dtype(dtype_str)
    mmap = np.memmap(data_file, dtype=dtype, mode="r", shape=shape)
    return mmap


def load_memmaps(data_path, label_path, frames_per_segment=50):
    data_path, label_path = Path(data_path), Path(label_path)
    events_list, labels_list = [], []
    num_frames_list, num_segments_list = [], []
    data_paths = natsorted(data_path.glob("events_batch_*.memmap"))
    data_info_paths = natsorted(data_path.glob("events_info_batch_*.txt"))
    data_indices_paths = natsorted(data_path.glob("events_indices_batch_*.memmap"))
    data_indices_info_paths = natsorted(
        data_path.glob("events_indices_info_batch_*.txt")
    )
    label_paths = natsorted(label_path.glob("labels_batch_*.memmap"))
    label_info_paths = natsorted(label_path.glob("labels_info_batch_*.txt"))
    label_indices_paths = natsorted(label_path.glob("labels_indices_batch_*.memmap"))
    label_indices_info_paths = natsorted(
        label_path.glob("labels_indices_info_batch_*.txt")
    )
    for (
        data_path,
        data_info_path,
        data_indices_path,
        data_indices_info_path,
        label_path,
        label_info_path,
        label_indices_path,
        label_indices_info_path,
    ) in zip(
        data_paths,
        data_info_paths,
        data_indices_paths,
        data_indices_info_paths,
        label_paths,
        label_info_paths,
        label_indices_paths,
        label_indices_info_paths,
    ):
        events = load_memmap(data_path, data_info_path)
        labels = load_memmap(label_path, label_info_path)
        events_indices = load_memmap(data_indices_path, data_indices_info_path)
        labels_indices = load_memmap(label_indices_path, label_indices_info_path)

        for indice in events_indices:
            event = events[:, indice[0] : indice[1]]
            events_list.append(event)

        for indice in labels_indices:
            num_frames = indice[1] - indice[0]
            num_frames_list.append(num_frames)
            num_segments_list.append(num_frames // frames_per_segment)
            labels = labels[:, indice[0] : indice[1]]
            labels_list.append(labels)
    total_segments = sum(num_segments_list)

    return events_list, labels_list, num_frames_list, num_segments_list, total_segments


class CacheDavisEyeCenterDataset:
    def __init__(
        self,
        root_path: Path | str,
        split="train",
        time_window=40000,
        frames_per_segment=50,
        spatial_downsample=(2, 2),
        events_interpolation="bilnear",
        batch_size=10,
    ):
        self.output_data_path = Path("/data/junyuan/eye-tracking/train/cached_data")
        self.output_label_path = Path("/data/junyuan/eye-tracking/train/cached_label")
        self.output_data_path.mkdir(parents=True, exist_ok=True)
        self.output_label_path.mkdir(parents=True, exist_ok=True)
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

        self.batch_size = batch_size
        self.batch_events = []
        self.batch_labels = []
        self.batch_counter = 0

    def cache(self):
        if self.split in ["train", "val", "test"]:
            self.data_base_path: Path = self.root_path / self.split / "data"
            self.label_base_path: Path = self.root_path / self.split / "label"
            self.data_paths: list = natsorted(self.data_base_path.glob("*.txt"))
            self.label_paths: list = natsorted(self.label_base_path.glob("*.txt"))
        else:
            raise ValueError("Invalid mode. Must be train, val or test.")

        for data_index, (data_path, label_path) in enumerate(
            tqdm(
                zip(self.data_paths, self.label_paths),
                total=len(self.data_paths),
                desc="Loading data and label...",
            )
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
            self.num_segments_list.append(num_frames // self.frames_per_segment)

            start_time = label["t"][0] - self.time_window
            final_time = start_time + num_frames * self.time_window
            start_index = np.searchsorted(event["t"], start_time, "left")
            final_index = np.searchsorted(event["t"], final_time, "left")
            event = event[start_index:final_index]

            event["t"] -= event["t"][0]
            label["t"] -= start_time

            # Transform the data and label into torch tensors
            # label = np.stack(
            #     [
            #         label["t"].astype("int32"),
            #         label["x"].astype("int32"),
            #         label["y"].astype("int32"),
            #         label["close"].astype("int32"),
            #     ],
            #     axis=0,
            # )
            # event = np.stack(
            #     [
            #         event["p"].astype("int32"),
            #         event["x"].astype("int32"),
            #         event["y"].astype("int32"),
            #         event["t"].astype("int32"),
            #     ],
            #     axis=0,
            # )
            # self.events.append(event)
            # self.labels.append(label)
            self.batch_events.append(event)
            self.batch_labels.append(label)

            if (data_index + 1) % self.batch_size == 0 or (data_index + 1) == len(
                self.data_paths
            ):
                merged_events, events_indices = merge_arrays(self.batch_events)
                merged_labels, labels_indices = merge_arrays(self.batch_labels)
                create_memmap(
                    merged_events,
                    f"{self.output_data_path}/events_batch_{self.batch_counter}.memmap",
                    f"{self.output_data_path}/events_info_batch_{self.batch_counter}.txt",
                )
                create_memmap(
                    merged_labels,
                    f"{self.output_label_path}/labels_batch_{self.batch_counter}.memmap",
                    f"{self.output_label_path}/labels_info_batch_{self.batch_counter}.txt",
                )
                create_memmap(
                    np.array(events_indices),
                    f"{self.output_data_path}/events_indices_batch_{self.batch_counter}.memmap",
                    f"{self.output_data_path}/events_indices_info_batch_{self.batch_counter}.txt",
                )
                create_memmap(
                    np.array(labels_indices),
                    f"{self.output_label_path}/labels_indices_batch_{self.batch_counter}.memmap",
                    f"{self.output_label_path}/labels_indices_info_batch_{self.batch_counter}.txt",
                )

                self.batch_events = []
                self.batch_labels = []
                self.batch_counter += 1

        self.total_segments = sum(self.num_segments_list)


def main():
    dataset = CacheDavisEyeCenterDataset(
        root_path="/mnt/data2T/junyuan/eye-tracking/datasets/DavisEyeCenterDataset",
        split="val",
        time_window=40000,
        frames_per_segment=50,
        spatial_downsample=(2, 2),
        events_interpolation="bilinear",
        batch_size=10,
    )
    dataset.cache()
    del dataset
    dataset = CacheDavisEyeCenterDataset(
        root_path="/mnt/data2T/junyuan/eye-tracking/datasets/DavisEyeCenterDataset",
        split="train",
        time_window=40000,
        frames_per_segment=50,
        spatial_downsample=(2, 2),
        events_interpolation="bilinear",
        batch_size=10,
    )
    dataset.cache()
    # start_time = time.time()
    # events = load_memmap(
    #     "/mnt/data2T/junyuan/eye-tracking/events_indices_batch_0.memmap",
    #     "/mnt/data2T/junyuan/eye-tracking/events_indices_info_batch_0.txt",
    # )
    # print(events.shape)
    # end_time = time.time()
    # print(f"Time taken: {end_time - start_time}s")


if __name__ == "__main__":
    main()
