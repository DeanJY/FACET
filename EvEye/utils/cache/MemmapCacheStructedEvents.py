import time
from pathlib import Path
import numpy as np
from natsort import natsorted
from tqdm import tqdm

from EvEye.utils.processor.TxtProcessor import TxtProcessor


def merge_structed_arrays(arrays: list) -> np.ndarray:
    total_length = sum([array.shape[0] if array.shape else 1 for array in arrays])
    merged_array = np.zeros(total_length, dtype=arrays[0].dtype)
    current_index = 0
    for array in tqdm(arrays, desc="Merging arrays..."):
        if array.shape:
            end_index = current_index + array.shape[0]
            merged_array[current_index:end_index] = array
            current_index = end_index
        else:
            merged_array[current_index] = array
            current_index += 1
    return merged_array


def get_indices(arrays: list) -> np.ndarray:
    indices_array = np.zeros((len(arrays), 2), dtype=np.int32)
    current_index = 0
    for index, array in tqdm(enumerate(arrays), desc="Getting indices..."):
        if array.shape:
            end_index = current_index + array.shape[0]
        else:
            end_index = current_index + 1
        indices_array[index] = [current_index, end_index]
        current_index = end_index
    return indices_array


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
        shape = tuple(
            int(num) for num in shape_str.strip("()").split(",") if num.strip()
        )

        dtype_str = dtype_line.split(": ")[1]
        dtype = eval(dtype_str)

    mmap = np.memmap(data_file, dtype=dtype, mode="r", shape=shape)
    return mmap


def cache_structed_events(batch_size, data_base_path, output_path, start_batch=0):
    data_base_path = Path(data_base_path)
    data_paths = natsorted(data_base_path.glob("*.txt"))

    output_path = Path(output_path)
    output_data_path = output_path / "cached_data"
    output_data_path.mkdir(parents=True, exist_ok=True)

    batch_counter = start_batch
    batch_events = []

    # 计算已经处理的文件数量
    files_processed = start_batch * batch_size

    for index, data_path in enumerate(
        tqdm(data_paths[files_processed:], total=len(data_paths) - files_processed)
    ):
        event = TxtProcessor(data_path).load_events_from_txt()
        batch_events.append(event)

        if (index + 1) % batch_size == 0 or (index + 1) == len(
            data_paths
        ) - files_processed:
            events_merged = merge_structed_arrays(batch_events)
            events_indices = get_indices(batch_events)
            create_memmap(
                events_merged,
                f"{output_data_path}/events_batch_{batch_counter}.memmap",
                f"{output_data_path}/events_batch_info_{batch_counter}.txt",
            )
            np.save(
                f"{output_data_path}/events_indices_{batch_counter}.npy", events_indices
            )

            batch_events = []
            batch_counter += 1


def cache_structed_ellipses(batch_size, ellipse_path, output_path):
    ellipse_path = Path(ellipse_path)
    ellipse_paths = natsorted(ellipse_path.glob("*.txt"))

    output_path = Path(output_path)
    output_data_path = output_path / "cached_ellipse"
    output_data_path.mkdir(parents=True, exist_ok=True)

    batch_conter = 0
    batch_ellipses = []

    for index, ellipse_path in enumerate(tqdm(ellipse_paths, total=len(ellipse_paths))):
        ellipse = TxtProcessor(ellipse_path).load_ellipses_from_txt()
        batch_ellipses.append(ellipse)

        if (index + 1) % batch_size == 0 or (index + 1) == len(ellipse_paths):
            ellipses_merged = merge_structed_arrays(batch_ellipses)
            ellipses_indices = get_indices(batch_ellipses)
            create_memmap(
                ellipses_merged,
                f"{output_data_path}/ellipses_batch_{batch_conter}.memmap",
                f"{output_data_path}/ellipses_batch_info_{batch_conter}.txt",
            )
            np.save(
                f"{output_data_path}/ellipses_indices_{batch_conter}.npy",
                ellipses_indices,
            )
            batch_ellipses = []
            batch_conter += 1


def cache_structed_data(
    time_window, frames_per_segment, batch_size, base_path, output_path
):
    base_path = Path(base_path)
    output_path = Path(output_path)
    output_data_path = output_path / "cached_data"
    output_label_path = output_path / "cached_label"
    output_data_path.mkdir(parents=True, exist_ok=True)
    output_label_path.mkdir(parents=True, exist_ok=True)
    data_base_path = base_path / "data"
    label_base_path = base_path / "label"
    data_paths = natsorted(data_base_path.glob("*.txt"))
    label_paths = natsorted(label_base_path.glob("*.txt"))

    batch_counter = 0
    batch_events = []
    batch_labels = []
    num_frames_list = []
    num_segments_list = []
    for index, (data_path, label_path) in enumerate(
        tqdm(zip(data_paths, label_paths), total=len(data_paths))
    ):
        event = TxtProcessor(data_path).load_events_from_txt()
        label = TxtProcessor(label_path).load_labels_from_txt()

        index_label = 0
        while index_label < len(label['t']):
            start_time_first = max(event['t'][0], label['t'][index_label] - time_window)
            end_time_first = label['t'][index_label]
            if start_time_first >= end_time_first:
                index_label += 1
                if index_label < len(label['t']):
                    continue
                else:
                    raise ValueError("No event before the first label")
            else:
                break
        label = label[index_label:]
        num_frames = label.shape[0]
        num_frames_list.append(num_frames)
        num_segments_list.append(num_frames // frames_per_segment)

        batch_events.append(event)
        batch_labels.append(label)

        if (index + 1) % batch_size == 0 or (index + 1) == len(data_paths):
            events_merged = merge_structed_arrays(batch_events)
            events_indices = get_indices(batch_events)
            labels_merged = merge_structed_arrays(batch_labels)
            labels_indices = get_indices(batch_labels)
            create_memmap(
                events_merged,
                f"{output_data_path}/events_batch_{batch_counter}.memmap",
                f"{output_data_path}/events_batch_info_{batch_counter}.txt",
            )
            create_memmap(
                labels_merged,
                f"{output_label_path}/labels_batch_{batch_counter}.memmap",
                f"{output_label_path}/labels_batch_info_{batch_counter}.txt",
            )
            np.save(
                f"{output_data_path}/events_indices_{batch_counter}.npy", events_indices
            )
            np.save(
                f"{output_label_path}/labels_indices_{batch_counter}.npy",
                labels_indices,
            )

            batch_counter += 1


def load_cached_structed_events(events_path):
    events_list = []
    events_path = Path(events_path)
    events_paths = natsorted(events_path.glob("events_batch_*.memmap"))
    events_info_paths = natsorted(events_path.glob("events_batch_info_*.txt"))
    events_indices_paths = natsorted(events_path.glob("events_indices_*.npy"))

    for events_path, events_info_path, events_indices_path in zip(
        events_paths, events_info_paths, events_indices_paths
    ):
        events = load_memmap(events_path, events_info_path)
        events_indices = np.load(events_indices_path)

        for index in range(events_indices.shape[0]):
            start_index, end_index = events_indices[index]
            events_list.append(events[start_index:end_index])

    return events_list


def load_cached_structed_labels(labels_path):
    labels_list = []
    labels_path = Path(labels_path)
    labels_paths = natsorted(labels_path.glob("labels_batch_*.memmap"))
    labels_info_paths = natsorted(labels_path.glob("labels_batch_info_*.txt"))
    labels_indices_paths = natsorted(labels_path.glob("labels_indices_*.npy"))

    for labels_path, labels_info_path, labels_indices_path in zip(
        labels_paths, labels_info_paths, labels_indices_paths
    ):
        labels = load_memmap(labels_path, labels_info_path)
        labels_indices = np.load(labels_indices_path)

        for index in range(labels_indices.shape[0]):
            start_index, end_index = labels_indices[index]
            labels_list.append(labels[start_index:end_index])

    return labels_list


def load_cached_structed_ellipses(ellipses_path):
    ellipses_list = []
    ellipses_path = Path(ellipses_path)
    ellipses_paths = natsorted(ellipses_path.glob("ellipses_batch_*.memmap"))
    ellipses_info_paths = natsorted(ellipses_path.glob("ellipses_batch_info_*.txt"))
    ellipses_indices_paths = natsorted(ellipses_path.glob("ellipses_indices_*.npy"))

    for ellipses_path, ellipses_info_path, ellipses_indices_path in zip(
        ellipses_paths, ellipses_info_paths, ellipses_indices_paths
    ):
        ellipses = load_memmap(ellipses_path, ellipses_info_path)
        ellipses_indices = np.load(ellipses_indices_path)

        for index in range(ellipses_indices.shape[0]):
            start_index, end_index = ellipses_indices[index]
            ellipses_list.append(ellipses[start_index:end_index])

    return ellipses_list


def load_event_segment(index, data_base_path, batch_size=5000):
    data_base_path = Path(data_base_path)
    batch_id = index // batch_size
    event_id = index % batch_size
    events_batch_path = data_base_path / f"events_batch_{batch_id}.memmap"
    events_info_path = data_base_path / f"events_batch_info_{batch_id}.txt"
    event_indices_path = data_base_path / f"events_indices_{batch_id}.npy"
    events = load_memmap(events_batch_path, events_info_path)
    start_index, end_index = np.load(event_indices_path)[event_id]
    event_segment = events[start_index:end_index]

    return event_segment


def load_ellipse(index, ellipse_base_path):
    ellipse_base_path = Path(ellipse_base_path)
    ellipses_path = ellipse_base_path / f"ellipses_batch_0.memmap"
    ellipses_info_path = ellipse_base_path / f"ellipses_batch_info_0.txt"
    ellipses = load_memmap(ellipses_path, ellipses_info_path)
    ellipse = ellipses[index]

    return ellipse


def get_nums(labels_path, frames_per_segment=50):
    labels_path = Path(labels_path)
    labels_indices_paths = natsorted(labels_path.glob("labels_indices_*.npy"))

    num_frames_list = []
    num_segments_list = []
    for labels_indices_path in labels_indices_paths:
        labels_indices = np.load(labels_indices_path)
        for index in range(labels_indices.shape[0]):
            start_index, end_index = labels_indices[index]
            num_frames = end_index - start_index
            num_frames_list.append(num_frames)
            num_segments_list.append(num_frames // frames_per_segment)
    total_segments = sum(num_segments_list)
    return num_frames_list, num_segments_list, total_segments


def main():
    cache_structed_events(
        50,
        "/mnt/data2T/junyuan/eye-tracking/datasets/DavisEyeCenterDataset/data",
        "/mnt/data2T/junyuan/eye-tracking/datasets/MemmapDavisEyeEllipseDataset",
        0,
    )

    print("Done")


if __name__ == "__main__":
    main()
