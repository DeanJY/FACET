import shutil
from pathlib import Path
from tqdm import tqdm
import os
import natsort

root_path = Path("/mnt/data2T/junyuan/eye-tracking/EV_Eye_dataset/raw_data/Data_davis")
new_events_folder = Path(
    "/mnt/data2T/junyuan/eye-tracking/datasets/DavisEyeCenterDataset"
)
new_events_folder.mkdir(parents=True, exist_ok=True)
new_frames_folder = Path(
    "/mnt/data2T/junyuan/eye-tracking/datasets/DavisEyeCenterDataset_frames"
)
new_frames_folder.mkdir(parents=True, exist_ok=True)

user_ids = range(1, 49)
sides = ["left", "right"]
session_patterns = [
    "session_1_0_1",
    "session_1_0_2",
    "session_2_0_1",
    "session_2_0_2",
]

events_pattern = "events/events.txt"
centers_pattern = "frames/centers.txt"
frames_pattern = "frames"


def move_events(
    user_ids, sides, session_patterns, events_pattern, centers_pattern, new_folder
):
    for user_id in tqdm(user_ids, desc="Copying files"):
        user_folder = root_path / f"user{user_id}"
        for side in sides:
            side_folder = user_folder / side
            for session_pattern in session_patterns:
                session_folder = side_folder / session_pattern
                events_path = session_folder / events_pattern
                centers_path = session_folder / centers_pattern
                if not events_path.exists():
                    print(f"events_path is None: {events_path}")
                    continue
                if not centers_path.exists():
                    print(f"centers_path is None: {centers_path}")
                    continue

                new_events_name = f"user{user_id}_{side}_{session_pattern}_events.txt"
                new_centers_name = f"user{user_id}_{side}_{session_pattern}_centers.txt"

                new_events_path = new_folder / new_events_name
                new_centers_path = new_folder / new_centers_name

                shutil.move(events_path, new_events_path)
                shutil.move(centers_path, new_centers_path)
    print("Done.")


def move_frames(user_ids, sides, session_patterns, frames_pattern, new_frames_folder):
    for user_id in tqdm(user_ids, desc="Moving frames folders"):
        user_folder = root_path / f"user{user_id}"
        for side in sides:
            side_folder = user_folder / side
            for session_pattern in session_patterns:
                session_folder = side_folder / session_pattern
                frames_path = session_folder / frames_pattern

                # 检查 frames 文件夹是否存在
                if not frames_path.exists():
                    print(f"frames_path does not exist: {frames_path}")
                    continue

                # 生成新的文件夹名称和路径
                new_frames_name = f"user{user_id}_{side}_{session_pattern}"
                new_frames_path = new_frames_folder / new_frames_name

                # 移动并重命名文件夹
                shutil.move(frames_path, new_frames_path)

    print("Frames folders moved and renamed.")


def split_train_val_test(base_path, train_ratio, val_ratio):
    # base_path下events.txt文件为data，centers.txt文件为label
    train_data_path = base_path / "train/data"
    train_label_path = base_path / "train/label"
    val_data_path = base_path / "val/data"
    val_label_path = base_path / "val/label"
    test_data_path = base_path / "test/data"
    test_label_path = base_path / "test/label"
    train_data_path.mkdir(parents=True, exist_ok=True)
    train_label_path.mkdir(parents=True, exist_ok=True)
    val_data_path.mkdir(parents=True, exist_ok=True)
    val_label_path.mkdir(parents=True, exist_ok=True)
    test_data_path.mkdir(parents=True, exist_ok=True)
    test_label_path.mkdir(parents=True, exist_ok=True)

    data_files = natsort.natsorted(
        [f for f in os.listdir(base_path) if f.endswith("events.txt")]
    )
    label_files = natsort.natsorted(
        [f for f in os.listdir(base_path) if f.endswith("centers.txt")]
    )

    num_files = len(data_files)
    num_train = int(num_files * train_ratio)
    num_val = int(num_files * val_ratio)
    num_test = num_files - num_train - num_val

    train_files = data_files[:num_train]
    val_files = data_files[num_train : num_train + num_val]
    test_files = data_files[num_train + num_val :]

    for file in tqdm(train_files, desc="Moving train files", unit="file"):
        shutil.move(base_path / file, train_data_path / file)
        shutil.move(
            base_path / file.replace("events", "centers"),
            train_label_path / file.replace("events", "centers"),
        )

    for file in tqdm(val_files, desc="Moving val files", unit="file"):
        shutil.move(base_path / file, val_data_path / file)
        shutil.move(
            base_path / file.replace("events", "centers"),
            val_label_path / file.replace("events", "centers"),
        )

    for file in tqdm(test_files, desc="Moving test files", unit="file"):
        shutil.move(base_path / file, test_data_path / file)
        shutil.move(
            base_path / file.replace("events", "centers"),
            test_label_path / file.replace("events", "centers"),
        )

    print("Train-val-test split complete. Train set, val set, and test set are ready.")


def main():
    # move_files(
    #     user_ids, sides, session_patterns, events_pattern, centers_pattern, new_folder
    # )
    move_frames(user_ids, sides, session_patterns, frames_pattern, new_frames_folder)


if __name__ == "__main__":
    main()
