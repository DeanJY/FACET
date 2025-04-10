import os
import shutil
from tqdm import tqdm
from random import shuffle


def get_num_files(path: str) -> int:
    num_files = len(
        [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    )
    print(f"Number of files in {path}: {num_files}")
    return num_files


def move_files(source_path: str, target_path: str) -> None:
    os.makedirs(target_path, exist_ok=True)
    # Only move files, not directories
    files = [
        f
        for f in os.listdir(source_path)
        if os.path.isfile(os.path.join(source_path, f))
    ]

    for file in tqdm(files, desc="Moving files", unit="file"):
        src_path = os.path.join(source_path, file)
        dst_path = os.path.join(target_path, file)
        shutil.move(src_path, dst_path)  # Move file
        print(f"Moved: {src_path} -> {dst_path}")


def split_train_val(base_path: str, train_ratio: float = 0.9) -> None:
    data_path = os.path.join(base_path, "data")
    label_path = os.path.join(base_path, "label")
    assert os.path.exists(data_path) and os.path.exists(
        label_path
    ), f"Data path {data_path} or label path {label_path} does not exist."

    # create train and val directories
    train_data_path = os.path.join(base_path, "train", "data")
    train_label_path = os.path.join(base_path, "train", "label")
    val_data_path = os.path.join(base_path, "val", "data")
    val_label_path = os.path.join(base_path, "val", "label")

    for path in [train_data_path, train_label_path, val_data_path, val_label_path]:
        os.makedirs(path, exist_ok=True)

    # get all file names
    file_names = [f for f in os.listdir(data_path) if f.endswith(".png")]
    shuffle(file_names)  # 打乱文件名以随机划分训练集和验证集

    # get number of files for training
    num_train = int(len(file_names) * train_ratio)

    # split files
    train_files = file_names[:num_train]
    val_files = file_names[num_train:]

    # move files
    for file in tqdm(train_files, desc="Moving train files", unit="file"):
        shutil.move(os.path.join(data_path, file), os.path.join(train_data_path, file))
        shutil.move(
            os.path.join(label_path, file),
            os.path.join(train_label_path, file),
        )

    for file in tqdm(val_files, desc="Moving validation files", unit="file"):
        shutil.move(os.path.join(data_path, file), os.path.join(val_data_path, file))
        shutil.move(os.path.join(label_path, file), os.path.join(val_label_path, file))

    print("Train-val split complete. Train set and val set are ready.")


def compare_filenames(path_1: str, path_2: str) -> bool:
    # get all filenames in the directories
    filenames_1 = {
        os.path.splitext(f)[0]
        for f in os.listdir(path_1)
        if os.path.isfile(os.path.join(path_1, f))
    }
    filenames_2 = {
        os.path.splitext(f)[0]
        for f in os.listdir(path_2)
        if os.path.isfile(os.path.join(path_2, f))
    }

    # compare filenames
    if filenames_1 == filenames_2:
        # print same filenames
        print("The directories have the same filenames.")
        return True
    else:
        print("The directories do not have the same filenames.")
        # print missing filenames
        missing_in_dir1 = filenames_2 - filenames_1
        missing_in_dir2 = filenames_1 - filenames_2
        if missing_in_dir1:
            print("Missing in directory 1:", missing_in_dir1)
        if missing_in_dir2:
            print("Missing in directory 2:", missing_in_dir2)
        return False


def main():
    # get_num_files(
    #     "/mnt/data2T/junyuan/eye-tracking/datasets/Data_davis_labelled_with_mask/val/data"
    # )

    # move_files(
    #     source_path="/mnt/data2T/junyuan/eye-tracking/datasets/Data_davis_labelled_with_mask/right/label",
    #     target_path="/mnt/data2T/junyuan/eye-tracking/datasets/Data_davis_labelled_with_mask/label",
    # )

    split_train_val(
        base_path="/mnt/data2T/junyuan/eye-tracking/datasets/Data_davis_labelled_with_mask",
        train_ratio=0.9,
    )

    # compare_filenames(
    #     path_1="/mnt/data2T/junyuan/eye-tracking/datasets/Data_davis_labelled_with_mask/val/data",
    #     path_2="/mnt/data2T/junyuan/eye-tracking/datasets/Data_davis_labelled_with_mask/val/label",
    # )


if __name__ == "__main__":
    main()
