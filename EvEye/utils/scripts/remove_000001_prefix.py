import re
from pathlib import Path


def remove_numeric_prefix(directory):
    dir_path = Path(directory)
    pattern = re.compile(r"^\d+_")
    for file_path in dir_path.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
        new_lines = [pattern.sub("", line) for line in lines]
        with open(file_path, "w", encoding="utf-8") as file:
            file.writelines(new_lines)
    print("Done.")


def main():
    directory_path = (
        "/mnt/data2T/junyuan/eye-tracking/datasets/DavisEyeCenterDataset/val/label"
    )
    remove_numeric_prefix(directory_path)


if __name__ == "__main__":
    main()
