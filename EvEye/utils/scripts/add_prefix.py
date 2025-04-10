import os


def add_prefix_to_png_files(directory, prefix="right_"):
    # 遍历指定目录下的所有文件和文件夹
    for filename in os.listdir(directory):
        # 检查文件扩展名是否为.png
        if filename.endswith(".png"):
            # 构建原始文件的完整路径
            old_file = os.path.join(directory, filename)
            # 构建新文件名，即在原文件名前添加"left_"
            new_filename = prefix + filename
            # 构建新文件的完整路径
            new_file = os.path.join(directory, new_filename)
            # 重命名文件
            os.rename(old_file, new_file)
            print(f"Renamed {old_file} to {new_file}")


def main():
    target_directory = "/mnt/data2T/junyuan/eye-tracking/datasets/Data_davis_labelled_with_mask/right/data"
    add_prefix_to_png_files(target_directory)


if __name__ == "__main__":
    main()
