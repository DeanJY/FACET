import os


def process_file(input_file):
    with open(input_file, "r") as file:
        lines = file.readlines()

    modified_lines = []
    last_valid_coords = None  # 用于保存最后一次有效的坐标
    future_coords_index = None  # 为“No contour found”后查找坐标时使用的索引

    for i in range(len(lines)):
        line = lines[i]
        if "No contour found" in line:
            if last_valid_coords:
                # 使用前一个有效的坐标替换
                new_line = f'{line.split(",")[0]},{last_valid_coords},1\n'
                modified_lines.append(new_line)
            else:
                # 如果前面没有有效坐标，向后查找有效坐标
                future_coords_index = i + 1
                while future_coords_index < len(lines):
                    future_line = lines[future_coords_index]
                    parts = future_line.strip().split(",")
                    if len(parts) > 3:
                        # 找到后续第一个有效坐标
                        future_valid_coords = ",".join(parts[1:3])
                        new_line = f'{line.split(",")[0]},{future_valid_coords},1\n'
                        modified_lines.append(new_line)
                        break
                    future_coords_index += 1
                # 如果整个文件都没找到有效坐标
                if future_coords_index >= len(lines):
                    modified_lines.append(line)
        else:
            parts = line.strip().split(",")
            if len(parts) > 3:
                last_valid_coords = ",".join(parts[1:3])
            modified_lines.append(line)

    # 写入同一文件以覆盖原始数据
    with open(input_file, "w") as file:
        file.writelines(modified_lines)


def process_all_txt_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            process_file(file_path)
            print(f"Processed and updated {file_path}")


def main():
    folder_path = "/mnt/data2T/junyuan/eye-tracking/datasets/DavisEyeCenterDataset/test/label"  # 替换为您的文件夹路径
    process_all_txt_files(folder_path)


if __name__ == "__main__":
    main()
