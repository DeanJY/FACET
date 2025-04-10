import os


def rename_png(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".png"):
                original_full_path = os.path.join(root, file)
                name_part, ext = os.path.splitext(file)

                parts = name_part.split("_")
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    pre = parts[0]
                    aft = parts[1]
                    aft = str(int(aft))
                    name = f"{int(pre + aft):06}.png"
                    new_full_path = os.path.join(root, name)
                    os.rename(original_full_path, new_full_path)
                    print(f"Renamed: {original_full_path} -> {new_full_path}")


def main():
    dir_path = (
        "/mnt/data2T/junyuan/eye-tracking/outputs/EventsFrames"  # 更改为您的目录路径
    )
    rename_png(dir_path)
    print("All files have been renamed successfully.")


if __name__ == "__main__":
    main()
