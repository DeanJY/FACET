import cv2
import os
import numpy as np


def play_images(
    image_path,
    height,  # canvas height
    width,  # canvas width
    play_mode: str = "fps",  # "fps" or "time"
    mode_arg=60,  # fps or total time
):
    """
    Play a sequence of images in a folder

    Args:
        image_path (str): Path to the folder containing images
        height (int): Height of the canvas
        width (int): Width of the canvas
        play_mode (str, optional): Play mode. "fps" or "time". If mode is time, the
            mode_arg should provide the total time in seconds. Defaults to "fps".
        mode_arg (int, optional): Argument for the play mode. Default is 60 (fps or total time).

    Returns:
        None
    """
    image_files = [
        os.path.join(image_path, f)
        for f in sorted(os.listdir(image_path))
        if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    image_count = len(image_files)
    if play_mode == "fps":
        fps = mode_arg
    elif play_mode == "time":
        fps = image_count / mode_arg
    else:
        raise ValueError("Invalid play mode")

    cv2.namedWindow("Image Sequence", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image Sequence", width, height)

    for image in image_files:
        img = cv2.imread(image)
        if img is not None:
            cv2.imshow("Image Sequence", img)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == 27:
                break
        else:
            print(f"Error loading image {image}")
    cv2.destroyAllWindows()


def main():
    image_path = "/mnt/data2T/junyuan/eye-tracking/outputs/50time_50segment_causal_linear_max200_user1_left_session_1_0_1"  # 设置图片所在的文件夹路径
    play_images(
        image_path=image_path,
        height=260,
        width=346*2,
        play_mode="fps",
        mode_arg=25,
    )


if __name__ == "__main__":
    main()
