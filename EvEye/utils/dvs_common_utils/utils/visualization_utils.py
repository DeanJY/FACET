import numpy as np
import os
import cv2

def event_to_bluered(
    fs: np.ndarray
) -> np.ndarray:
    fs_mean = fs.mean(axis=2)
    rgb = np.ones(fs.shape)
    center = np.median(fs)
    mask_b = fs_mean >= (center + 1)
    mask_r = fs_mean <= (center - 1)
    rgb[mask_r, 1:] = 0
    rgb[mask_b, :2] = 0
    return (rgb*255).astype(np.uint8)

def framestack_to_bluered_with_rgb(
    frame_stack: np.ndarray, 
    rgb: np.ndarray, 
    alpha: float = 0.4
) -> np.ndarray:
    fs_mean = np.mean(frame_stack, axis=-1)
    center = np.median(frame_stack)
    mask_b = fs_mean >= (center + 1)
    mask_r = fs_mean <= (center - 1)
    rgb = rgb.astype(np.int32)
    rgb[mask_r, 1:] = rgb[mask_r, 1:] + alpha * (0 - rgb[mask_r, 1:])
    rgb[mask_r, 0] = rgb[mask_r, 0] + alpha * (255 - rgb[mask_r, 0])
    rgb[mask_b, :2] = rgb[mask_b, :2] + alpha * (0 - rgb[mask_b, :2])
    rgb[mask_b, 2] = rgb[mask_b, 2] + alpha * (255 - rgb[mask_b, 2])
    
    return rgb.astype(np.uint8)

def get_video(
    frame_rate: int,
    figure_dir: str = "/workspace/EventOCR/outputs/figures",
    output_path: str = "/workspace/EventOCR/outputs/video/test_video.mp4"
) -> None:
    figure_list = os.listdir(figure_dir)
    figure_list.sort(key=lambda x: int(x.split('.')[0]))
    frame = cv2.imread(figure_dir + '/' + figure_list[0])
    width, height = frame.shape[1], frame.shape[0]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

    for figure in figure_list:
        video.write(cv2.imread(figure_dir + '/' + figure))

    cv2.destroyAllWindows()
    video.release()
    print(f"{output_path} video generation finished.")