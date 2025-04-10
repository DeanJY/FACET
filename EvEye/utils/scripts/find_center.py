import os
import cv2
from pathlib import Path
from tqdm import tqdm
from EvEye.utils.visualization.visualization import save_image


image_path = "/mnt/data2T/junyuan/eye-tracking/outputs/OutputGroundTruth"
image_output_path = "/mnt/data2T/junyuan/eye-tracking/outputs/image_with_center"


def find_center(image):
    if len(image.shape) == 2:  # 图像是灰度图
        gray = image
    elif len(image.shape) == 3 and image.shape[2] == 3:  # 图像是BGR
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Unsupported image format")

    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = contours[0]
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy
    return None, None


def write_centers(image_path):
    center_txt = f"{image_path}/centers.txt"
    with open(center_txt, "w") as f:
        for filename in tqdm(sorted(os.listdir(image_path)), desc="Processing:"):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                full_image_path = os.path.join(image_path, filename)
                image = cv2.imread(full_image_path)
                if image is not None:
                    cx, cy = find_center(image)
                    if cx is not None:
                        f.write(f"{filename},{cx},{cy},0\n")
                    else:
                        f.write(f"{filename},No contour found\n")
                else:
                    f.write(f"{filename},File cannot be opened or read\n")
    print("Center points saved in:", center_txt)


def draw_centers(image_path, center_txt, image_output_path):
    os.makedirs(image_output_path, exist_ok=True)
    with open(center_txt, "r") as center_txt:
        lines = center_txt.readlines()
    for line in tqdm(lines, desc="Processing:"):
        filename, x, y = line.strip().split(",")
        x = int(x.strip())
        y = int(y.strip())
        image_path_full = os.path.join(image_path, filename)
        image = cv2.imread(image_path_full)
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        save_image(image, f"{image_output_path}/{filename}_with_center.png", "color")
    print("Images with center points saved in:", image_output_path)


def main():
    write_centers(image_path, center_txt)
    # draw_centers(image_path, center_txt, image_output_path)


if __name__ == "__main__":
    main()
