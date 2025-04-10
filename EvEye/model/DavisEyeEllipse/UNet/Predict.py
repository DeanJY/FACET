import torch
import numpy as np
import cv2
import os
import time

import onnxruntime as ort
from thop import profile
from pathlib import Path
from tqdm import tqdm
from natsort import natsorted
from EvEye.model.DavisEyeEllipse.UNet.UNet import UNet
from EvEye.utils.visualization.visualization import *


def pre_process(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 256))
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image).float()

    return image


def post_process(output):
    output = torch.softmax(output, dim=1)
    output = output.argmax(dim=1)
    output = output.squeeze(0)
    output = output.cpu().numpy()
    output[output == 1] = 255
    output = cv2.resize(output, (346, 260), interpolation=cv2.INTER_NEAREST)

    return output


def predict_once(model_path, image_path, output_path):
    model = UNet(n_channels=1, n_classes=2)
    model.load_state_dict(torch.load(model_path)["state_dict"])
    model.eval()
    model.cuda()
    input = pre_process(image_path)
    with torch.no_grad():
        output = model(input.cuda())
    mask = post_process(output)
    os.makedirs(output_path, exist_ok=True)
    save_image(mask, f"{output_path}/mask.png")
    print(mask.shape)


def predict(model_path, data_paths, ellipse_paths, output_path, num=None):
    model = UNet(n_channels=1, n_classes=2)
    model.load_state_dict(torch.load(model_path)["state_dict"])
    model.eval()
    model.cuda()
    images = natsorted(list(data_paths.rglob("*.png")))
    ellipses = natsorted(list(ellipse_paths.rglob("*.png")))
    total_frames = len(images) if num is None else min(len(images), num)
    for index in tqdm(range(total_frames)):
        image_path = images[index]
        input = pre_process(image_path)
        with torch.no_grad():
            output = model(input.cuda())
        mask = post_process(output)
        ellipse_path = ellipses[index]
        ellipse = cv2.imread(str(ellipse_path), cv2.IMREAD_GRAYSCALE)

        image = cv2.imread(str(image_path))
        mask_edge = cv2.Canny(mask.astype(np.uint8), 100, 200)
        ellipse_edge = cv2.Canny(ellipse.astype(np.uint8), 100, 200)
        image[mask_edge != 0] = [0, 255, 0]
        image[ellipse_edge != 0] = [255, 255, 255]

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        fitted = False
        for contour in contours:
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                if ellipse[1][0] > 0 and ellipse[1][1] > 0:
                    cv2.ellipse(image, ellipse, (0, 255, 0), 2)
                    center = (int(ellipse[0][0]), int(ellipse[0][1]))
                    cv2.circle(image, center, 2, (0, 255, 0), -1)
                    fitted = True

        if not fitted:
            print(
                f"Frame {index} does not have enough points for ellipse fitting, saving original image."
            )

        os.makedirs(output_path, exist_ok=True)
        save_image(image, f"{output_path}/{index:05}.png")


def predict_pure(model_path, image_paths, output_path, num=None):
    model = UNet(n_channels=1, n_classes=2)
    model.load_state_dict(torch.load(model_path)["state_dict"])
    model.eval()
    model.cuda()
    images = natsorted(list(image_paths.rglob("*.png")))
    total_frames = len(images) if num is None else min(len(images), num)
    for index in tqdm(range(total_frames)):
        image_path = images[index]
        # print(image_path)
        input = pre_process(image_path)
        with torch.no_grad():
            output = model(input.cuda())
        mask = post_process(output)
        os.makedirs(output_path, exist_ok=True)
        save_image(mask, f"{output_path}/{os.path.basename(image_path)}")


def test_inference_time(model_path, device="cuda:0"):
    model = UNet(n_channels=1, n_classes=2)
    model.load_state_dict(torch.load(model_path)["state_dict"])
    model.eval()
    model.to(device)
    input = torch.rand((1, 1, 256, 256), dtype=torch.float32).to(device)
    with torch.no_grad():
        times = []
        for _ in tqdm(range(400)):
            start_time = time.time()
            output = model(input)
            end_time = time.time()
            times.append(end_time - start_time)
    avg_time_heat = sum(times) / len(times)
    print(f"Average inference heat time: {avg_time_heat}")

    with torch.no_grad():
        times = []
        for _ in tqdm(range(200)):
            start_time = time.time()
            output = model(input)
            end_time = time.time()
            times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"Average inference time: {avg_time}")
    return avg_time


def main():
    model_path = "/mnt/data2T/junyuan/eye-tracking/logs/RGBUNetTest/version_0/checkpoints/epoch=57-val_mean_distance=0.4287.ckpt"
    output_path = "/mnt/data2T/junyuan/eye-tracking/video/output/test_results"
    # predict_once(model_path, image_path, output_path)

    image_paths = Path("/mnt/data2T/junyuan/eye-tracking/video/output/3")
    ellipse_paths = Path("/mnt/data2T/junyuan/Datasets/EventUNetDataset/val/label")

    # predict(model_path, image_paths, ellipse_paths, output_path, num=500)
    predict_pure(model_path, image_paths, output_path, num=None)

    # test_inference_time(model_path)
    # model = UNet(n_channels=1, n_classes=2)
    # input = torch.randn((1, 1, 256, 256), dtype=torch.float32)
    # flops, params = profile(model, inputs=(input,))
    # print(f"FLOPs: {flops}")
    # print(f"Total parameters: {params}")


if __name__ == "__main__":
    main()
