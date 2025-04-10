import numpy as np
import click
import torch
import cv2
import copy
import natsort

from tqdm import tqdm
from pathlib import Path
from EvEye.utils.visualization.visualization import *
from EvEye.utils.scripts.process_model_output import process_model_output
from EvEye.utils.scripts.load_config import load_config
from EvEye.dataset.dataset_factory import make_dataloader
from EvEye.model.model_factory import make_model
from EvEye.utils.scripts.find_center import find_center, write_centers

config_path = "OutputGroundTruth.yaml"
folder_path = "/mnt/data2T/junyuan/eye-tracking/EV_Eye_dataset/raw_data/Data_davis"


def find_image_folders(base_path):
    folders_with_images = []
    # for user_id in range(1, 49):
    for user_id in range(45, 46):
        user_folder = f"{base_path}/user{user_id}"
        # for side in ["left", "right"]:
        for side in ["left"]:
            side_folder = f"{user_folder}/{side}"
            # 检查所有session目录
            session_patterns = [
                # "session_1_0_1",
                # "session_1_0_2",
                # "session_2_0_1",
                "session_2_0_2",
            ]
            for session in session_patterns:
                session_folder = f"{side_folder}/{session}/frames"
                folders_with_images.append(session_folder)

    return natsort.natsorted(folders_with_images)


def output_groundtruth(folder_path, config, model):
    if not Path(folder_path).exists():
        return
    local_config = copy.deepcopy(config)
    local_config["dataloader"]["test"]["dataset"]["dataset_path"] = folder_path
    test_dataloader = make_dataloader(local_config["dataloader"]["test"])
    dataset = test_dataloader.dataset
    center_txt = f"{folder_path}/centers.txt"
    with open(center_txt, "w") as f:
        for data_index, data in enumerate(
            tqdm(dataset, total=len(dataset), desc=f"Writing {folder_path} ...")
        ):
            if data_index >= len(dataset):
                break
            image, origin_image, image_name = data
            image = image.unsqueeze(0)
            with torch.no_grad():
                output = model(image.cuda())
            mask = process_model_output(output, use_softmax=True)
            mask = mask.detach().cpu().numpy()
            mask = cv2.resize(
                mask,
                (origin_image.shape[1], origin_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            if mask is not None:
                mask = draw_label(mask)
                cx, cy = find_center(mask)
                if cx is not None:
                    f.write(f"{image_name},{cx},{cy},0\n")
                else:
                    f.write(f"{image_name},No contour found\n")
            else:
                f.write(f"{image_name},File cannot be opened or read\n")


def main(config_path, folder_path):
    config = load_config(config_path)
    model = make_model(config["model"])
    model.load_state_dict(
        torch.load(config["test"]["ckpt_path"], map_location="cuda:0")["state_dict"]
    )
    model = model.cuda().eval()

    image_folders = find_image_folders(folder_path)

    for folder_path in image_folders:
        output_groundtruth(folder_path, config, model)


# def main(
#     config: str,
#     output_path: str,
#     num: int,
# ) -> None:

#     torch.set_float32_matmul_precision("medium")
#     output_path = Path(output_path)
#     output_path.mkdir(parents=True, exist_ok=True)
#     config = load_config(config)

#     model = make_model(config["model"])
#     model.load_state_dict(
#         torch.load(config["test"]["ckpt_path"], map_location="cuda:0")["state_dict"]
#     )
#     model = model.cuda()
#     model.eval()

#     test_dataloader = make_dataloader(config["dataloader"]["test"])
#     dataset = test_dataloader.dataset
#     iterations = min(len(dataset), num) if num is not None else len(dataset)
#     for data_index, data in enumerate(
#         tqdm(dataset, total=iterations, desc="Saving images...")
#     ):
#         if data_index >= iterations:
#             break
#         image, origin_image, image_name = data  # image.shape: [1, 260, 346]
#         image = image.unsqueeze(0)  # image.shape: [1, 1, 260, 346]
#         with torch.no_grad():
#             output = model(image.cuda())  # shape: [1, 2, 260, 346]
#         mask = process_model_output(output, use_softmax=True)  # shape: [1, 260, 346]
#         mask = mask.detach().cpu().numpy()  # shape: [260, 346]
#         mask = cv2.resize(
#             mask,
#             (origin_image.shape[1], origin_image.shape[0]),
#             interpolation=cv2.INTER_NEAREST,
#         )
#         cx, cy = find_center(mask)

#         image_name = f"{output_path}/{image_name}.png"
#         save_image(mask_image, image_name, BGR2RGB=True)


if __name__ == "__main__":
    main(config_path, folder_path)
