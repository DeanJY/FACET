import numpy as np
import click
import torch
import cv2

from tqdm import tqdm
from pathlib import Path
from EvEye.utils.visualization.visualization import *
from EvEye.utils.scripts.process_model_output import process_model_output
from EvEye.utils.scripts.load_config import load_config
from EvEye.dataset.dataset_factory import make_dataloader
from EvEye.model.model_factory import make_model


@click.command()
@click.option("--config", "-c", type=str, default="TestDavisEyeCenter_TennSt.yaml")
@click.option(
    "--output_path",
    "-o",
    type=str,
    default="/mnt/data2T/junyuan/eye-tracking/outputs/OutputGroundTruth",
)
@click.option("--num", default=None, help="An optional integer parameter.")
def main(
    config: str,
    output_path: str,
    num: int,
) -> None:
    """
    Test the model and save the output images to the output_path.

    Args:
        config (str): The path to the configuration file.
        output_path (str): The path to the output directory.
        num (int): The number of images to save.
    Returns:
        None
    """
    torch.set_float32_matmul_precision("medium")
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    config = load_config(config)

    model = make_model(config["model"])
    model.load_state_dict(
        torch.load(config["test"]["ckpt_path"], map_location="cuda:0")["state_dict"]
    )
    # model.load_state_dict(
    #     torch.load(config["val"]["ckpt_path"], map_location="cuda:0")["state_dict"]
    # )
    model = model.cuda()
    model.eval()

    # test_dataloader = make_dataloader(config["dataloader"]["test"])
    test_dataloader = make_dataloader(config["dataloader"]["test"])
    dataset = test_dataloader.dataset
    iterations = min(len(dataset), num) if num is not None else len(dataset)
    for data_index, data in enumerate(
        tqdm(dataset, total=iterations, desc="Saving images...")
    ):
        if data_index >= iterations:
            break
        image, origin_image, image_name = data  # image.shape: [1, 260, 346]
        image = image.unsqueeze(0)  # image.shape: [1, 1, 260, 346]
        with torch.no_grad():
            output = model(image.cuda())  # shape: [1, 2, 260, 346]
        mask = process_model_output(output, use_softmax=True)  # shape: [1, 260, 346]
        mask = mask.detach().cpu().numpy()  # shape: [260, 346]
        mask = cv2.resize(
            mask,
            (origin_image.shape[1], origin_image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        origin_image = convert_to_color(origin_image)
        overlayed_image = overlay_contour(
            origin_image, mask, alpha=0.5, ignore_background=True
        )

        image_name = f"{output_path}/{image_name}.png"
        save_image(overlayed_image, image_name, BGR2RGB=True)


if __name__ == "__main__":
    main()
