import lightning
import torch
import click
import numpy as np
import os

from EvEye.utils.scripts.load_config import load_config
from EvEye.logger.logger_factory import make_logger
from EvEye.callback.callback_factory import make_callbacks
from EvEye.dataset.dataset_factory import make_dataloader
from EvEye.model.model_factory import make_model


@click.command()
@click.option("--config", "-c", type=str, default="MemmapDavisEyeCenter_TennSt.yaml")
@click.option("--num_validations", "-n", type=int, default=10)
def main(config: str, num_validations: int) -> None:
    torch.set_float32_matmul_precision("medium")
    config = load_config(config)

    val_dataloader = make_dataloader(config["dataloader"]["val"])

    model_cfg = config["model"]
    model = make_model(model_cfg)

    trainer = lightning.Trainer(
        devices=[0],
        max_epochs=config["train"].get("max_epochs", 50),
        check_val_every_n_epoch=1,
        logger=make_logger(config["logger"]),
        callbacks=make_callbacks(config["callback"]),
    )

    metrics_list = []

    for _ in range(num_validations):
        metrics = trainer.validate(
            model=model,
            dataloaders=val_dataloader,
            ckpt_path=config["val"].get("ckpt_path"),
        )
        metrics_list.append(metrics)

    # 计算平均值
    avg_metrics = {
        key: np.mean([m[0][key] for m in metrics_list])
        for key in metrics_list[0][0].keys()
    }

    # 打印平均值
    print("Average Metrics over {} validations:".format(num_validations))
    for key, value in avg_metrics.items():
        print(f"{key}: {value}")

    # 获取ckpt文件路径并提取文件名
    ckpt_path = config["val"].get("ckpt_path")
    if ckpt_path:
        parent_dir = os.path.dirname(os.path.dirname(ckpt_path))
        dir_name = os.path.basename(parent_dir)
        output_path = "/mnt/data2T/junyuan/eye-tracking/Results"
        os.makedirs(output_path, exist_ok=True)
        result_file_name = f"{output_path}/{dir_name}.txt"

        # 将结果写入文件
        with open(result_file_name, 'w') as f:
            f.write("Average Metrics over {} validations:\n".format(num_validations))
            for key, value in avg_metrics.items():
                f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    main()
