import click
import pdb

import torch
import lightning
import os


from EvEye.utils.scripts.load_config import load_config
from EvEye.logger.logger_factory import make_logger
from EvEye.callback.callback_factory import make_callbacks
from EvEye.dataset.dataset_factory import make_dataloader
from EvEye.model.model_factory import make_model


@click.command()
@click.option("-c", "--config", type=str, default="DavisEyeEllipse_EPNet.yaml")
def main(config: str) -> None:
    lightning.seed_everything(42)
    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.set_float32_matmul_precision("medium")
    config = load_config(config)

    if os.environ.get("SM_CHANNEL_ROOT"):
        config["dataloader"]["train"]["dataset"]["root_path"] = os.environ[
            "SM_CHANNEL_ROOT"
        ]
        config["dataloader"]["val"]["dataset"]["root_path"] = os.environ[
            "SM_CHANNEL_ROOT"
        ]
    train_dataloader = make_dataloader(config["dataloader"]["train"])
    val_dataloader = make_dataloader(config["dataloader"]["val"])

    torch.multiprocessing.current_process()._children = None

    model_cfg = config["model"]
    model = make_model(model_cfg)
    if "optimizer" in config["train"].keys():
        model.set_optimizer_config(**config["train"]["optimizer"])

    trainer = lightning.Trainer(
        devices=[2],
        max_epochs=config["train"].get("max_epochs", 50),
        check_val_every_n_epoch=config["train"].get("check_val_every_n_epoch", 1),
        logger=make_logger(config["logger"]),
        callbacks=make_callbacks(config["callback"]),
        # num_sanity_val_steps=0,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        # ckpt_path=config["train"].get("ckpt_path"),  # For resuming training
    )


if __name__ == "__main__":
    main()
