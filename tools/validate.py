import lightning
import torch
import click

from EvEye.utils.scripts.load_config import load_config
from EvEye.logger.logger_factory import make_logger
from EvEye.callback.callback_factory import make_callbacks
from EvEye.dataset.dataset_factory import make_dataloader
from EvEye.model.model_factory import make_model


@click.command()
@click.option("--config", "-c", type=str, default="MemmapDavisEyeCenter_TennSt.yaml")
def main(config: str) -> None:
    torch.set_float32_matmul_precision("medium")
    config = load_config(config)

    val_dataloader = make_dataloader(config["dataloader"]["val"])

    model_cfg = config["model"]
    model = make_model(model_cfg)

    trainer = lightning.Trainer(
        devices=[2],
        max_epochs=config["train"].get("max_epochs", 50),
        check_val_every_n_epoch=1,
        logger=make_logger(config["logger"]),
        callbacks=make_callbacks(config["callback"]),
    )

    trainer.validate(
        model=model,
        dataloaders=val_dataloader,
        ckpt_path=config["val"].get("ckpt_path"),
    )


if __name__ == "__main__":
    main()
