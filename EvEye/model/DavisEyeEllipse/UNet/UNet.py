import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from torchmetrics import JaccardIndex

import lightning
from lightning import LightningModule
from lightning.pytorch.utilities.types import (
    STEP_OUTPUT,
    LRSchedulerTypeUnion,
    OptimizerLRScheduler,
)

from timm.scheduler.step_lr import StepLRScheduler
from typing import Any

from EvEye.model.DavisEyeEllipse.UNet.Metric import *

"""
Unet model for semantic segmentation
Args:
    n_channels: number of input channels
    n_classes: number of output classes
    bilinear: whether to use bilinear interpolation or transposed convolutions for upsampling
Returns:
    logits: output of the model
"""


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(
                in_channels, out_channels, mid_channels=in_channels // 2
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=2,
                stride=2,
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            input=x1,
            pad=[diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(lightning.LightningModule):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.iou = JaccardIndex(task="multiclass", num_classes=n_classes)

        self.validation_outputs = []

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def _log(self, name, metric):
        self.log(name, metric, on_step=False, on_epoch=True, prog_bar=True)

    def set_optimizer_config(self, learning_rate: float, weight_decay: float):
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay

    def lr_scheduler_step(
        self, scheduler: LRSchedulerTypeUnion, metric: Any | None  # 指定学习率调度器  #
    ) -> None:
        scheduler.step(epoch=self.current_epoch)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self._learning_rate,  # 学习率
            weight_decay=self._weight_decay,
        )
        scheduler = StepLRScheduler(
            optimizer, decay_t=10, decay_rate=0.7, warmup_lr_init=1e-5, warmup_t=5
        )
        super().configure_optimizers()  # 这个函数是必须调用的
        return dict(optimizer=optimizer, lr_scheduler=scheduler)

    def training_step(self, batch, batch_idx):
        close = batch["close"]
        valid_mask = close == 0
        image = batch["image"][valid_mask]
        mask = batch["mask"][valid_mask]
        logits = self(image)
        loss = F.cross_entropy(logits, mask.long())
        self._log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        close = batch["close"]
        valid_mask = close == 0
        image = batch["image"][valid_mask]
        mask = batch["mask"][valid_mask]
        logits = self(image)
        loss = F.cross_entropy(logits, mask.long())
        preds = torch.argmax(logits, dim=1)

        metrics = {
            "val_loss": loss,
            "val_p10_acc": p_acc(preds, mask, 10),
            "val_p5_acc": p_acc(preds, mask, 5),
            "val_p3_acc": p_acc(preds, mask, 3),
            "val_p1_acc": p_acc(preds, mask, 1),
            "val_mean_distance": cal_mean_distance(preds, mask),
            "val_IoU": cal_batch_iou(preds, mask),
        }

        if any(torch.isnan(tensor) for tensor in metrics.values()):
            return None

        self.validation_outputs.append(metrics)
        return metrics

    def on_validation_epoch_end(self):
        results = self.validation_outputs
        if not results:
            print("No validation results to process.")
            return

        metrics = {
            key: torch.stack([x[key] for x in results]).mean() for key in results[0]
        }
        for key, value in metrics.items():
            self._log(key, value)
        self.validation_outputs.clear()


def main():
    model = UNet(n_channels=1, n_classes=2, bilinear=True)
    print(model)
    input_tensor = torch.rand(32, 1, 256, 256)
    output = model(input_tensor)
    print(output.shape)


if __name__ == "__main__":
    main()
