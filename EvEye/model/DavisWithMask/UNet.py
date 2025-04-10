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

from EvEye.model.DavisWithMask.validation import get_auc

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
        images, masks = batch
        logits = self(images)
        masks = masks.long()
        loss = F.cross_entropy(logits, masks)
        self.log("train_loss", loss)
        return dict(loss=loss)

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = F.cross_entropy(logits, masks)
        preds = torch.argmax(logits, dim=1)
        iou = self.iou(preds, masks)
        absolute_auc = get_auc(preds, masks)
        eye_metrics = {
            "val/val_loss": loss,
            "val/val_iou": iou,
            "val/absolute_auc": absolute_auc,
        }
        self.log_dict(eye_metrics, on_step=False, on_epoch=True, logger=True)


# Example of how to setup the data loaders
def main():
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    model = UNet(n_channels=3, n_classes=2)
    if isinstance(model, LightningModule):
        print("model 是 LightningModule 的一个实例。")
    else:
        print("model 不是 LightningModule 的一个实例。")
    # trainer = pl.Trainer(max_epochs=25)
    # trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
