import timm
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Any
import lightning
from lightning import LightningModule
from lightning.pytorch.utilities.types import (
    STEP_OUTPUT,
    LRSchedulerTypeUnion,
    OptimizerLRScheduler,
)
from timm.scheduler.step_lr import StepLRScheduler
from EvEye.dataset.DavisEyeEllipse.losses import cal_loss
from torchinfo import summary
from thop import profile
from pprint import pprint


class EllipseMobileNetV3(lightning.LightningModule):
    def __init__(
        self,
        model_name="mobilenetv3_large_100",
        model_path="/mnt/data2T/junyuan/eye-tracking/TimmModels/mobilenetv3_large_100_ra-f55367f5.pth",
        in_channels=2,
        num_classes=6,
    ):
        super(EllipseMobileNetV3, self).__init__()
        self.model = timm.create_model(
            'mobilenetv3_large_100',  # 模型名字
            pretrained=False,  # 是否加载预训练权重
            num_classes=num_classes,  # 调整最后一层的分类个数
            pretrained_cfg_overlay=dict(file=model_path),  # 模型预训练权重.pth地址
            in_chans=in_channels,  # 输入通道数
        )

        # 移除最后的分类头
        self.model.reset_classifier(0)

        # 添加自定义的检测头
        self.detection_head = nn.Sequential(
            nn.Linear(self.model.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.detection_head(x)
        return x

    def _log(self, name, metric):
        self.log(name, metric, on_step=False, on_epoch=True, prog_bar=True)

    def set_optimizer_config(self, learning_rate: float, weight_decay: float):
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay

    def lr_scheduler_step(
        self, scheduler: LRSchedulerTypeUnion, metric: Any | None
    ) -> None:
        scheduler.step(epoch=self.current_epoch)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self._learning_rate,
            weight_decay=self._weight_decay,
        )
        scheduler = StepLRScheduler(
            optimizer, decay_t=10, decay_rate=0.7, warmup_lr_init=1e-5, warmup_t=5
        )
        super().configure_optimizers()
        return dict(optimizer=optimizer, lr_scheduler=scheduler)

    def training_step(self, batch, batch_idx):
        event_frame, label = batch
        pred = self(event_frame)
        loss = cal_loss(pred, label)
        self._log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        event_frame, label = batch
        pred = self(event_frame)
        loss = cal_loss(pred, label)
        self._log("val_loss", loss)
        return loss

    def on_train_epoch_start(self) -> None:
        return super().on_train_epoch_start()
