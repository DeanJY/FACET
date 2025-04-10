import torch
import torch.nn.functional as F
import lightning
from torch import nn
from torchvision import models, transforms
from torchmetrics import JaccardIndex

from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from lightning.pytorch.utilities.types import (
    STEP_OUTPUT,
    LRSchedulerTypeUnion,
    OptimizerLRScheduler,
)

from timm.scheduler.step_lr import StepLRScheduler
from typing import Any

from EvEye.model.DavisWithMask.validation import get_auc

"""
DeepLabV3 model for semantic segmentation
Args:
    n_classes: number of output classes
    pretrained: whether to use pretrained weights
Returns:
    logits: output of the model
"""


class DeepLabV3(lightning.LightningModule):
    def __init__(self, n_classes: int, pretrained: bool = True):
        super(DeepLabV3, self).__init__()
        self.n_classes = n_classes

        # Load a pretrained DeepLabV3 model, modify it for n_classes
        self.model = models.segmentation.deeplabv3_resnet50(
            pretrained=pretrained, progress=True
        )
        self.model.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )  # Change the input channel from 3 to 1
        self.model.classifier[4] = nn.Conv2d(
            256, n_classes, kernel_size=(1, 1), stride=(1, 1)
        )  # Replace the classifier head

        self.iou = JaccardIndex(task="multiclass", num_classes=n_classes)

    def forward(self, x):
        return self.model(x)["out"]

    def set_optimizer_config(self, learning_rate: float, weight_decay: float):
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay

    def lr_scheduler_step(
        self, scheduler: LRSchedulerTypeUnion, metric: Any | None
    ) -> None:
        scheduler.step(epoch=self.current_epoch)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self._learning_rate, weight_decay=self._weight_decay
        )
        scheduler = StepLRScheduler(
            optimizer, decay_t=10, decay_rate=0.7, warmup_lr_init=1e-5, warmup_t=5
        )
        return dict(optimizer=optimizer, lr_scheduler=scheduler)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        masks = masks.long()
        loss = F.cross_entropy(logits, masks.long())
        self.log("train_loss", loss, on_step=False, on_epoch=True)
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
    model = DeepLabV3(n_classes=2, pretrained=True)
    # trainer = pl.Trainer(max_epochs=25)
    # trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
