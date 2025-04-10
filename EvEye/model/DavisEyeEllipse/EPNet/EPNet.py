import torch
import torch.nn as nn
import torch.nn.init as init
import lightning

from typing import Any
from torch.nn import functional as F
from collections import OrderedDict
from lightning.pytorch.utilities.types import (
    STEP_OUTPUT,
    LRSchedulerTypeUnion,
    OptimizerLRScheduler,
)
from functools import partial
from timm.scheduler.step_lr import StepLRScheduler
from EvEye.model.DavisEyeEllipse.EPNet.Backbone.MobileNetV3Backbone import (
    MobileNetV3Backbone,
)
from EvEye.model.DavisEyeEllipse.EPNet.Head.EPHead import EPHead
from EvEye.model.DavisEyeEllipse.EPNet.Loss import *
from EvEye.model.DavisEyeEllipse.EPNet.Predict import *
from EvEye.model.DavisEyeEllipse.EPNet.Metric import *

LOSS_WEIGHT = {
    "hm_weight": 1,
    "ab_weight": 0.1,
    "ang_weight": 0,  # 0.1
    "trig_weight": 1,
    "reg_weight": 0.1,
    "iou_weight": 15,
    "mask_weight": 1,
}
# HEAD_DICT = {"hm": 1, "ab": 2, "ang": 1, "reg": 2, "mask": 1}
HEAD_DICT = {"hm": 1, "ab": 2, "trig": 2, "reg": 2, "mask": 1}


def conv2d(filter_in, filter_out, kernel_size, groups=1, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(
        OrderedDict(
            [
                (
                    "conv",
                    nn.Conv2d(
                        filter_in,
                        filter_out,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=pad,
                        groups=groups,
                        bias=False,
                    ),
                ),
                ("bn", nn.BatchNorm2d(filter_out)),
                ("relu", nn.ReLU6(inplace=True)),
            ]
        )
    )


def conv_dw(filter_in, filter_out, stride=1):
    return nn.Sequential(
        nn.Conv2d(filter_in, filter_in, 3, stride, 1, groups=filter_in, bias=False),
        nn.BatchNorm2d(filter_in),
        nn.ReLU6(inplace=True),
        nn.Conv2d(filter_in, filter_out, 1, 1, 0, bias=False),
        nn.BatchNorm2d(filter_out),
        nn.ReLU6(inplace=True),
    )


def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m


def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),
        conv2d(filters_list[1], filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m


class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList(
            [nn.MaxPool2d(pool_size, 1, pool_size // 2) for pool_size in pool_sizes]
        )

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

    def forward(self, x):
        x = self.upsample(x)
        return x

class Upsample_dw(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample_dw, self).__init__()

        self.upsample = nn.Sequential(
            conv_dw(in_channels, out_channels),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

    def forward(self, x):
        x = self.upsample(x)
        return x

class EPNet(lightning.LightningModule):
    def __init__(
        self,
        input_channels=2,
        head_dict=HEAD_DICT,
        mode="standard",
        loss_weight=LOSS_WEIGHT,
    ):
        super(EPNet, self).__init__()
        self.mode = mode
        self.criterion = CtdetLoss(loss_weight)
        self.backbone = MobileNetV3Backbone(input_channels=input_channels)
        self.head = EPHead(in_channels=64, head_dict=head_dict)
        self.in_filters = self.backbone.in_filters

        if self.mode == "standard":
            self.SPP = SpatialPyramidPooling()
            self.conv_for_P4 = conv2d(self.in_filters[2], 256, 1)
            self.upsample2 = Upsample(256, 128)
            self.conv_for_P3 = conv2d(self.in_filters[1], 128, 1)
            self.upsample3 = Upsample(128, 64)
            self.conv_for_P2 = conv2d(self.in_filters[0], 64, 1)

            self.conv1 = make_three_conv([512, 1024], self.in_filters[3])
            self.conv2 = make_three_conv([512, 1024], 2048)
            self.upsample1 = Upsample(512, 256)
            self.make_five_conv1 = make_five_conv([256, 512], 512)
            self.make_five_conv2 = make_five_conv([128, 256], 256)
            self.make_five_conv3 = make_five_conv([64, 128], 128)

        elif self.mode == "light":
            self.SPP = SpatialPyramidPooling()
            self.conv_for_P4 = conv2d(self.in_filters[2], 256, 1)
            self.upsample2 = Upsample(256, 128)
            self.conv_for_P3 = conv2d(self.in_filters[1], 128, 1)
            self.upsample3 = Upsample(128, 64)
            self.conv_for_P2 = conv2d(self.in_filters[0], 64, 1)

            self.conv1_light = make_three_conv([256, 512], self.in_filters[3])
            self.conv2_light = make_three_conv([256, 512], 1024)
            self.upsample1_light = Upsample(256, 256)
            self.make_three_conv1 = make_three_conv([256, 512], 512)
            self.make_three_conv2 = make_three_conv([128, 256], 256)
            self.make_three_conv3 = make_three_conv([64, 128], 128)

        elif self.mode == "fpn_2d":
            self.conv_for_P5 = conv2d(160, 512, 1)
            self.upsample5 = Upsample(512, 256)
            self.conv_for_P4 = conv2d(112, 256, 1)
            self.upsample4 = Upsample(512, 128)
            self.conv_for_P3 = conv2d(40, 128, 1)
            self.upsample3 = Upsample(256, 64)
            self.conv_for_P2 = conv2d(24, 64, 1)
            self.final_conv = conv2d(128, 64, 1)

        elif self.mode == "fpn_dw":
            self.conv_for_P5 = conv_dw(160, 512)
            self.upsample5 = Upsample_dw(512, 256)
            self.conv_for_P4 = conv_dw(112, 256)
            self.upsample4 = Upsample_dw(512, 128)
            self.conv_for_P3 = conv_dw(40, 128)
            self.upsample3 = Upsample_dw(256, 64)
            self.conv_for_P2 = conv_dw(24, 64)
            self.final_conv = conv_dw(128, 64)


        self.validation_outputs = []

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)

    def forward(self, x):
        if self.mode == "standard":
            # Backbone
            out2, out3, out4, out5 = self.backbone(x)

            # Top-down path
            P5 = self.conv1(out5)
            P5 = self.SPP(P5)
            P5 = self.conv2(P5)

            P5_upsample = self.upsample1(P5)
            P4 = self.conv_for_P4(out4)
            P4 = torch.cat([P4, P5_upsample], dim=1)
            P4 = self.make_five_conv1(P4)

            P4_upsample = self.upsample2(P4)
            P3 = self.conv_for_P3(out3)
            P3 = torch.cat([P3, P4_upsample], dim=1)
            P3 = self.make_five_conv2(P3)

            P3_upsample = self.upsample3(P3)
            P2 = self.conv_for_P2(out2)
            P2 = torch.cat([P2, P3_upsample], dim=1)
            P2 = self.make_five_conv3(P2)

            # Head
            output = self.head(P2)

            return output
        elif self.mode == "light":
            out2, out3, out4, out5 = self.backbone(x)
            P5 = self.conv1_light(out5)
            P5 = self.SPP(P5)
            P5 = self.conv2_light(P5)

            P5_upsample = self.upsample1_light(P5)
            P4 = self.conv_for_P4(out4)
            P4 = torch.cat([P4, P5_upsample], dim=1)
            P4 = self.make_three_conv1(P4)

            P4_upsample = self.upsample2(P4)
            P3 = self.conv_for_P3(out3)
            P3 = torch.cat([P3, P4_upsample], dim=1)
            P3 = self.make_three_conv2(P3)

            P3_upsample = self.upsample3(P3)
            P2 = self.conv_for_P2(out2)
            P2 = torch.cat([P2, P3_upsample], dim=1)
            P2 = self.make_three_conv3(P2)

            # Head
            output = self.head(P2)

            return output

        elif self.mode == "fpn_2d" or self.mode == "fpn_dw":
            out2, out3, out4, out5 = self.backbone(x)
            P5 = self.conv_for_P5(out5)
            P5_upsample = self.upsample5(P5)

            P4 = self.conv_for_P4(out4)
            P4 = torch.cat([P5_upsample, P4], dim=1)
            P4_upsample = self.upsample4(P4)

            P3 = self.conv_for_P3(out3)
            P3 = torch.cat([P4_upsample, P3], dim=1)
            P3_upsample = self.upsample3(P3)

            P2 = self.conv_for_P2(out2)
            P2 = torch.cat([P3_upsample, P2], dim=1)
            P2 = self.final_conv(P2)

            output = self.head(P2)

            return output

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
        input = batch["input"]
        pred = self(input)
        loss, loss_show = self.criterion(pred, batch)

        self._log("train_loss", loss)
        for key, value in loss_show.items():
            self._log(f"train_{key}", value)

        self.logger.experiment.add_image(
            "train_mask", pred["mask"][0], self.global_step
        )

        return loss

    def validation_step(self, batch, batch_idx):
        input = batch["input"]
        center = batch["center"]
        close = batch["close"]
        ellipse = batch["ellipse"]
        pred = self(input)
        dets = post_process(pred)
        loss, _ = self.criterion(pred, batch)
        iou = cal_batch_iou(dets, ellipse, close)
        ap = cal_batch_ap(dets, ellipse, close, iou_thres=0.5, score_threshold=0.5)

        metrics = {
            "val_loss": loss,
            "val_p10_acc": p_acc(dets, center, close, 10),
            "val_p5_acc": p_acc(dets, center, close, 5),
            "val_p3_acc": p_acc(dets, center, close, 3),
            "val_p1_acc": p_acc(dets, center, close, 1),
            "val_mean_distance": cal_mean_distance(dets, center, close),
            "val_IoU": iou,
            "val_AP": ap,
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
    # 创建一个EPNet实例
    model = EPNet()

    # 打印模型结构
    print(model)

    # 创建一个随机输入张量，尺寸为 (32, 2, 256, 256)
    input_tensor = torch.randn(32, 2, 256, 256)

    # 运行前向传播
    output = model(input_tensor)

    # 打印每个输出头的输出形状
    for head in output:
        print(f"{head} output shape: {output[head].shape}")


if __name__ == "__main__":
    main()
