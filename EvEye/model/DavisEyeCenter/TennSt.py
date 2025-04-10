import debugpy

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from typing import Any
import cv2
import time

import lightning
from lightning import LightningModule
from lightning.pytorch.utilities.types import (
    STEP_OUTPUT,
    LRSchedulerTypeUnion,
    OptimizerLRScheduler,
)
from timm.scheduler.step_lr import StepLRScheduler

from EvEye.dataset.DavisEyeCenter.losses import *
from functools import partial
import warnings
from EvEye.dataset.DavisEyeCenter.losses import process_detector_prediction

warnings.formatwarning = (
    lambda message, category, filename, lineno, line=None: f"{category.__name__}: {message}\n"
)


class ActivateLayer(nn.Module):
    """A simple activation layer that uses ReLU as the activation function."""

    def __init__(self):
        super().__init__()
        self.act_layer = nn.ReLU()

    def forward(self, x):
        return self.act_layer(x)


class BatchNormBlock(nn.Module):
    """A simple batch normalization block that uses BatchNorm3d as the normalization layer."""

    def __init__(self, features):
        super().__init__()
        self.bn_block = nn.Sequential(nn.BatchNorm3d(features), ActivateLayer())

    def forward(self, x):
        return self.bn_block(x)


class CausalGroupNormBlock(nn.GroupNorm):
    """A GroupNorm that does not use temporal statistics, to ensure causality"""

    def __init__(self, num_groups, num_channels, **kwargs):
        super().__init__(num_groups, num_channels, **kwargs)

    def forward(self, input):
        x = input.moveaxis(1, 2)  # (B, T, C, H, W)
        x_shape = x.shape
        x = x.flatten(0, 1)  # (B * T, C, H, W)
        x = super().forward(x).reshape(x_shape)
        return x.moveaxis(1, 2)  # (B, C, T, H, W)


class GroupNormBlock(nn.Module):
    """A simple group normalization block that uses GroupNorm as the normalization layer."""

    def __init__(self, features):
        super().__init__()
        self.gn_block = nn.Sequential(
            CausalGroupNormBlock(4, features), ActivateLayer()
        )

    def forward(self, x):
        return self.gn_block(x)


class PointWiseConv(nn.Module):
    """A simple pointwise convolution block that uses Conv3d as the convolution layer."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pw_block = nn.Conv3d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        return self.pw_block(x)


class SpatialBlock(nn.Module):  # 定义空间处理块的类，继承自 nn.Module
    def __init__(
        self,
        in_channels,  # 输入通道数
        out_channels,  # 输出通道数
        depthwise=False,  # 是否使用深度可分卷积
        kernel_size=1,  # 卷积核大小
        full_conv3d=False,  # 是否使用完整的3D卷积
        norms="mixed",  # 归一化类型，可以是混合的或者全部是组归一化
    ):
        super().__init__()  # 初始化父类
        kernel = (kernel_size, 3, 3)  # 定义卷积核尺寸
        self.kernel_size = kernel_size  # 保存卷积核大小
        self.full_conv3d = full_conv3d  # 保存是否使用完整的3D卷积
        self.norms = norms  # 保存归一化类型
        self.streaming_mode = False  # 流式处理模式，默认关闭
        self.fifo = None  # 用于流式处理的先进先出缓冲区

        if self.norms == "all_gn":  # 如果归一化类型为全部使用组归一化
            norm_block = GroupNormBlock  # 使用组归一化块
        else:
            norm_block = BatchNormBlock  # 否则使用批量归一化块

        if depthwise:  # 如果使用深度可分卷积
            self.block = nn.Sequential(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel,
                    stride=(1, 2, 2),
                    padding=(0, 1, 1),
                    groups=in_channels,
                    bias=False,
                ),  # 深度可分卷积层
                norm_block(in_channels),  # 归一化层
                PointWiseConv(in_channels, out_channels),  # 逐点卷积，变换通道数
                norm_block(out_channels),  # 再次归一化
            )

        else:  # 如果不使用深度可分卷积
            self.block = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels, kernel, (1, 2, 2), (0, 1, 1), bias=False
                ),  # 常规卷积层
                norm_block(out_channels),  # 归一化层
            )

    def streaming(self, enabled=True):  # 设置是否启用流式处理模式
        if enabled:
            assert (
                not self.training
            ), "Can only use streaming mode during evaluation."  # 断言确保只在评估模式下启用流式处理
        self.streaming_mode = enabled  # 保存流式处理模式状态

    def reset_memory(self):  # 重置内存或缓冲区
        self.fifo = None  # 清空 FIFO 缓冲区

    def forward(self, input):  # 定义前向传播过程
        if self.full_conv3d:  # 如果使用完整的3D卷积
            if self.streaming_mode:  # 如果启用流式处理
                return self._streaming_forward(input)  # 执行流式前向传播
            input = F.pad(
                input, (0, 0, 0, 0, self.kernel_size - 1, 0)
            )  # 对输入数据进行填充
            return self.block(input)  # 通过卷积块处理数据
        else:
            return self.block(input)  # 直接通过卷积块处理数据

    def _streaming_forward(self, input):  # 定义流式前向传播过程
        if self.fifo is None:  # 如果 FIFO 缓冲区未初始化
            self.fifo = torch.zeros(
                *input.shape[:2], self.kernel_size, *input.shape[3:]
            ).type_as(
                input
            )  # 初始化 FIFO 缓冲区
        self.fifo = torch.cat([self.fifo[:, :, 1:], input], dim=2)  # 更新 FIFO 缓冲区
        return self.block(self.fifo)  # 通过卷积块处理 FIFO 中的数据


class TemporalBlock(nn.Module):  # 定义时间处理块的类，继承自 nn.Module
    def __init__(
        self,
        in_channels,  # 输入通道数
        out_channels,  # 输出通道数
        kernel_size=3,  # 卷积核大小，默认为3
        depthwise=False,  # 是否使用深度可分卷积
        full_conv3d=False,  # 是否使用完整的3D卷积
        norms="mixed",  # 归一化类型，可以是混合的、全部批量归一化或全部组归一化
    ):
        super().__init__()  # 初始化父类
        assert out_channels % 4 == 0  # 确保输出通道数是4的倍数，这是组归一化工作所需
        self.kernel_size = kernel_size  # 保存卷积核大小
        self.depthwise = depthwise  # 保存是否使用深度可分卷积
        self.norms = norms  # 保存归一化类型
        kernel = (
            (kernel_size, 3, 3) if full_conv3d else (kernel_size, 1, 1)
        )  # 根据是否全3D卷积选择卷积核形状

        self.streaming_mode = False  # 流式处理模式，默认关闭
        self.fifo = None  # 用于流式处理的先进先出缓冲区

        if self.norms == "mixed":  # 如果归一化类型为混合
            norm1_block = BatchNormBlock  # 第一层归一化使用批量归一化
            norm2_block = GroupNormBlock  # 第二层归一化使用组归一化
        elif self.norms == "all_bn":  # 如果归一化类型全部为批量归一化
            norm1_block = BatchNormBlock
            norm2_block = BatchNormBlock
        elif self.norms == "all_gn":  # 如果归一化类型全部为组归一化
            norm1_block = GroupNormBlock
            norm2_block = GroupNormBlock

        if depthwise:  # 如果使用深度可分卷积
            self.block = nn.Sequential(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel,
                    groups=in_channels,
                    bias=False,
                ),  # 深度可分卷积层
                norm1_block(in_channels),  # 第一层归一化
                PointWiseConv(in_channels, out_channels),  # 逐点卷积，变换通道数
                norm2_block(out_channels),  # 第二层归一化
            )

        else:  # 如果不使用深度可分卷积
            self.block = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel, bias=False),  # 常规卷积层
                norm2_block(out_channels),  # 归一化层
            )

    def streaming(self, enabled=True):  # 设置是否启用流式处理模式
        if enabled:
            assert (
                not self.training
            ), "Can only use streaming mode during evaluation."  # 断言确保只在评估模式下启用流式处理
        self.streaming_mode = enabled  # 保存流式处理模式状态

    def reset_memory(self):  # 重置内存或缓冲区
        self.fifo = None  # 清空 FIFO 缓冲区

    def forward(self, input):  # 定义前向传播过程
        if self.streaming_mode:  # 检查是否启用了流式处理模式
            return self._streaming_forward(
                input
            )  # 如果启用了流式处理，则调用流式前向传播方法处理数据

        # 如果没有启用流式处理，对输入数据进行处理以适应卷积操作
        input = F.pad(
            input, (0, 0, 0, 0, self.kernel_size - 1, 0)
        )  # 对输入数据进行填充，主要是在时间维度上进行填充，以避免卷积时数据丢失
        # 参数解释：
        # (0, 0, 0, 0, self.kernel_size - 1, 0) 中的第五个参数self.kernel_size - 1是填充的核心，它在时间维度的开始处添加了self.kernel_size - 1个0，
        # 这样做是为了确保当卷积核滑动时，可以覆盖足够的时间范围，从而保持时间维度的连续性和完整性。

        return self.block(
            input
        )  # 使用定义好的卷积块对处理后的输入数据进行卷积操作，然后返回结果
        # self.block 是一个包含了一系列卷积层和可能的归一化层的模块（如 nn.Sequential 容器），
        # 它负责执行实际的数据处理任务，如特征提取等。

    def _streaming_forward(self, input):  # 定义流式前向传播过程
        if self.fifo is None:  # 如果 FIFO 缓冲区未初始化
            self.fifo = torch.zeros(
                *input.shape[:2], self.kernel_size, *input.shape[3:]
            ).type_as(
                input
            )  # 初始化 FIFO 缓冲区
        self.fifo = torch.cat([self.fifo[:, :, 1:], input], dim=2)  # 更新 FIFO 缓冲区
        return self.block(self.fifo)  # 通过卷积块处理 FIFO 中的数据


class TennSt(lightning.LightningModule):
    # TennSt类，一个继承自PyTorch的nn.Module的网络模块，用于处理时空数据。
    def __init__(
        self,
        channels,
        t_kernel_size,
        n_depthwise_layers,
        detector_head,
        detector_depthwise,
        full_conv3d=False,
        norms="mixed",
        activity_regularization=0,
    ):
        super().__init__()

        self.loss_fn = Losses(detector_head, activity_regularization, self)
        self.metric_p1fn = partial(p_acc, tolerance=1, detector_head=detector_head)
        self.metric_p3fn = partial(p_acc, tolerance=3, detector_head=detector_head)
        self.metric_p5fn = partial(p_acc, tolerance=5, detector_head=detector_head)
        self.metric_p10fn = partial(p_acc, tolerance=10, detector_head=detector_head)

        self.detector = detector_head
        depthwises = [False] * (10 - n_depthwise_layers) + [True] * n_depthwise_layers
        temporals = [True, False] * 5

        self.backbone = nn.Sequential()
        for i in range(len(depthwises)):
            in_channels, out_channels = (
                channels[i],
                channels[i + 1],
            )
            depthwise = depthwises[i]
            temporal = temporals[i]

            if temporal:
                self.backbone.append(
                    TemporalBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=t_kernel_size,
                        depthwise=depthwise,
                        full_conv3d=full_conv3d,
                        norms=norms,
                    )
                )
            else:
                self.backbone.append(
                    SpatialBlock(
                        in_channels,
                        out_channels,
                        depthwise=depthwise,
                        full_conv3d=full_conv3d,
                        kernel_size=t_kernel_size if full_conv3d else 1,
                        norms=norms,
                    )
                )

        if detector_head:
            self.head = nn.Sequential(
                TemporalBlock(
                    channels[-1],
                    channels[-1],
                    t_kernel_size,
                    depthwise=detector_depthwise,
                ),
                nn.Conv3d(channels[-1], channels[-1], (1, 3, 3), (1, 1, 1), (0, 1, 1)),
                ActivateLayer(),
                nn.Conv3d(channels[-1], 3, 1),
            )
        else:
            self.head = nn.Sequential(
                nn.Conv1d(channels[-1], channels[-1], 1),
                ActivateLayer(),
                nn.Conv1d(channels[-1], 2, 1),
            )

    @staticmethod
    def streaming_inference(model, frames):
        model.eval()
        model.streaming()
        model.reset_memory()

        predictions = []
        inference_times = []
        with torch.inference_mode():
            for frame_id in range(frames.shape[2]):  # stream the frames to the model
                start_time = time.time()
                prediction = model(frames[:, :, [frame_id]])
                end_time = time.time()
                inference_times.append(end_time - start_time)
                predictions.append(prediction)

        predictions = torch.cat(predictions, dim=2)
        return predictions, inference_times

    def streaming(self, enabled=True):
        if enabled:
            warnings.warn(
                "You have enabled the streaming mode of the network. It is expected, but not checked, that the input will be of shape (batch, 1, H, W)."
            )
        for name, module in self.named_modules():
            if name and hasattr(module, "streaming"):
                module.streaming(enabled)

    def reset_memory(self):
        for name, module in self.named_modules():
            if name and hasattr(module, "reset_memory"):
                module.reset_memory()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.detector:
            return self.head((self.backbone(input)))
        else:
            return self.head(self.backbone(input).mean((-2, -1)))

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
        event, center, openness = batch
        height, width = event.shape[-2], event.shape[-1]
        pred = self(event)
        loss = self.loss_fn(pred, center, openness)
        p1_acc, _, _ = self.metric_p1fn(pred, center, openness, height, width)
        p3_acc, _, _ = self.metric_p3fn(pred, center, openness, height, width)
        p5_acc, _, _ = self.metric_p5fn(pred, center, openness, height, width)
        p10_acc, metric_noblinks, distance = self.metric_p10fn(
            pred, center, openness, height, width
        )
        self._log("train_loss", loss)
        self._log("train_p1_acc", p1_acc)
        self._log("train_p3_acc", p3_acc)
        self._log("train_p5_acc", p5_acc)
        self._log("train_p10_acc", p10_acc)
        self._log("train_metric_noblinks", metric_noblinks)
        self._log("train_distance", distance)
        return loss

    def validation_step(self, batch, batch_idx):
        event, center, openness = batch
        height, width = event.shape[-2], event.shape[-1]
        pred = self(event)
        loss = self.loss_fn(pred, center, openness)
        p1_acc, _, _ = self.metric_p1fn(pred, center, openness, 64, 64)
        p3_acc, _, _ = self.metric_p3fn(pred, center, openness, 64, 64)
        p5_acc, _, _ = self.metric_p5fn(pred, center, openness, 64, 64)
        p10_acc, metric_noblinks, distance = self.metric_p10fn(
            pred, center, openness, 64, 64
        )
        pred_frame_list = self.visualize(event, center, pred)
        # self.logger.experiment.add_image()
        self._log("val_loss", loss)
        self._log("val_p1_acc", p1_acc)
        self._log("val_p3_acc", p3_acc)
        self._log("val_p5_acc", p5_acc)
        self._log("val_p10_acc", p10_acc)
        self._log("val_metric_noblinks", metric_noblinks)
        self._log("val_distance", distance)
        for index in range(len(pred_frame_list)):
            self.logger.experiment.add_image(
                f"pred_frames_val/pred_frame_{index:03}",
                pred_frame_list[index],
                dataformats="HWC",
                global_step=self.global_step,
            )

    def on_train_epoch_start(self) -> None:
        return super().on_train_epoch_start()

    def visualize(self, event, center, pred):
        event, center, pred = event.clone(), center.clone(), pred.clone()
        event = event[0]
        event = event.permute(1, 0, 2, 3)
        event = event.detach().cpu().numpy().astype(np.int32)
        t, n, h, w = event.shape

        center = center[0]
        center = center.permute(1, 0)
        center[:, 1] *= h
        center[:, 0] *= w
        center = center.detach().cpu().numpy().astype(np.int32)

        pred = process_detector_prediction(pred)
        pred = pred[0]
        pred = pred.permute(1, 0)
        pred[:, 1] *= h
        pred[:, 0] *= w
        pred = pred.detach().cpu().numpy().astype(np.int32)

        frame_list = []
        for i in range(t):
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            off_frame_stack = event[i, 0]
            on_frame_stack = event[i, 1]
            center_x = center[i, 0]
            center_y = center[i, 1]
            pred_x = pred[i, 0]
            pred_y = pred[i, 1]
            canvas[off_frame_stack > 20] = [0, 0, 255]
            canvas[on_frame_stack > 20] = [255, 0, 0]
            canvas = cv2.circle(canvas, (center_x, center_y), 3, (255, 255, 255), -1)
            canvas = cv2.circle(canvas, (pred_x, pred_y), 3, (0, 255, 0), -1)
            frame_list.append(canvas)
        return frame_list


def main():
    input = torch.randn(32, 2, 50, 96, 128)
    # weights = torch.load()
    model = TennSt(
        channels=[2, 8, 16, 32, 48, 64, 80, 96, 112, 128, 256],
        t_kernel_size=5,
        n_depthwise_layers=4,
        detector_head=True,
        detector_depthwise=True,
        full_conv3d=False,
        norms="mixed",
    )
    output = model(input)
    print(output.shape)
    # print(model)


if __name__ == "__main__":
    main()
