import torch
import torch.nn as nn
import math
import numpy as np

# HEAD_DICT = {'hm': 1, 'ab': 2, 'ang': 1, 'trig': 2, 'reg': 2, 'mask': 1}
HEAD_DICT = {"hm": 1, "ab": 2, "trig": 2, "reg": 2, "mask": 1}


class EPHead(nn.Module):
    def __init__(self, in_channels, head_conv=256, head_dict=HEAD_DICT):
        super(EPHead, self).__init__()
        self.heads = head_dict
        for head in self.heads:
            classes = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    head_conv,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    head_conv,
                    classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                ),
            )
            if head == "hm":
                fc[-1].bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        outputs = {}
        for head in self.heads:
            outputs[head] = self.__getattr__(head)(x)
        return outputs


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def main():
    def test_head_module():
        # 输入特征图的通道数
        in_channels = 128
        # 输出头中第一个卷积层的通道数
        head_conv = 256
        # 创建 Head 模块实例
        head_module = EPHead(in_channels, head_conv, HEAD_DICT)

        # 打印 Head 模块结构
        print(head_module)

        # 创建一个随机输入张量，形状为 (batch_size, channels, height, width)
        input_tensor = torch.randn(32, in_channels, 64, 64)

        # 前向传播
        output = head_module(input_tensor)

        # 打印每个输出头的输出形状
        for head in output:
            print(f"{head} output shape: {output[head].shape}")

        # # 打印每个输出头的最后一层卷积层的权重
        # for head in HEAD_DICT:
        #     layer = head_module.__getattr__(head)[-1]  # 获取最后一层卷积层
        #     print(f"{head} head last layer weights: {layer.weight.data}")

    # 运行测试函数
    test_head_module()


if __name__ == "__main__":
    main()
