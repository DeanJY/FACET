import torch
import torch.nn as nn
import timm

WEIGHT_PATH = "/mnt/data2T/junyuan/el-net/DeanNet/Weights/resnet18_a1_0-d63eafa0.pth"


class ResNet18Backbone(nn.Module):
    def __init__(self, model_path=WEIGHT_PATH):
        super(ResNet18Backbone, self).__init__()

        self.backbone = timm.create_model(
            model_name='resnet18',
            pretrained=True,
            pretrained_cfg_overlay=dict(file=model_path),
        )

        self.backbone.conv1 = nn.Conv2d(
            2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )

        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

    def forward(self, x):
        x = self.backbone(x)
        return x


def main():
    model = ResNet18Backbone(model_path=WEIGHT_PATH)

    # 打印模型结构（可选）
    print(model)

    # 创建一个随机的双通道图像输入（batch_size=1, channels=2, height=256, width=256）
    input_tensor = torch.randn(1, 2, 224, 224)

    # 前向传播
    output = model(input_tensor)

    # 打印输出特征图的形状
    print("输出特征图的形状:", output.shape)


if __name__ == '__main__':
    main()
