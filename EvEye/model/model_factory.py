from typing import Any, Type
import torch.nn as nn

from EvEye.model.DavisWithMask.DeepLabV3 import DeepLabV3
from EvEye.model.CitiBike.ConvLSTM import ConvLSTM
from EvEye.model.DavisEyeCenter.TennSt import TennSt
from EvEye.model.DavisEyeEllipse.EPNet.EPNet import EPNet
from EvEye.model.DavisEyeEllipse.ElNet.ElNet import Creat_MyNet
from EvEye.model.DavisEyeEllipse.UNet.UNet import UNet


MODEL_CLASSES: dict[str, Type[nn.Module]] = dict(
    DeepLabV3=DeepLabV3,
    ConvLSTM=ConvLSTM,
    TennSt=TennSt,
    EPNet=EPNet,
    ElNet=Creat_MyNet,
    UNet=UNet,
)

MODEL_TYPE = UNet | DeepLabV3 | ConvLSTM | TennSt | EPNet


def make_model(model_cfg: dict[str, Any]) -> MODEL_TYPE:
    """make model out of configs

    Args:
        model_cfg (dict[str, Any]): config dict
    """
    assert model_cfg["type"] in MODEL_CLASSES.keys()
    return MODEL_CLASSES[model_cfg.pop("type")](**model_cfg)


def main():
    import torch
    import time
    from thop import profile
    from tqdm import tqdm

    # model = UNet(n_channels=1, n_classes=2)
    # input = torch.rand(1, 1, 256, 256)
    # model_path = "/mnt/data2T/junyuan/eye-tracking/logs/EventUNet/version_0/checkpoints/epoch=26-val_mean_distance=1.6679.ckpt"

    # model = UNet(n_channels=1, n_classes=2)
    # input = torch.randn(1, 1, 256, 256)
    # model_path = "/mnt/data2T/junyuan/eye-tracking/logs/RGBUNet/version_0/checkpoints/epoch=27-val_mean_distance=0.3231.ckpt"

    model = Creat_MyNet(
        base_name="dla34",
        heads={"hm": 1, "ab": 2, "ang": 1, "trig": 2, "reg": 2, "mask": 1},
        pretrained=True,
        down_ratio=4,
        final_kernel=1,
        last_level=5,
        head_conv=256,
        out_channel=0,
    )
    input = torch.rand(1, 3, 256, 256)
    model_path = "/mnt/data2T/junyuan/eye-tracking/logs/ElNet_FixedCount5000/version_0/checkpoints/epoch=00-val_mean_distance=0.6273.ckpt"

    # model = TennSt(
    #     channels=[2, 8, 16, 32, 48, 64, 80, 96, 112, 128, 256],
    #     t_kernel_size=5,
    #     n_depthwise_layers=4,
    #     detector_head=True,
    #     detector_depthwise=True,
    #     full_conv3d=False,
    #     norms="mixed",
    #     activity_regularization=0,
    # )
    # input = torch.randn(1, 2, 50, 130, 176)
    # model_path = "/mnt/data2T/junyuan/eye-tracking/weights/96.89%-FixedCount5000-Down-Aug-NoFlip/checkpoints/epochepoch=47-val_p10_accval_p10_acc=0.9689.ckpt"

    # model = EPNet(
    #     input_channels=2,
    #     head_dict={"hm": 1, "ab": 2, "trig": 2, "reg": 2, "mask": 1},
    #     mode="fpn_dw",
    #     loss_weight={
    #         "hm_weight": 1,
    #         "ab_weight": 0.1,
    #         "ang_weight": 0,  # 0.1
    #         "trig_weight": 1,
    #         "reg_weight": 0.1,
    #         "iou_weight": 15,
    #         "mask_weight": 1,
    #     },
    # )
    # input = torch.randn(1, 2, 256, 256)
    # model_path = "/mnt/data2T/junyuan/eye-tracking/logs/EPNet_FixedCount5000_TrigERAugFPNdw/version_0/checkpoints/epoch=53-val_mean_distance=0.2249.ckpt"

    def get_model_size(model, input):
        flops, params = profile(model, inputs=(input,))
        flops_g = flops / 1e9
        params_m = params / 1e6
        print(f"Model name: {model.__class__.__name__}")
        print(f"FLOPs: {flops_g} GFLOPs")
        print(f"Total parameters: {params_m} M")

    # get_model_size(model, input)

    def test_inference_time(model, input, model_path, device="cuda:0"):
        model.load_state_dict(torch.load(model_path)["state_dict"])
        model.eval()
        model.to(device)
        input = input.to(device)

        with torch.no_grad():
            times = []
            for _ in tqdm(range(400)):
                start_time = time.time()
                output = model(input)
                end_time = time.time()
                times.append(end_time - start_time)
        avg_time_heat = sum(times) / len(times)
        print(f"Average inference heat time: {avg_time_heat}")

        avg_times = []
        for _ in tqdm(range(10)):
            with torch.no_grad():
                times = []
                for _ in range(200):
                    start_time = time.time()
                    output = model(input)
                    end_time = time.time()
                    times.append(end_time - start_time)
            avg_time = sum(times) / len(times)
            avg_times.append(avg_time)

        final_avg_time = sum(avg_times) / len(avg_times)
        print(f"Average inference time ms: {final_avg_time * 1000:.4f}ms")

        return final_avg_time

    test_inference_time(model, input, model_path)


if __name__ == "__main__":
    main()
