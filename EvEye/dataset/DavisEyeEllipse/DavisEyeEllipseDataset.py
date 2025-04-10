import os
import cv2
import torch
import tonic
import time
import random
import math
import numpy as np
import albumentations as A

from natsort import natsorted
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from tonic import slicers, transforms, functional
from EvEye.utils.tonic.functional.ToFrameStack import to_frame_stack_numpy
from EvEye.utils.cache.MemmapCacheStructedEvents import *
from EvEye.utils.visualization.visualization import *
from EvEye.utils.tonic.functional.CutMaxCount import cut_max_count
from EvEye.dataset.DavisEyeEllipse.utils import *


class DavisEyeEllipseDataset(Dataset):
    def __init__(
        self,
        root_path: Path | str,
        split="train",
        accumulate_mode="fixed_time",
        sensor_size=(346, 260, 2),
        events_interpolation="causal_linear",
        pupil_area=200,
        num_classes=1,
        default_resolution=[256, 256],
        model_type=None,
    ):
        super(DavisEyeEllipseDataset, self).__init__()
        self.root_path = Path(root_path)
        self.split = split
        self.data_path = self.root_path / self.split / "cached_data"
        self.ellipse_path = self.root_path / self.split / "cached_ellipse"
        self.accumulate_mode = accumulate_mode
        self.sensor_size = sensor_size
        self.events_interpolation = events_interpolation
        self.pupil_area = pupil_area
        self.num_classes = num_classes
        self.max_objs = 100
        self.default_resolution = default_resolution
        self.model_type = model_type
        self.total_frames = self.get_nums()

    def get_nums(self):
        num_frames_list = []

        ellipses_list = load_cached_structed_ellipses(self.ellipse_path)
        for ellipses in ellipses_list:
            num_frames_list.append(len(ellipses))
        total_frames = sum(num_frames_list)
        return total_frames

    def get_transforms(self):
        if self.split == "train":
            transform = A.ReplayCompose(
                [
                    A.Resize(self.default_resolution[0], self.default_resolution[1]),
                    A.ShiftScaleRotate(
                        shift_limit=0.2,
                        scale_limit=0.2,
                        rotate_limit=15,
                        interpolation=cv2.INTER_LINEAR,
                        border_mode=cv2.BORDER_CONSTANT,
                        p=1,
                    ),
                    A.HorizontalFlip(p=0.5),
                ]
            )
            return transform
        else:
            transform = A.ReplayCompose(
                [
                    A.Resize(self.default_resolution[0], self.default_resolution[1]),
                ]
            )
            return transform

    def get_event_transforms(self):
        if self.split == "train":
            transform = tonic.transforms.Compose(
                [
                    tonic.transforms.DropEvent(p=0.3),
                    tonic.transforms.DropEventByArea(
                        sensor_size=self.sensor_size, area_ratio=0.1
                    ),
                ]
            )
            return transform
        else:
            transform = tonic.transforms.Compose([])
            return transform

    def transform_ellipse(self, ellipse, replay):
        # 创建一个空白画布
        canvas = np.zeros((self.sensor_size[1], self.sensor_size[0], 3), dtype=np.uint8)
        cv2.ellipse(canvas, ellipse, (255, 255, 255), -1)

        # 应用变换
        transformed_ellipse = A.ReplayCompose.replay(replay, image=canvas)["image"]

        # 转换为灰度图像
        gray = cv2.cvtColor(transformed_ellipse, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            raise ValueError("No contours found")

        cnt = contours[0]

        # 检查点的数量是否足够
        if len(cnt) < 5:
            raise ValueError("Not enough points to fit an ellipse")

        # 拟合椭圆
        ellipse = cv2.fitEllipse(cnt)

        # 处理拟合后的椭圆参数
        x, y = ellipse[0]
        a, b = ellipse[1]
        ang = ellipse[2]
        if a < b:
            a, b = b, a
            ang += 90
        while ang > 90 or ang < -90:
            if ang > 90:
                ang -= 180
            elif ang < -90:
                ang += 180
        x, y, a, b, ang = [round(val, 2) for val in [x, y, a, b, ang]]
        new_ellipse = ((x, y), (a, b), ang)

        return new_ellipse

    def cal_trig(self, ang):
        ang = np.deg2rad(ang)
        sin2A = np.sin(2 * ang)
        cos2A = np.cos(2 * ang)

        return np.array([sin2A, cos2A])

    def get_edge(self, img, prob=0.5):
        if random.randint(0, 9) in range(0, int(10 * prob)):
            img_b = img[:, :, 0]
            img_g = img[:, :, 1]
            img_r = img_b + img_g
            img_r = np.clip(img_r, 0, 255)
            img = np.dstack([img_b, img_g, img_r])
            img = img.astype(np.uint8)
            img = np.where(img > 0, 255, img)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.GaussianBlur(img, (5, 5), 0)

            img_1 = cv2.Laplacian(img, -1, ksize=5)
            img_1 = cv2.normalize(
                img_1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
            )
            img_2 = cv2.Canny(img, 50, 150)
            img_2 = cv2.normalize(
                img_2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
            )
            img_3 = 0.5 * cv2.Sobel(img, -1, 0, 1, 5) + 0.5 * cv2.Sobel(
                img, -1, 1, 0, 5
            )
            img_3 = cv2.normalize(
                img_3, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
            )
            img_4 = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 2
            )
            img_4 = cv2.normalize(
                img_4, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
            )

            img = np.stack([img_b, img_g, img_r, img_1, img_2, img_3, img_4], axis=2)
            img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        return img

    def __len__(self):
        return self.total_frames

    def __getitem__(self, index):
        # 数据加载
        ellipse = convert_to_ellipse(load_ellipse(index, self.ellipse_path))
        event_segment = load_event_segment(index, self.data_path, 5000)

        # 事件增广
        event_transform = self.get_event_transforms()
        event_segment = event_transform(event_segment)

        # 累帧并变换通道
        # event_frame.shape: (2, 260, 346) -> (260, 346, 2)
        # start_time = time.time()
        if self.events_interpolation == "causal_linear_ori":
            weight = 1
            inter = "causal_linear"
            top = None
        elif self.events_interpolation == "causal_linear":
            weight = 10
            inter = "causal_linear"
            top = 255
        elif self.events_interpolation == "bilinear":
            weight = 1
            inter = "bilinear"
            top = None

        event_frame = to_frame_stack_numpy(
            event_segment,
            self.sensor_size,
            1,
            inter,
            event_segment["t"][0],
            event_segment["t"][-1],
            weight,
        ).squeeze(0)
        if self.events_interpolation == "causal_linear":
            cut_max_count(event_frame, maxcount=255)
        # end_time = time.time()
        # binning_time = end_time - start_time

        event_frame = np.moveaxis(event_frame, 0, -1)
        # 数据增广
        # event_frame.shape: (260, 346, 2) -> (2, 256, 256)
        transform = self.get_transforms()
        transformed = transform(image=event_frame)
        event_frame = transformed["image"]
        replay = transformed["replay"]
        if (
            ellipse[0] != (0, 0)
            and cal_ellipse_area(ellipse[1][0], ellipse[1][1]) > self.pupil_area
        ):
            ellipse = self.transform_ellipse(ellipse, replay)
            close = 0
        else:
            close = 1
        # 用于ElNet提取边缘
        if self.model_type == "ElNet":
            event_frame = self.get_edge(event_frame, prob=1)

        event_frame = event_frame.astype(np.float32) / 255.0
        event_frame = np.moveaxis(event_frame, -1, 0)

        # 下采样标签
        down_ratio = 4
        output_height = self.default_resolution[0] // down_ratio
        output_width = self.default_resolution[1] // down_ratio
        x, y = ellipse[0]
        a, b = ellipse[1]
        an = ellipse[2]
        x, y, a, b = [round(val / down_ratio, 2) for val in [x, y, a, b]]
        x = np.clip(x, 0, output_width - 1)
        y = np.clip(y, 0, output_height - 1)
        a = np.clip(a, 0, output_width - 1)
        b = np.clip(b, 0, output_height - 1)
        ellipse_downsampled = ((x, y), (a, b), an)
        ellipse_downsampled_tensor = torch.tensor([x, y, a, b, an], dtype=torch.float32)

        # 初始化输出
        hm = np.zeros((self.num_classes, output_height, output_width), dtype=np.float32)
        ab = np.zeros((self.max_objs, 2), dtype=np.float32)
        ang = np.zeros((self.max_objs, 1), dtype=np.float32)
        trig = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        mask = np.zeros((self.num_classes, 64, 64), dtype=np.float32)

        # 生成输出
        radius = gaussian_radius((math.ceil(b), math.ceil(a)))
        radius = max(0, int(radius))
        center = np.array([x, y], dtype=np.float32)
        center_int = center.astype(np.int32)
        draw_umich_gaussian(hm[0], center_int, radius)
        ab[0] = a * 1.0, b * 1.0
        ang[0] = an * 1.0 + 90
        trig[0] = self.cal_trig(an * 1.0 + 90)
        ind[0] = center_int[1] * output_width + center_int[0]
        reg[0] = center - center_int
        reg_mask[0] = 1
        cv2.ellipse(mask[0], ellipse_downsampled, 1, -1)

        # 可视化mask
        # event_frame_downsampled = cv2.resize(
        #     event_frame_vis,
        #     (output_width, output_height),
        #     interpolation=cv2.INTER_LINEAR,
        # )
        # cv2.ellipse(event_frame_downsampled, ellipse_downsampled, (0, 255, 0), 1)
        # save_image(
        #     event_frame_downsampled,
        #     "/mnt/data2T/junyuan/eye-tracking/event_frame_downsampled.png",
        # )
        # cv2.ellipse(mask[0], ellipse_downsampled, 255, 1)
        # cv2.ellipse(mask[0], center_int, (int(a), int(b)), int(an - 90), 0, 360, 1, -1)

        ret = {
            "input": event_frame,
            "hm": hm,
            "reg_mask": reg_mask,
            "ind": ind,
            "ab": ab,
            "ang": ang,
            "trig": trig,
            "mask": mask,
            "reg": reg,
            "center": center,
            "close": close,
            "ellipse": ellipse_downsampled_tensor,
            # "binning_time": binning_time,
        }

        return ret


# def main():
#     # from torch.utils.data import DataLoader

#     dataset = DavisEyeEllipseDataset(
#         root_path="/mnt/data2T/junyuan/Datasets/FixedTime10000Dataset",
#         split="val",
#         accumulate_mode="fixed_count",
#         sensor_size=(346, 260, 2),
#         events_interpolation="causal_linear_ori",
#         model_type="EPNet",
#     )
#     binning_time_list = []
#     for i in tqdm(range(len(dataset))):
#         data = dataset[i]
#         binning_time_list.append(data["binning_time"])
#     avg_binning_time = sum(binning_time_list) / len(binning_time_list)
#     print(f"Average binning time: {avg_binning_time*1000:.4f} ms")

#     # 使用 DataLoader 加载数据集
#     # dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)

#     # # 打印数据集长度
#     # print(f"Dataset length: {len(dataset)}")

#     # # 迭代 DataLoader 并打印一些基本信息
#     # for batch_idx, batch_data in enumerate(dataloader):
#     #     print(f"Batch {batch_idx + 1}")
#     #     for key, value in batch_data.items():
#     #         print(f"{key} shape: {value.shape}")
#     #     break


# if __name__ == "__main__":
#     main()
