# -*- coding: utf-8 -*-
import torch
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

np.random.seed(0)
sample_list = []
count = 0
while count < 64:
    temInt = np.random.randint(0, 1700)
    if temInt not in sample_list:
        sample_list.append(int(temInt))
        count += 1
sample_list.sort()
print(sample_list)


class CitiBikeDataset(Dataset):
    def __init__(
        self,
        mode: str = "train",
        dataset_path: str = "/mnt/data2T/junyuan/eye-tracking/BikeDataset",
        split_factor: tuple = None,
    ):
        super(CitiBikeDataset, self).__init__()
        people_data_x, people_data_y = self.make_dataset(data_path=dataset_path)
        self.people_data_x = people_data_x.transpose(0, 4, 1, 2, 3)
        self.people_data_y = people_data_y.transpose(0, 4, 1, 2, 3)
        self.mode = mode
        self.split_factor = split_factor
        self.total_num = people_data_x.shape[0]
        if self.split_factor is None:
            self.train_index = int(self.total_num * 0.8)
            self.validate_index = int(self.total_num * 0.1)
            self.test_index = int(
                self.total_num - self.train_index - self.validate_index
            )
        else:
            self.train_index = int(self.total_num * self.split_factor[0])
            self.validate_index = int(self.total_num * self.split_factor[1])
            self.test_index = int(
                self.total_num - self.train_index - self.validate_index
            )
        self.split = [self.train_index, self.validate_index, self.test_index]

        if self.mode == "train":
            self.data = self.people_data_x[0 : self.split[0]]
            self.label = self.people_data_y[0 : self.split[0]]
        elif self.mode == "val":
            self.data = self.people_data_x[
                self.split[0] : self.split[0] + self.split[1]
            ]
            self.label = self.people_data_y[
                self.split[0] : self.split[0] + self.split[1]
            ]
        elif self.mode == "test":
            self.data = self.people_data_x[self.split[0] + self.split[1] :]
            self.label = self.people_data_y[self.split[0] + self.split[1] :]
        else:
            raise ValueError("Invalid mode! Choose from 'train', 'val', or 'test'.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index].astype(np.float32), self.label[index].astype(np.float32)

    def make_dataset(
        self,
        data_duration=3,
        label_duration=1,
        data_path="/mnt/data2T/junyuan/eye-tracking/BikeDataset",
    ):
        flow_data_list = []
        x_data_list = []
        y_data_list = []
        for mon in range(1, 13):
            if mon < 10:
                date_str = str(2023) + "0" + str(mon)
            else:
                date_str = str(2023) + str(mon)
            flow_data_list.append(np.fromfile(f"{data_path}/bike_{date_str}"))

        raw_flow_data = np.concatenate(flow_data_list, axis=0).reshape((-1, 32, 32, 2))
        raw = (raw_flow_data - raw_flow_data.min()) / (
            raw_flow_data.max() - raw_flow_data.min()
        )

        for i in range(data_duration, len(raw)):
            x_data = raw[i - data_duration : i]
            x_data = x_data.reshape(1, -1, raw.shape[1], raw.shape[2], 2)
            x_data_list.append(x_data)

            y_data = raw[i : i + label_duration]
            y_data = y_data.reshape(1, -1, raw.shape[1], raw.shape[2], 2)
            y_data_list.append(y_data)
        x_data_list = np.concatenate(x_data_list, axis=0)
        y_data_list = np.concatenate(y_data_list, axis=0)
        return x_data_list, y_data_list


def main():
    dataset = CitiBikeDataset(mode="train")
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for i, (x, y) in enumerate(dataloader):
        print(f"Batch {i+1}:")
        print(f"Data shape: {x.shape}")
        print(f"Data dtype: {x.dtype}")
        print(f"Label shape: {y.shape}")
        print(f"Label dtype: {y.dtype}")
        # print(f'Input data: {x}')
        # print(f'Output data: {y}')
        print()
    # print(dataset[0])


if __name__ == "__main__":
    main()
