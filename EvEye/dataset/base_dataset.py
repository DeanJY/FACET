from typing import Any, Type
from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class DvEyeDataset(Dataset, ABC):
    # 此处的split表示数据集的划分方式，如train、val、test等， **kargs表示其他参数
    def __init__(self, split: str, **kargs) -> None:
        super().__init__()
        self._split = split
        self._init_dataset(**kargs)

    @abstractmethod
    def _init_dataset(self, **kargs):
        pass
