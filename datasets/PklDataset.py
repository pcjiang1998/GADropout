from typing import Any, Tuple, Optional, Callable

import torch.utils.data

from utils import load_obj


class PklDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_file: str,
        train: None or bool = None,
        raw: None or bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.data_file = data_file
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        data_dict = load_obj(self.data_file)
        if self.train is None:
            assert "test" not in data_dict.keys()
            data_dict = data_dict["train"]
        else:
            assert "test" in data_dict.keys()
            if train:
                data_dict = data_dict["train"]
            else:
                data_dict = data_dict["test"]
        if raw:
            assert "raw" in data_dict["data"].keys()
            self.datas, self.targets = data_dict["raw"], data_dict["target"]
        else:
            self.datas, self.targets = data_dict["data"], data_dict["target"]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data, target = self.datas[index], int(self.targets[index])
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target
