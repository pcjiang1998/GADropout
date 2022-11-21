import os

import numpy as np
import torch.utils.data as data

from datasets.PklDataset import PklDataset

_data_file_path: None or str = None
_folder = None
_dataset_name = None


def get_data_file_path() -> str:
    global _data_file_path
    assert isinstance(_data_file_path, str), "File not found"
    assert os.path.isfile(_data_file_path), "File not found"
    return _data_file_path


def set_data_file_path(dataset_name: str) -> None:
    global _data_file_path, _dataset_name, _folder
    _dataset_name = dataset_name
    _data_file_path = os.path.join(_folder, f"{dataset_name}.pkl")


def get_train_val_test_data(
    valid_size=0.16,
    test_size=0.2,
    train_data_transform=None,
    val_data_transform=None,
    test_data_transform=None,
    label_transform=None,
):
    if _dataset_name in ["dorothea"]:  # has test dataset
        valid_size = valid_size / (1 - test_size)
        dataset = PklDataset(
            data_file=get_data_file_path(),
            train=True,
            raw=None,
            transform=train_data_transform,
            target_transform=label_transform,
        )
        dataset_val = PklDataset(
            data_file=get_data_file_path(),
            train=True,
            raw=False,
            transform=val_data_transform,
            target_transform=label_transform,
        )
        dataset_test = PklDataset(
            data_file=get_data_file_path(),
            train=False,
            raw=None,
            transform=test_data_transform,
            target_transform=label_transform,
        )
        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]
        train_dataset = data.Subset(dataset, train_idx)
        valid_dataset = data.Subset(dataset_val, valid_idx)
        return {"train": train_dataset, "val": valid_dataset, "test": dataset_test}
    else:
        dataset = PklDataset(
            data_file=get_data_file_path(),
            train=None,
            raw=None,
            transform=train_data_transform,
            target_transform=label_transform,
        )
        dataset_val = PklDataset(
            data_file=get_data_file_path(),
            train=None,
            raw=False,
            transform=val_data_transform,
            target_transform=label_transform,
        )
        dataset_test = PklDataset(
            data_file=get_data_file_path(),
            train=None,
            raw=False,
            transform=test_data_transform,
            target_transform=label_transform,
        )
        num_train = len(dataset)
        indices = list(range(num_train))
        valid_split = int(np.floor(valid_size * num_train))
        test_split = int(np.floor((valid_size + test_size) * num_train))
        np.random.shuffle(indices)
        valid_idx, test_idx, train_idx = (
            indices[:valid_split],
            indices[valid_split:test_split],
            indices[test_split:],
        )
        train_dataset = data.Subset(dataset, train_idx)
        valid_dataset = data.Subset(dataset_val, valid_idx)
        test_dataset = data.Subset(dataset_test, test_idx)
        return {"train": train_dataset, "val": valid_dataset, "test": test_dataset}


def get_train_test_data(
    test_size=0.2,
    train_data_transform=None,
    test_data_transform=None,
    label_transform=None,
):
    if _dataset_name in ["dorothea"]:  # has test dataset
        dataset = PklDataset(
            data_file=get_data_file_path(),
            train=True,
            raw=None,
            transform=train_data_transform,
            target_transform=label_transform,
        )
        dataset_test = PklDataset(
            data_file=get_data_file_path(),
            train=False,
            raw=None,
            transform=test_data_transform,
            target_transform=label_transform,
        )
        return {"train": dataset, "test": dataset_test}
    else:
        dataset = PklDataset(
            data_file=get_data_file_path(),
            train=True,
            raw=None,
            transform=train_data_transform,
            target_transform=label_transform,
        )
        dataset_test = PklDataset(
            data_file=get_data_file_path(),
            train=True,
            raw=False,
            transform=test_data_transform,
            target_transform=label_transform,
        )
        num_train = len(dataset)
        indices = list(range(num_train))
        test_split = int(np.floor(test_size * num_train))
        np.random.shuffle(indices)
        train_idx, test_idx = indices[test_split:], indices[:test_split]
        train_dataset = data.Subset(dataset, train_idx)
        test_dataset = data.Subset(dataset_test, test_idx)
        return {"train": train_dataset, "test": test_dataset}
