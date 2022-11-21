import pickle

import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda
import torch.utils.data as Data
from torchvision.transforms import transforms


def save_obj(obj, pkl_file):
    with open(pkl_file, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(pkl_file):
    with open(pkl_file, "rb") as f:
        return pickle.load(f)


def split_dataset(data, target):
    assert isinstance(data, np.ndarray) and isinstance(
        target, np.ndarray
    ), "data and target should be np.ndarray"
    assert (
        data.shape[0] == target.shape[0]
    ), "data and target should have the same number of instances"
    indices = list(range(len(data)))
    split = int(np.floor(0.2 * len(data)))
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    return {"data": data[train_idx], "target": target[train_idx]}, {
        "data": data[valid_idx],
        "target": target[valid_idx],
    }


def pca(data, dim=2):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=dim)
    data_pca = pca.fit_transform(data)
    return data_pca


class RecurrentDataloader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.it = iter(dataloader)

    def get_data(self):
        try:
            return self.it.next()
        except StopIteration:
            self.it = iter(self.dataloader)
            return self.it.next()


def init_dataloader(
    dataset_name,
    normal=False,
    flatten=True,
    train_batch_size=100,
    val_batch_size=100,
    test_batch_size=100,
    n_workers=0,
    pin_memory=True,
):
    label_transform = None
    # dataset_name in ['abalone', 'Aligned', 'boston', 'breast_cancer', 'diabetes',
    #                     'digits', 'dorothea', 'DryBeanDataset', 'Flocking', 'Grouped',
    #                     'iris', 'letter-recognition', 'ShillBiddingDataset', 'wine', 'WinnipegDataset']:
    from data_provider.pkl import get_train_val_test_data, set_data_file_path

    set_data_file_path(dataset_name)
    train_data_transform = transforms.Compose(
        [transforms.Lambda(lambda x: torch.from_numpy(x).float())]
    )
    val_data_transform = transforms.Compose(
        [transforms.Lambda(lambda x: torch.from_numpy(x).float())]
    )
    test_data_transform = transforms.Compose(
        [transforms.Lambda(lambda x: torch.from_numpy(x).float())]
    )
    datasets = get_train_val_test_data(
        valid_size=0.16,
        test_size=0.2,
        train_data_transform=train_data_transform,
        val_data_transform=val_data_transform,
        test_data_transform=test_data_transform,
        label_transform=label_transform,
    )
    train_loader = Data.DataLoader(
        datasets["train"],
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=pin_memory,
    )
    valid_loader = Data.DataLoader(
        datasets["val"],
        batch_size=val_batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=pin_memory,
    )
    test_loader = Data.DataLoader(
        datasets["test"],
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=pin_memory,
    )
    if normal:
        return train_loader, valid_loader, test_loader
    else:
        return (
            RecurrentDataloader(train_loader),
            RecurrentDataloader(valid_loader),
            RecurrentDataloader(test_loader),
        )
