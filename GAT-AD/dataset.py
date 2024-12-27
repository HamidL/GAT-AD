import torch
from torch_geometric.data import (
    Dataset,
    Data
)
import numpy as np


class TimeDataset(Dataset):
    def __init__(self, data, config):
        super().__init__()
        self.config = config
        self.eps = 1e-6

        # Apply logarithm if necessary
        if self.config.get("log", False):
            self.data_paths = torch.log(data["paths"] + 1)
        else:
            self.data_paths = data["paths"]

        if "train_paths" in self.config:
            # Validation or testing TimeDataset: use training data mean and std
            if self.config.get("log", False):
                self._mean = torch.log(self.config.get("train_paths") + 1).mean(dim=1, keepdim=True)
                self._std = torch.log(self.config.get("train_paths") + 1).std(dim=1, keepdim=True) + self.eps
            else:
                self._mean = self.config.get("train_paths").mean(dim=1, keepdim=True)
                self._std = self.config.get("train_paths").std(dim=1, keepdim=True) + self.eps
        else:
            # Compute mean and std
            self._mean = self.data_paths.mean(dim=1, keepdim=True)
            self._std = self.data_paths.std(dim=1, keepdim=True) + self.eps

        # Apply standardization if necessary
        if self.config.get("standardize", False):
            self.data_paths = (self.data_paths - self.mean) / self.std

        self.edge_index = data["path_to_path"]

    def len(self):
        return len(self.data_paths[0]) - self.config.get("window_size", 0)

    @property
    def num_paths(self):
        return len(self.data_paths)

    @property
    def std(self):
        return self._std

    @property
    def mean(self):
        return self._mean

    @property
    def status(self):
        return self.data.get("status", None)

    @property
    def num_nodes(self):
        return self.data_paths.shape[0]

    def get(self, idx):
        if self.config.get("window_size", 0) == 0:  # only one sample
            x = self.data_paths[:, idx].clone()
            y = self.data_paths[:, idx].clone()
        else:
            x = self.data_paths[:, idx:idx+self.config["window_size"]].clone()
            y = self.data_paths[:, idx+self.config["window_size"]].clone()

        data = Data(
            x=x,
            y=y,
            edge_index=self.edge_index
        )
        return data


def read_data(path, config):
    test_ratio, val_ratio = config["test_ratio"], config["val_ratio"]
    data = torch.load(path)
    num_samples = len(data["paths"][0])

    train_samples = int(num_samples * (1 - test_ratio - val_ratio))
    val_samples = int(num_samples * val_ratio)

    train_paths = data["paths"][:, :train_samples]
    val_paths = data["paths"][:, train_samples:train_samples+val_samples]
    test_paths = data["paths"][:, train_samples+val_samples:]

    if "status" in data.keys():
        train_status = data["status"][:, :train_samples]
        val_status = data["status"][:, train_samples:train_samples+val_samples]
        test_status = data["status"][:, train_samples+val_samples:]
    else:
        train_status, val_status, test_status = None, None, None

    path_to_path = data["path_to_path"]

    train_data = dict(
        paths=train_paths,
        status=train_status,
        path_to_path=path_to_path
    )
    val_data = dict(
        paths=val_paths,
        status=val_status,
        path_to_path=path_to_path
    )
    test_data = dict(
        paths=test_paths,
        status=test_status,
        path_to_path=path_to_path
    )

    train_dataset = TimeDataset(train_data, config)
    test_config = {**config, "train_paths": train_paths}
    val_dataset = TimeDataset(val_data, test_config)
    test_dataset = TimeDataset(test_data, test_config)

    return train_dataset, val_dataset, test_dataset


def read_wadi_data(path, config):
    val_ratio = config.get("val_ratio", 0.1)
    data = torch.load(path)
    num_train_samples = data["train"].shape[1]

    train_samples = int(num_train_samples * (1 - val_ratio))
    val_samples = int(num_train_samples * val_ratio)

    train_nodes = data["train"][:, :train_samples]
    val_nodes = data["train"][:, train_samples:]
    test_nodes = data["test"]

    train_status, val_status, test_status = torch.zeros(train_nodes.shape[1]), torch.zeros(val_nodes.shape[1]), data["test_labels"]

    node_to_node = data["node_to_node"]

    train_data = dict(
        paths=train_nodes,
        status=train_status,
        path_to_path=node_to_node
    )
    val_data = dict(
        paths=val_nodes,
        status=val_status,
        path_to_path=node_to_node
    )
    test_data = dict(
        paths=test_nodes,
        status=test_status,
        path_to_path=node_to_node
    )

    train_dataset = TimeDataset(train_data, config)
    test_config = {**config, "train_paths": train_nodes}
    val_dataset = TimeDataset(val_data, test_config)
    test_dataset = TimeDataset(test_data, test_config)

    return train_dataset, val_dataset, test_dataset
