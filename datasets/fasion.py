# 导入相关库
import torch
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)


def get_train_data_loader(batch_size):
    # Setup training data
    train_data = datasets.FashionMNIST(
        root="data",  # where to download data to?
        train=True,  # get training data
        download=True,  # download data if it doesn't exist on disk
        transform=ToTensor(),  # images come as PIL format, we want to turn into Torch tensors
        target_transform=None  # you can transform labels as well
    )

    train_dataloader = DataLoader(train_data,  # dataset to turn into iterable
                                  batch_size=batch_size,  # how many samples per batch?
                                  shuffle=True  # shuffle data every epoch?
                                  )
    print(f"Length of train dataloader: {len(train_dataloader)} batches of {batch_size}")

    return train_dataloader


def get_test_data_loader(batch_size):
    # Setup testing data
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,  # get test data
        download=True,
        transform=ToTensor()
    )

    # Turn datasets into iterables (batches)
    test_dataloader = DataLoader(test_data,
                                 batch_size=batch_size,
                                 shuffle=False  # don't necessarily have to shuffle the testing data
                                 )

    print(f"Length of test dataloader: {len(test_dataloader)} batches of {batch_size}")
    return test_dataloader


if __name__ == '__main__':
    test_loader = get_test_data_loader(2)
    a = next(iter(test_loader))
    print(a[0].to('cpu').numpy().tolist())
    print(a[0].shape)  # torch.Size([32, 1, 28, 28])
