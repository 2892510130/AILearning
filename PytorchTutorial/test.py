import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

def download_fmnist_data():
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",    
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

if __name__ == "__main__":
    download_fmnist_data()
