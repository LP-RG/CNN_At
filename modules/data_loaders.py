import torch
import torchvision.transforms as transforms
from torchvision import datasets, transforms

def get_cifar10(batch_size, data_root, **kwargs):
    data_root = data_root
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-10 data loader with {} workers".format(num_workers))
    ds = []

    # training data with data augmentation
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root=data_root, train=True, download=True,
            transform=transforms.Compose([
                transforms.Pad(4),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    ds.append(train_loader)

    # testing data
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root=data_root, train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])),
        batch_size=batch_size, shuffle=False, **kwargs)
    ds.append(test_loader)

    # training data without data augmentation
    train_loader_no_aug = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root=data_root, train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    ds.append(train_loader_no_aug)

    return train_loader, test_loader, train_loader_no_aug



def get_datasets(batch_size):
    train_loader, test_loader, _ = get_cifar10(batch_size=batch_size, data_root='./data/cifar')
    return train_loader, test_loader, _