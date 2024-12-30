
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np 



def get_loader(args):
    
    # Computer Vision Dataset
    if args.data_name == 'MNIST':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        trainset = torchvision.datasets.MNIST(
            root=args.data_path, train=True, download=True, transform=transform
        )

        valset = torchvision.datasets.MNIST(
            root=args.data_path, train=False, download=True, transform=transform
        )

        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
        input_shape = (28, 28, 1)
        n_classes = 10
        return train_loader, val_loader, input_shape, n_classes


    elif args.data_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize with mean and std
        ])

        # Load CIFAR-10 dataset
        trainset = torchvision.datasets.CIFAR10(
            root=args.data_path, train=True, download=True, transform=transform
        )
        valset = torchvision.datasets.CIFAR10(
            root=args.data_path, train=False, download=True, transform=transform
        )
        
        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
        input_shape = (32, 32, 3)
        n_classes = 10
        return train_loader, val_loader, input_shape, n_classes


    elif args.data_name == 'CIFAR100':
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize with mean and std
        ])

        # Load CIFAR-10 dataset
        trainset = torchvision.datasets.CIFAR100(
            root=args.data_path, train=True, download=True, transform=transform
        )
        valset = torchvision.datasets.CIFAR100(
            root=args.data_path, train=False, download=True, transform=transform
        )
        
        
        train_loader = DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            valset, batch_size=args.batch_size, shuffle=False
        )
        
        
        input_shape = (32, 32, 3)
        n_classes = 100
        return train_loader, val_loader, input_shape, n_classes


    elif args.data_name == 'SVHN':
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize with mean and std
        ])

        # Load SVHN dataset
        trainset = torchvision.datasets.SVHN(
            root=args.data_path, split='train', download=True, transform=transform
        )
        valset = torchvision.datasets.SVHN(
            root=args.data_path, split='test', download=True, transform=transform
        )
        
        train_loader = DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True
        )
        
        val_loader = DataLoader(
            valset, batch_size=args.batch_size, shuffle=False
        )
        
        
        input_shape = (32, 32, 3)
        n_classes = 10
        return train_loader, val_loader, input_shape, n_classes

    elif args.data_name == 'STL10':
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize with mean and std
        ])

        # Load STL10 dataset
        trainset = torchvision.datasets.STL10(
            root=args.data_path, split='train', download=True, transform=transform
        )
        valset = torchvision.datasets.STL10(
            root=args.data_path, split='test', download=True, transform=transform
        )
        
        train_loader = DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True
        )
        
        val_loader = DataLoader(
            valset, batch_size=args.batch_size, shuffle=False
        )
        
        
        input_shape = (96, 96, 3)
        n_classes = 10
        return train_loader, val_loader, input_shape, n_classes