import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets, transforms

class CIFAR10data:
    def __init__(self):
        self.batch_size = 512
        self.shuffle = True
        self.root = './data'

    def get_data(self):
        cifar_dataset = datasets.CIFAR10(root=self.root, train=True, download=True)
        self.data_classes = cifar_dataset.classes
        return cifar_dataset

    def get_dataset_mean_std(self, data):
        data = data.data / 255 # data is numpy array
        mean = data.mean(axis = (0,1,2)) 
        std = data.std(axis = (0,1,2))
        print(f"INFO: Mean : {mean}   STD: {std}")
        return mean, std

    def get_transform(self, mean, std):
        train_transforms = A.Compose([
                                A.Normalize(
                                    mean=tuple(mean), 
                                    std=tuple(std),
                                    always_apply=True
                                ),
                                A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
                                A.RandomCrop(height=32, width=32, always_apply=True),
                                A.HorizontalFlip(),
                                 A.CoarseDropout(
                                                max_holes=1, max_height=16, 
                                                max_width=16, min_holes=1,
                                                min_height=16, min_width=16,
                                                fill_value=mean, mask_fill_value=None
                                            ),  # type: ignore
                            ToTensorV2(),
                            ])

        test_transforms = A.Compose([
                                                A.Normalize(
                                                    mean=tuple(mean), 
                                                    std=tuple(std),
                                                    always_apply=True
                                                ),
                                                ToTensorV2(),
                            ])
        return train_transforms, test_transforms

    def get_transform_data(self, train_transforms, test_transforms):
        train_data = datasets.CIFAR10(root=self.root, train=True, download=True, transform=lambda img:train_transforms(image=np.array(img))["image"])
        test_data = datasets.CIFAR10(root=self.root, train=False, download=True, transform=lambda img:test_transforms(image=np.array(img))["image"])
        return train_data, test_data

    def get_dataloader(self, train_data, test_data):
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=self.shuffle, pin_memory=True)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=self.shuffle, pin_memory=True) 
        return train_dataloader, test_dataloader 
