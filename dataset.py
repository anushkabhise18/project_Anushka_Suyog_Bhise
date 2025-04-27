import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import os
from config import *

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])
    
    return train_transform, val_transform

class ASLDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root=root_dir, transform=transform)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    @property
    def class_names(self):
        return self.classes

def create_dataloaders(dataset_dir=data_dir, batch_size=batch_size):
    
    train_transform, val_transform = get_transforms()
    
    
    base_dataset = datasets.ImageFolder(root=dataset_dir)
    dataset_size = len(base_dataset)
    
    
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    
    indices = list(range(dataset_size))
    torch.manual_seed(random_seed)
    
    
    random_indices = torch.randperm(len(indices)).tolist()
    
    
    train_indices = random_indices[:train_size]
    val_indices = random_indices[train_size:train_size + val_size]
    test_indices = random_indices[train_size + val_size:]
    
    
    train_dataset = ASLDataset(root_dir=dataset_dir, transform=train_transform)
    val_dataset = ASLDataset(root_dir=dataset_dir, transform=val_transform)
    test_dataset = ASLDataset(root_dir=dataset_dir, transform=val_transform)
    
   
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    test_subset = Subset(test_dataset, test_indices)
    
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    print(f"Dataset split: Train={train_size}, Validation={val_size}, Test={test_size}")
    
    return train_loader, val_loader, test_loader