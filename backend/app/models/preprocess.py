import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

# Input: FER2013 images 
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels = 1), # grayscale the images
    transforms.RandomCrop((46, 46)), # crop the images for more randomess
    transforms.Resize((48, 48)), # resize images
    transforms.RandomHorizontalFlip(), # flip randomly 
    transforms.RandomRotation(20), # rotate randomly
    transforms.ToTensor(), # turn into a tensor
    transforms.Normalize(mean=[0.5], std=[0.5]) # Normalize the tensor
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels = 1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
]


# loading training dataset
train_batch_size = 64
train_dataset = datasets.ImageFolder(root = "../data/archive/train", transform=train_transform) 

# using sampler to balance classes 
targets = train_dataset.targets # list of class indices
class_counts = np.bincount(targets) # num of images per class

# computing class weights
class_weights = 1.0 / class_counts
sample_weights = [class_weights[label] for label in targets]

# create sampler
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=train_batch_size, sampler=sampler, num_workers = 0) # num_workers: num of subprocesses for loading data
print(f"Loading Training set with batch size {train_batch_size}")

# loading test dataset
test_batch_size = 256
test_dataset = datasets.ImageFolder(root = "../data/archive/test", transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
print(f"Loading Test set with batch size {test_batch_size}")

print("Datasets loaded successfully!")

