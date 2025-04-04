import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Input: FER2013 images 
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels = 1), # grayscale the images
    transforms.RandomCrop((44, 44)), # crop the images for more randomess
    transforms.Resize((48, 48)), # resize images
    transforms.RandomHorizontalFlip(), # flip randomly 
    transforms.RandomRotation(15), # rotate randomly
    transforms.ToTensor(), # turn into a tensor
    transforms.Normalize(mean=[0.5], std=[0.5]) # Normalize the tensor
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels = 1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_batch_size = 64
train_dataset = datasets.ImageFolder(root = "../data/archive/train", transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
print(f"Loading Training set with batch size {train_batch_size}")

test_batch_size = 256
test_dataset = datasets.ImageFolder(root = "../data/archive/test", transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
print(f"Loading Test set with batch size {test_batch_size}")

print("Datasets loaded successfully!")
