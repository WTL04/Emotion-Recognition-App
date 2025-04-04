import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Input: FER2013 images 
# transforming the inputs to be: grayscale, 48x48 size, conver to pytorch tensor, normalize the images
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels = 1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
train_batch_size = 16
train_dataset = datasets.ImageFolder(root = "../data/archive/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
print(f"Loading Training set with batch size {train_batch_size}")

test_batch_size = 256
test_dataset = datasets.ImageFolder(root = "../data/archive/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
print(f"Loading Test set with batch size {test_batch_size}")

print("Datasets loaded successfully!")
