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

train_dataset = datasets.ImageFolder(root = "../data/archive/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.ImageFolder(root = "../data/archive/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

print("Datasets loaded successfully!")
