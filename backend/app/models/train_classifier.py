import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

# import the CNN model 
from model_architecture import Model 

# Input: FER2013 images 
# transforming the inputs to be: grayscale, 48x48 size, conver to pytorch tensor, normalize the images
transform = transform.Compose([
    transforms.Graysclae(num_output_channels = 1),
    transforms.Resize((48, 48))
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])



