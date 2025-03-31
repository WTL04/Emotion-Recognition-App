import torch
import torch.nn as nn 
import torch.nn.functional as F


class Model(nn.Module):
    """A simple Convolutional Neural Network (CNN) for image processing."""

    def __init__(self):
        """Initialize the model layers."""
        super().__init__()

        # First convolutional layer (extracts basic features)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.activation1 = nn.ReLU()

        # Second convolutional layer (learns more complex features)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.activation2 = nn.ReLU()

        # Max pooling layer (reduces image size, retains important features)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input image tensor (batch_size, 1, 48, 48).

        Returns:
            torch.Tensor: Processed feature map.
        """
        x = self.activation1(self.conv1(x))  # Apply first conv layer + ReLU
        x = self.activation2(self.conv2(x))  # Apply second conv layer + ReLU
        x = self.pool(x)  # Downsample using max pooling
        return x


