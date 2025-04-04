import torch
import torch.nn as nn 
import torch.nn.functional as F


class Model(nn.Module):
    """A simple Convolutional Neural Network (CNN) for image processing."""

    def __init__(self):
        """Initialize the model layers."""
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1), # (64, 48, 48)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (64, 24, 24)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), # (128, 24, 24)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (128, 12, 12)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (256, 6, 6)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 256),
            nn.ReLU(),
            nn.Linear(256, 7) # outputs 1 classification out of 7
        )

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input image tensor (batch_size, 1, 48, 48).

        Returns:
            torch.Tensor: Processed feature map.
        """
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

