import torch
import torch.nn as nn 
import torch.nn.functional as F


class Model(nn.Module):
    """
    A Convolutional Neural Network (CNN) model for image processing.

    This model consists of two convolutional layers followed by ReLU activation functions 
    and a max pooling layer to reduce spatial dimensions while preserving important features.
    """

    def __init__(self):
        """
        Initializes the CNN model layers.
        """
        super().__init__()

        """
        nn.Conv2d:
        Applies a 2D convolution operation to extract features from the input images.

        Parameters:
            in_channels (int): Number of input channels. FER-2013 images are grayscale, so in_channels = 1.
            out_channels (int): Number of filters (feature maps) learned by the layer.
            kernel_size (int or tuple): Size of the convolutional kernel. A 3x3 kernel is commonly used.
        """
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.activation1 = nn.ReLU()

        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.activation2 = nn.ReLU()

        """
        nn.MaxPool2d:
        Performs downsampling by selecting the maximum value within each region of the feature map.

        Parameters:
            kernel_size (int or tuple): The size of the window over which max pooling is computed.
            stride (int or tuple): The step size for moving the pooling window. If stride is not specified, 
                                   it defaults to the kernel size.
        
        Purpose:
            - Reduces spatial dimensions, decreasing computation and memory usage.
            - Helps in reducing overfitting by focusing on dominant features.
            - Improves model invariance to small translations in the input.
        """

        # Pooling Layer - reduces spatial dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input image tensor with shape (batch_size, 1, 48, 48).
        
        Returns:
            torch.Tensor: Feature map after passing through convolutional, activation, and pooling layers.
        """
        x = self.activation1(self.conv1(x))  # First convolution + activation
        x = self.activation2(self.conv2(x))  # Second convolution + activation
        x = self.pool(x)  # Max pooling to reduce feature map size
        return x
