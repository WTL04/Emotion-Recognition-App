import torch
import torch.nn as nn
from preprocess import test_loader

model = torch.load("emotion_model_full.pth", weights_only = False) # load saved trained model
model.eval()

# initiate 
correct = 0
total = 0
test_loss = 0

criterion = nn.CrossEntropyLoss()

# disable gradient for testing (saves memory)
with torch.no_grad():
    for images, labels in test_loader:
        output = model(images) # outputs tensor
        _, predicted = torch.max(output, 1) # return tensor containing indexes of maximum value
        loss = criterion(output, labels) # compute loss
        test_loss += loss.item()

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
average_loss = test_loss / len(test_loader)

print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Test Loss: {average_loss:.4f}")
