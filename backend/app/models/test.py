import torch
import torch.nn as nn
from preprocess import test_loader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from classifier_architecture import Model

# initiate model 
model = Model()
model.load_state_dict(torch.load("emotion_model_full.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))) # load saved trained model weights
model.eval()

correct = 0
total = 0
test_loss = 0

# Track all predictions and labels for confusion matrix
all_preds = []
all_labels = []
emotion_labels = ["Angry", "Happy", "Neutral", "Sad"]

criterion = nn.CrossEntropyLoss()

# disable gradient for testing (saves memory)
with torch.no_grad():
    for images, labels in test_loader:
        output = model(images) # outputs tensor
        _, predicted = torch.max(output, 1) # return tensor containing indexes of maximum value

        # store prediction and labeels for confusion matrix
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        loss = criterion(output, labels) # compute loss
        test_loss += loss.item()

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
average_loss = test_loss / len(test_loader)

print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Test Loss: {average_loss:.4f}")

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Plot it
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=emotion_labels, yticklabels=emotion_labels, cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Saved confusion matrix to confusion_matrix.png")
