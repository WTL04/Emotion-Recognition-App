import torch
import torch.nn as nn
import torch.optim as optim
from preprocess import train_loader
from classifier_architecture import Model

model = Model()

# define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training loop 
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels, in train_loader:
        optimizer.zero_grad() # reset gradients
        outputs = model(images) # input training set
        loss = criterion(outputs, labels) # calculate loss comparing predicted vs actual
        loss.backward() # backwards propagation optimizing
        optimizer.step() # optimize
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")

torch.save(model, "emotion_model_full.pth")  # Saves the full model


