import torch
import torch.nn as nn
import torch.optim as optim
from preprocess import train_loader
from classifier_architecture import Model

# train model on either cpu or gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training using: {device}")
model = Model().to(device)

# define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# scheduler that adjusts learning rate every n epochs
# multiples learning rate by gamma every n epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# training loop 
num_epochs = 60

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels, in train_loader:

        # move images and labels to same device as model
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad() # reset gradients
        outputs = model(images) # input training set
        loss = criterion(outputs, labels) # calculate loss comparing predicted vs actual
        loss.backward() # backwards propagation optimizing
        optimizer.step() # optimize
        running_loss += loss.item()

    scheduler.step() # adjust learning rate
    
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

model.eval()
torch.save(model.state_dict(), "emotion_model_full.pth")  # Saves the model's weights

