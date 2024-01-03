import os

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from models.model import MyAwesomeModel

model = MyAwesomeModel()

file_prefix = "data/processed/"

train_images = torch.load(f"{file_prefix}train_images.pt")
train_labels = torch.load(f"{file_prefix}train_labels.pt")


train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_images, train_labels), batch_size=64, shuffle=True
)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 10

losses = []
for e in range(epochs):
    running_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(train_loader)}")
        losses.append(running_loss / len(train_loader))

# create a folder for the model with the class name of the model
os.makedirs(f"models/{model.__class__.__name__}", exist_ok=True)
# Save the model
torch.save(model.state_dict(), f"models/{model.__class__.__name__}/trained_model.pth")
print(f"Model trained and saved at models/{model.__class__.__name__}/trained_model.pth")

# make a plot of the training loss
plt.plot(losses)
plt.savefig("reports/figures/training_loss.png")
