import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=11664, out_features=10)

    def forward(self, x):
        conv_result = self.conv(x)
        relu_result = self.relu(conv_result)
        pool_result = self.pooling(relu_result)
        flatten_result = self.flatten(pool_result)
        linear_result = self.linear(flatten_result)

        return linear_result


transform = transforms.ToTensor()
train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1000)

model = SimpleCNN()

# cross entropy loss for classification
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()  # Train mode

epochs = 3
running_loss = 0
epsilon = 1e-3
for epoch in range(epochs):
    for images, labels in train_loader:
        logits = model(images)

        optimizer.zero_grad()
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    running_loss /= len(train_loader)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, loss = {running_loss}")
    running_loss = 0

print('Model trained')
print('Evaluating...')

model.eval()  # Evaluation mode
correct = 0
total = 0
for images, labels in test_loader:
    with torch.no_grad():
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Accuracy: {accuracy * 100:.2f}%")

