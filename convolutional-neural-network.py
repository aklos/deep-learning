import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

"""Resize images to 32x32"""
transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

"""Download datasets"""
trainset = datasets.FashionMNIST(
    ".pytorch/F_MNIST_data", download=True, train=True, transform=transform
)
testset = datasets.FashionMNIST(
    ".pytorch/F_MNIST_data", download=True, train=False, transform=transform
)

"""Prep datasets"""
# 64 images per batch
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# Human readable labels for the datasets
labels = [
    "t-shirt",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle boot",
]

"""Visualize"""
dataiter = iter(trainloader)
images, ids = next(dataiter)

fig = plt.figure(figsize=(15, 5))
for idx in np.arange(20):
    ax = fig.add_subplot(4, 20 / 4, idx + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]))
    ax.set_title(labels[ids[idx].item()])
    fig.tight_layout()
plt.show()

def init_cnn(module):
    """Initialize weights for CNNs."""
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(120), nn.ReLU(),
            nn.LazyLinear(84), nn.ReLU(),
            nn.LazyLinear(num_classes)
        )
        
    def forward(self, X):
        return self.net(X)
        

model = LeNet()
model.apply(init_cnn)


def train_model():
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    epochs = 10

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            X_train, y_train = data
            y_pred = model.forward(X_train)
            loss = loss_function(y_pred, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:  # print every 10 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}")
                running_loss = 0.0

    torch.save(model.state_dict(), "./cnn-classifier.pth")


def test_saved_model():
    model.load_state_dict(torch.load("./cnn-classifier.pth"))

    image = None
    label = None

    for images, expected_labels in testloader:
        image = images[0]
        label = expected_labels[0]
        break

    print(labels[label.item()])
    image = image.unsqueeze(0) # Prep as batch for the model
    prediction = model(image).argmax(axis=1)
    print(labels[prediction.item()])

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(image))
    ax.set_title(labels[label.item()] + " -> " + labels[prediction.item()])
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    globals()[sys.argv[1]]()