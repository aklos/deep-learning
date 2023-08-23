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


class MultilayerPerceptron(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super(MultilayerPerceptron, self).__init__()
        self.num_inputs = num_inputs

        # Can't these be two nn.Linear layers?
        # What is the sigma for?
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * sigma)
        self.b2 = nn.Parameter(torch.zeros(num_outputs))

    def forward(self, X):
        X = X.reshape((-1, self.num_inputs))  # Isn't this the same as nn.Flatten()?
        H = self.relu(torch.matmul(X, self.W1) + self.b1)
        return torch.matmul(H, self.W2) + self.b2

    def relu(self, X):
        a = torch.zeros_like(X)
        return torch.max(X, a)


model = MultilayerPerceptron(32 * 32, 10, 256, 0.1)


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

    torch.save(model.state_dict(), "./mlp-classifier.pth")


def test_saved_model():
    model.load_state_dict(torch.load("./mlp-classifier.pth"))

    image = None
    label = None

    for images, expected_labels in testloader:
        image = images[0]
        label = expected_labels[0]
        break

    print(labels[label.item()])
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
