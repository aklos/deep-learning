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
# indices = list(range(len(trainset)))
# np.random.shuffle(indices)
# # k-fold cross validation setup (5-fold in this case?)
# split = int(np.floor(0.2 * len(trainset)))  # 20% of training set
# train_sample = SubsetRandomSampler(indices[:split])
# valid_sample = SubsetRandomSampler(indices[split:])

# 64 images per batch
# trainloader = torch.utils.data.DataLoader(trainset, sampler=train_sample, batch_size=64)
# validloader = torch.utils.data.DataLoader(trainset, sampler=valid_sample, batch_size=64)
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


"""Model implementation"""


class LinearClassificationModel(nn.Module):
    def __init__(self):
        super(LinearClassificationModel, self).__init__()

        self.flatten = nn.Flatten()
        self.out = nn.Linear(32 * 32, len(labels))

    def forward(self, x):
        x = self.flatten(x)
        x = self.out(x)
        # Activation function!
        # x = torch.softmax(x, dim=1)
        x = self.softmax(x)
        return x

    def softmax(self, x):
        x_exp = torch.exp(x)
        partition = x_exp.sum(1, keepdims=True)
        return x_exp / partition

    # def validation_step(self, batch):
    #     Y_hat = self(*batch[:-1])

    # def accuracy(self, Y_hat, Y, averaged=True):
    #     Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
    #     predictions = Y_hat.argmax(axis=1).type(Y.dtype)
    #     compare = (predictions == Y.reshape(-1)).type(torch.float32)
    #     return compare.mean() if averaged else compare


model = LinearClassificationModel()


def train_model():
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    epochs = 100

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

    torch.save(model.state_dict(), "./linear-classifier.pth")


def test_saved_model():
    model.load_state_dict(torch.load("./linear-classifier.pth"))

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
