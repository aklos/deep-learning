import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from utils import generate_training_data


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()

        self.out = nn.Linear(2, 1)  # Single layer (not really, just output)

    def forward(self, x):
        x = self.out(x)
        return x


model = LinearRegressionModel()
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.000001)
epochs = 1000
final_losses = []

[X, y] = generate_training_data(200000)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=0
)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)
y_train = y_train.view(-1, 1)
y_test = y_test.view(-1, 1)

model.train()
for i in range(epochs):
    i = i + 1
    y_pred = model.forward(X_train)
    loss = loss_function(y_pred, y_train)
    final_losses.append(loss)

    if i % 10 == 1:
        print("Epoch number: {} and the loss : {}".format(i, loss.item()))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()
for i, x in enumerate(X_test):
    prediction = model(x)
    print(
        "prediction:",
        int(prediction.squeeze().item()),
        "\tanalysis:",
        int(y_test[i]),
        "\tdifference:",
        abs(int(prediction.squeeze().item() - y_test[i])),
        "\tfeatures:",
        x,
    )
