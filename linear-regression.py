import torch
from torch import nn
from tqdm import tqdm
from statistics import mean
from utils import generate_training_data

# Exercise: implement a linear regression model that estimates the value of a house given its age (years), and size (squared meters)


# 1. Model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        # Randomize weights around 0.
        # We only have two features: size and age, so tensor has two vectors with one scalar
        self.w = torch.normal(0.0, 1.0, (2, 1), requires_grad=False)
        # Set bias to 0
        self.b = torch.zeros(1, requires_grad=False)

    # 2. Cost function: squared difference
    def cost(self, a, e):
        c = (a - e) ** 2
        return c

    # The layer calculations.
    def forward(self, X):
        weighted_sum = torch.matmul(X, self.w)  # Compute the weighted sum
        output = weighted_sum + self.b  # Add the bias
        return output

    # 3. Backpropagation
    # https://www.youtube.com/watch?v=8d6jf7s6_Qs&t=0s
    def backprop(self, X, p, e, lr):
        for i, x in enumerate(X):
            w = self.w[i]
            # dC_da = 2 * ((x * w) - e) <- THIS IS WRONG!
            dC_da = 2 * (p - e)
            da_dw = x  # <- THIS IS WHERE WEIGHT AFFECTS GRAD
            grad = dC_da * da_dw
            self.w[i] -= lr * grad
            self.b -= lr * dC_da


# 4. Training
m = LinearRegressionModel()

samples = 200000

[test_data_tensor, test_data_outputs] = generate_training_data(samples)

m.train()
for i, X in tqdm(enumerate(test_data_tensor), total=samples):
    p = m(X)
    e = test_data_outputs[i]
    m.backprop(X, p, e, 0.000001)

# 5. Prediction: feed model entry with [age, size]
m.eval()
# predict_data_tensor = torch.tensor([[1.0, 1000.0]])
# prediction = m(predict_data_tensor)
# print(prediction)
[predict_data_tensor, predict_data_outputs] = generate_training_data(10)
costs = []
for i, x in enumerate(predict_data_tensor):
    prediction = m(x)
    costs.append(m.cost(prediction, predict_data_outputs[i]).squeeze().item())
    print(
        "prediction:",
        int(prediction.squeeze().item()),
        "\tanalysis:",
        int(predict_data_outputs[i]),
        "\tdifference:",
        abs(int(prediction.squeeze().item() - predict_data_outputs[i])),
        "\tfeatures:",
        x,
    )
mean_cost = mean(costs)
print("mean cost:", mean_cost, "(", len(str(mean_cost)) - 2, "sf", ")")
