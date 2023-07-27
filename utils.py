import random
import math
import torch

age_min = 0
age_max = 100
size_min = 30
size_max = 1000


def fa(a):
    return 100 - a


def fs(s):
    return math.log(s) * s * 1000


def generate_training_data(num):
    data = []
    expected_outputs = []

    for x in range(0, num):
        age = random.uniform(age_min, age_max)
        size = random.uniform(size_min, size_max)
        data.append([float(age), float(size)])
        expected_outputs.append(fa(age) + fs(size))

    return [torch.tensor(data), expected_outputs]
