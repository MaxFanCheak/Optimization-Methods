import numpy as np
from itertools import count
import matplotlib.pyplot as plt


def generator_fn(optimizing_variables):
    return sum([var ** 2 + var for var in optimizing_variables])


def generator_fndn(optimizing_variables):
    return np.array([(2 * var + 1) for var in optimizing_variables])


def next_value_for_n(point, lr):
    return point + generator_fndn(point) * -lr


def gd():
    count = 0
    x1 = np.array(point)
    epochs = 1
    x2 = next_value_for_n(x1, learning_rate)
    i = 2
    while True:
        count += 1
        if abs(generator_fn(x2) - generator_fn(x1)) < eps:
            break
        if i > 10000000:
            print("Ne povezlo ne fortanulo")
            break
        x1 = x2
        x2 = next_value_for_n(x2, learning_rate)
        epochs += 1
        i += 1
    return count


learning_rate = 0.05
eps = 0.1
check_dim = [3, 5, 6, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
answer = []
for dim in check_dim:
    point = [5] * dim
    last_value = generator_fn(point)
    answer.append(gd())
plt.plot(check_dim, answer)
plt.show()
