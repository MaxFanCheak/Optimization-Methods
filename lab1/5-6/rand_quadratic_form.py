import numpy as np
import matplotlib.pyplot as plt


def rand_matrix(n):
    return np.random.rand(n, n)


def find_delta_f(start_point, next_point):
    delta_f = start_point - next_point
    if delta_f < eps:
        return next_point


def generator_fn(matrix, optimizing_variables):
    ans = sum([np.diagonal(matrix)[i] * (optimizing_variables[i] ** 2) for i in range(n)])
    ans += sum([matrix[i][j] * optimizing_variables[i] * optimizing_variables[j] + matrix[j][i] * optimizing_variables[
        j] * optimizing_variables[i] for i in range(n) for j in range(i)])
    return ans


def generator_fndn(matrix, optimizing_variables):
    ans = [2 * np.diagonal(matrix)[i] * optimizing_variables[i] for i in range(n)]
    for i in range(n):
        for j in range(i):
            scal = matrix[i][j] + matrix[j][i]
            ans[i] = scal * optimizing_variables[j]
            ans[j] = scal * optimizing_variables[i]
    return np.array(ans)


def next_value_for_n(matrix, point, lr):
    return point + generator_fndn(matrix, point) * -lr


def gd(matrix):
    count = 0
    x1 = np.array(point)
    epochs = 1
    x2 = next_value_for_n(matrix, x1, learning_rate)
    i = 2
    while True:
        count += 1
        if abs(generator_fn(matrix, x2) - generator_fn(matrix, x1)) < eps:
            break
        if i > 10000000:
            print("Ne povezlo ne fortanulo")
            break
        x1 = x2
        x2 = next_value_for_n(matrix, x2, learning_rate)
        epochs += 1
        i += 1
    return count


learning_rate = 0.05
eps = 0.1
check_dim = [3, 5, 6, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 250, 300, 400, 500]

for dim in check_dim:
    n = dim
    point = [5] * dim
    rand_func = rand_matrix(dim)
    last_value = generator_fn(rand_func, point)
    print(gd(rand_func))
