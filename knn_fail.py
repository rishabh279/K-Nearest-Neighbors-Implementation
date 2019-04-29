import numpy as np
import matplotlib.pyplot as plt
from knn import KNN


def get_data():
    width = 8
    height = 8
    N = width * height
    x = np.zeros((N, 2))
    y = np.zeros(N)
    n = 0
    start_t = 0
    for i in range(width):
        t = start_t
        for j in range(height):
            x[n] = [i, j]
            y[n] = t
            n += 1
            t = (t + 1) % 2
        start_t = (start_t + 1) % 2
    return x, y

x, y = get_data()
plt.scatter(x[:,0], x[:,1], s=100, c=y, alpha=0.5)
plt.show()
print()