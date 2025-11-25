import numpy as np
import matplotlib.pyplot as plt

def plot1(start, end, last, full):
    x = np.arange(full)
    y = []
    for i in range(full):
        y.append(end + (start - end) * np.exp(-1. * i / last))

    y = np.array(y)
    plt.plot(x, y)
    plt.show()
    print(y)

def plot2(start, end, last, epoch):
    ran = np.arange(-10, 10, 20 / last)
    y =  1 - 1 / (1 + np.exp(-0.3 * ran))
    y_min = y[0]
    y_max = y[-1]
    y = start + (end - start) / (y_max - y_min) * (y - y_min)
    y = y.tolist() + [0 for _ in range(epoch - last)]
    plt.plot(np.arange(0, epoch), y)
    plt.show()
    print(np.array(y))

plot2(200, 0, 50, 50)