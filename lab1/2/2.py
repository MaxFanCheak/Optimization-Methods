import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from itertools import count
import csv
plt.rcParams["figure.figsize"] = (8, 7)
np.set_printoptions(suppress=True)

t = np.linspace(0.001, 1, 200)
ox, oy = np.meshgrid(t, t)
startPoint = [0.01, 0.9]
init_epochs = 25
init_lr = 0.35
init_lrdrop = 0.7
init_epochs_drop = 3
eps = 0.0000000000001


def f(x, y):
    return np.log(x * x + y * y) * x * y


def fdy(x, y):
    return np.log(x * x + y * y) * x + 2 * x * y * y / (x * x + y * y)


def fdx(x, y):
    return np.log(x * x + y * y) * y + 2 * x * x * y / (x * x + y * y)


def grad(point):
    return np.array([fdx(*point), fdy(*point)])


def next_value(point, lr):
    return point + grad(point) * -lr

with open("2/2_stat.csv", mode="w", encoding='utf-8') as w_file:
    file_writer = csv.writer(w_file, delimiter = ",", lineterminator="\r")
    file_writer.writerow(["LR", "lrDrop"," Grad"])

    def gd(epochs=init_epochs, learning_rate=init_lr, lrdrop=init_lrdrop, epochs_drop=init_epochs_drop, writer=file_writer):
        epochsArray = np.arange(1, epochs)
        lrs = np.array([learning_rate * np.power(lrdrop, np.floor((epoch - 1) / epochs_drop)) for epoch in epochsArray])
        points = np.array([startPoint])
        points = np.append(points, [next_value(points[0], learning_rate)], axis=0)
        for i in count(start=2, step=1):
            if abs(f(points[i - 1][0], points[i - 1][1]) - f(points[i - 2][0], points[i - 2][1])) < eps:
                break
            if i > 10000000:
                print("Ne povezlo ne fortanulo")
                break
            points = np.append(points, [next_value(points[i - 1], learning_rate)], axis=0)
            epochs += 1
        print(points)
        writer.writerow(["{:.2f}".format(learning_rate), "{:.2f}".format(lrdrop), epochs])
        return points, epochsArray, lrs


    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    points, epochsArray, lrs = gd()
    GDLine, = ax1.plot(points[:, 0], points[:, 1], 'o-')
    GDContour = ax1.contour(ox, oy, f(ox, oy), levels=sorted({f(p[0], p[1]) for p in points}))
    steps1, = ax2.step(epochsArray, lrs, where='mid')
    steps2, = ax2.plot(epochsArray, lrs, 'o', color='grey', alpha=0.3)

    ax_epochs = plt.axes([0.15, 0.1, 0.75, 0.03])
    ax_lr = plt.axes([0.15, 0.15, 0.75, 0.03])
    ax_lrdrop = plt.axes([0.15, 0.2, 0.75, 0.03])
    ax_drop = plt.axes([0.15, 0.25, 0.75, 0.03])
    epochs_slider = Slider(
        ax=ax_epochs,
        label="Epochs",
        valmin=1,
        valmax=50,
        valinit=init_epochs,
        valstep=1
    )
    lr_slider = Slider(
        ax=ax_lr,
        label="Learning rate",
        valmin=0.01,
        valmax=1,
        valinit=init_lr,
        valstep=0.01
    )
    lrdrop_slider = Slider(
        ax=ax_lrdrop,
        label="drop",
        valmin=0.1,
        valmax=1,
        valinit=init_lrdrop,
        valstep=0.1
    )
    drop_slider = Slider(
        ax=ax_drop,
        label="epochs for drop",
        valmin=1,
        valmax=10,
        valinit=init_epochs_drop,
        valstep=1
    )


    def update(val):
        global GDContour
        points, epochs, lrs = gd(epochs_slider.val, lr_slider.val, lrdrop_slider.val, drop_slider.val)
        GDLine.set_data(points[:, 0], points[:, 1])
        for coll in GDContour.collections:
            coll.remove()
        GDContour = ax1.contour(ox, oy, f(ox, oy), levels=sorted({f(p[0], p[1]) for p in points}))
        steps1.set_data(epochsArray, lrs)
        steps2.set_data(epochsArray, lrs)
        fig.canvas.draw_idle()


    epochs_slider.on_changed(update)
    lr_slider.on_changed(update)
    lrdrop_slider.on_changed(update)
    drop_slider.on_changed(update)
    plt.show()
