import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from itertools import count
import csv


plt.rcParams["figure.figsize"] = (7, 7)
np.set_printoptions(suppress=True)

t = np.linspace(0.001, 1, 200)
ox, oy = np.meshgrid(t, t)
startPoint = [0.01, 0.9]
eps = 0.0000000000001
init_lr = 0.05
with open("1/1_stat.csv", mode="w", encoding='utf-8') as w_file:
    file_writer = csv.writer(w_file, delimiter = ",", lineterminator="\r")
    file_writer.writerow(["LR", "Epochs"])


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


    def gd(learning_rate=init_lr, writer=file_writer):
        points = np.array([startPoint])
        epochs = 1
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
        writer.writerow(["{:.2f}".format(learning_rate), epochs])
        return points


    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')
    array = gd()
    GDLine, = ax1.plot(array[:, 0], array[:, 1], 'o-')
    GDContour = ax1.contour(ox, oy, f(ox, oy), levels=sorted({f(p[0], p[1]) for p in array}))
    ax2.plot_surface(ox, oy, f(ox, oy))

    plt.subplots_adjust(bottom=0.4)
    ax_lr = plt.axes([0.15, 0.15, 0.75, 0.03])
    lr_slider = Slider(
        ax=ax_lr,
        label="Learning rate",
        valmin=0.01,
        valmax=1,
        valinit=0.01,
        valstep=0.01
    )


    def update(val):
        global GDContour
        points = gd(lr_slider.val, file_writer)
        #epochs_slider.val = epochs
        GDLine.set_data(points[:, 0], points[:, 1])
        for coll in GDContour.collections:
            coll.remove()
        GDContour = ax1.contour(ox, oy, f(ox, oy), levels=sorted({f(p[0], p[1]) for p in points}))
        fig.canvas.draw_idle()
        plt.savefig("1/" + "{:.2f}".format(lr_slider.val) + ".png")


    lr_slider.on_changed(update)
    plt.show()
