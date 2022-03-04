import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from itertools import count
import csv


plt.rcParams["figure.figsize"] = (7, 7)
np.set_printoptions(suppress=True)

t = np.linspace(0.001, 1, 200)
ox, oy = np.meshgrid(t, t)

phi = np.divide(np.add(1, np.sqrt(5)), 2)
e = 0.01
startPoint = [0.01, 0.9]
init_epochs = 20
eps = 0.0000000000001
fc = 0
def f(x, y):
    return np.log(x * x + y * y) * x * y


def fdy(x, y):
    return np.log(x * x + y * y) * x + 2 * x * y * y / (x * x + y * y)


def fdx(x, y):
    return np.log(x * x + y * y) * y + 2 * x * x * y / (x * x + y * y)


def grad(point):
    return np.array([fdx(*point), fdy(*point)])

with open("3/3_stat.csv", mode="w", encoding='utf-8') as w_file:
    file_writer = csv.writer(w_file, delimiter = ",", lineterminator="\r")
    file_writer.writerow(["e", "Grad", "Func"])


    def next_value_by_golden_ratio(point):
        global fc
        fc = 1
        a = point
        v = -grad(point)
        d = 0.01
        b = a
        tmp = f(*a)
        while f(*b) <= tmp:
            tmp = f(*b)
            d *= phi
            b = point + d * v
            fc += 2
        while np.linalg.norm(b - a) > e:
            dist = (b - a) / phi
            x1 = b - dist
            x2 = a + dist
            fx1 = f(*x1)
            fx2 = f(*x2)
            if fx1 >= fx2:
                a = x1
            else:
                b = x2
            fc += 2
        return (a + b) / 2


    def gd(e, writer=file_writer):
        points = np.array([startPoint])
        epochs = 1
        global fc
        fc = 0
        points = np.append(points, [next_value_by_golden_ratio(points[0])], axis=0)
        for i in count(start=2, step=1):
            if abs(f(points[i - 1][0], points[i - 1][1]) - f(points[i - 2][0], points[i - 2][1])) < eps:
                break
            if i > 10000000:
                print("Ne povezlo ne fortanulo")
                break

            points = np.append(points, [next_value_by_golden_ratio(points[i-1])], axis=0)
            epochs += 1
        print(points)
        writer.writerow(["{:.2f}".format(e), epochs, fc])
        return points


    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')
    array = gd(e, file_writer)
    GDLine, = ax1.plot(array[:, 0], array[:, 1], 'o-')
    GDContour = ax1.contour(ox, oy, f(ox, oy), levels=sorted({f(p[0], p[1]) for p in array}))
    ax2.plot_surface(ox, oy, f(ox, oy))

    plt.subplots_adjust(bottom=0.3)
    ax_e = plt.axes([0.15, 0.15, 0.75, 0.03])

    e_slider = Slider(
        ax=ax_e,
        label="e for\nGolden ratio",
        valmin=0.005,
        valmax=0.45,
        valinit=e,
        valstep=0.005
    )


    def update(_):
        global GDContour, e
        e = e_slider.val
        points = gd(e, file_writer)
        GDLine.set_data(points[:, 0], points[:, 1])
        for coll in GDContour.collections:
            coll.remove()
        GDContour = ax1.contour(ox, oy, f(ox, oy), levels=sorted({f(p[0], p[1]) for p in points}))
        fig.canvas.draw_idle()

    e_slider.on_changed(update)
    plt.show()
