import math

import numpy as np
import matplotlib.pyplot as plt


def f(x, y, z):
    return ((x + 1) * z - y) / x


def g(x, y, z):
    return z


def exact_solution(x):
    return x + 1 + (math.e ** x)


def euler_method(f, g, y0, z0, interval, h):
    l, r = interval
    x = [i for i in np.arange(l, r + h, h)]
    y = [y0]
    z = z0
    for i in range(len(x) - 1):
        z += h * f(x[i], y[i], z)
        y.append(y[i] + h * g(x[i], y[i], z))
    return x, y


def runge_kutta_method(f, g, y0, z0, interval, h, return_z=False, exclude_num=None):
    l, r = interval
    x = [i for i in np.arange(l, r + h, h)]
    if exclude_num:
        x.pop(x.index(exclude_num))
    y = [y0]
    z = [z0]
    for i in range(len(x) - 1):
        K1 = h * g(x[i], y[i], z[i])
        L1 = h * f(x[i], y[i], z[i])
        K2 = h * g(x[i] + 0.5 * h, y[i] + 0.5 * K1, z[i] + 0.5 * L1)
        L2 = h * f(x[i] + 0.5 * h, y[i] + 0.5 * K1, z[i] + 0.5 * L1)
        K3 = h * g(x[i] + 0.5 * h, y[i] + 0.5 * K2, z[i] + 0.5 * L2)
        L3 = h * f(x[i] + 0.5 * h, y[i] + 0.5 * K2, z[i] + 0.5 * L2)
        K4 = h * g(x[i] + h, y[i] + K3, z[i] + L3)
        L4 = h * f(x[i] + h, y[i] + K3, z[i] + L3)
        delta_y = (K1 + 2 * K2 + 2 * K3 + K4) / 6
        delta_z = (L1 + 2 * L2 + 2 * L3 + L4) / 6
        y.append(y[i] + delta_y)
        z.append(z[i] + delta_z)

    if not return_z:
        return x, y
    else:
        return x, y, z


def adams_method(f, g, y0, z0, interval, h):
    x_runge, y_runge, z_runge = runge_kutta_method(f, g, y0, z0, interval, h, return_z=True)
    x = x_runge
    y = y_runge[:4]
    z = z_runge[:4]
    for i in range(3, len(x_runge) - 1):
        z_i = z[i] + h * (55 * f(x[i], y[i], z[i]) -
                          59 * f(x[i - 1], y[i - 1], z[i - 1]) +
                          37 * f(x[i - 2], y[i - 2], z[i - 2]) -
                          9 * f(x[i - 3], y[i - 3], z[i - 3])) / 24
        z.append(z_i)
        y_i = y[i] + h * (55 * g(x[i], y[i], z[i]) -
                          59 * g(x[i - 1], y[i - 1], z[i - 1]) +
                          37 * g(x[i - 2], y[i - 2], z[i - 2]) -
                          9 * g(x[i - 3], y[i - 3], z[i - 3])) / 24
        y.append(y_i)
    return x, y


def runge_romberg_method(h1, h2, y1, y2, p): # p - порядок точности
    assert h1 == h2 * 2
    norm = 0
    for i in range(len(y1)):
        norm += y1[i] - y2[i * 2]
    return norm / (2**p - 1)


def mae(y1, y2):
    assert len(y1) == len(y2)
    res = 0
    for i in range(len(y1)):
        res += abs(y1[i] - y2[i])
    return res / len(y1)


if __name__ == '__main__':
    y0 = 2 + math.e
    dy0 = 1 + math.e
    interval = (1, 2)
    h = 0.1

    x_euler, y_euler = euler_method(f, g, y0, dy0, interval, h)
    plt.plot(x_euler, y_euler, label=f'Метод Эйлера, шаг={h}')
    x_euler2, y_euler2 = euler_method(f, g, y0, dy0, interval, h/2)
    plt.plot(x_euler2, y_euler2, label=f'Метод Эйлера, шаг={h/2}')

    x_runge, y_runge = runge_kutta_method(f, g, y0, dy0, interval, h)
    plt.plot(x_runge, y_runge, label=f'Метод Рунге-Кутта, шаг={h}')
    x_runge2, y_runge2 = runge_kutta_method(f, g, y0, dy0, interval, h/2)
    plt.plot(x_runge2, y_runge2, label=f'Метод Рунге-Кутта, шаг={h/2}')

    x_adams, y_adams = adams_method(f, g, y0, dy0, interval, h)
    plt.plot(x_adams, y_adams, label=f'Метод Адамса, шаг={h}')
    x_adams2, y_adams2 = adams_method(f, g, y0, dy0, interval, h/2)
    plt.plot(x_adams2, y_adams2, label=f'Метод Адамса, шаг={h/2}')

    x_exact = [i for i in np.arange(interval[0], interval[1] + h, h)]
    x_exact2 = [i for i in np.arange(interval[0], interval[1] + h/2, h/2)]
    y_exact = [exact_solution(x_i) for x_i in x_exact]
    y_exact2 = [exact_solution(x_i) for x_i in x_exact2]
    plt.plot(x_exact, y_exact, label='Точное решение')

    print('Значения абсолютных ошибок')
    print(f'шаг = {h}')
    print('Метод Эйлера: ', mae(y_euler, y_exact))
    print('Метод Рунге-Кутта: ', mae(y_runge, y_exact))
    print('Метод Адамса: ', mae(y_adams, y_exact))
    print(f'шаг = {h/2}')
    print('Метод Эйлера: ', mae(y_euler2, y_exact2))
    print('Метод Рунге-Кутта: ', mae(y_runge2, y_exact2))
    print('Метод Адамса: ', mae(y_adams2, y_exact2))
    print()

    print('Оценка погрешности методом Рунге-Румберга')
    print('Метод Эйлера: ', runge_romberg_method(h, h / 2, y_euler, y_euler2, 1))
    print('Метод Рунге-Кутта: ', runge_romberg_method(h, h / 2, y_runge, y_runge2, 4))
    print('Метод Адамса: ', runge_romberg_method(h, h / 2, y_adams, y_adams2, 4))

    plt.legend()
    plt.show()
