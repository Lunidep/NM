import numpy as np
import math
import matplotlib.pyplot as plt

from lab1.lab1_2 import tridiagonal_solve
from lab3.lab3_4 import df
from lab4.lab4_1 import runge_kutta_method, runge_romberg_method, mae


def f(x, y, z):
    return (y - (x - 3) * z) / (x ** 2 - 1)


def g(x, y, z):
    return z


def p_fd(x):
    return (x - 3) / (x ** 2 - 1)


def q_fd(x):
    return -1 / (x ** 2 - 1)


def f_fd(x):
    return 0


def exact_solution(x):
    return x - 3 + (1 / (x + 1))


def get_n(n_prev, n, ans_prev, ans, b, delta, gamma, y1):
    x, y = ans_prev[0], ans_prev[1]
    y_der = df(b, x, y)
    phi_n_prev = delta * y[-1] + gamma * y_der - y1
    x, y = ans[0], ans[1]
    y_der = df(b, x, y)
    phi_n = delta * y[-1] + gamma * y_der - y1
    return n - (n - n_prev) / (phi_n - phi_n_prev) * phi_n


def check_finish(x, y, b, delta, gamma, y1, eps):
    y_der = df(b, x, y)
    return abs(delta * y[-1] + gamma * y_der - y1) > eps


def shooting_method(f, g, alpha, beta, delta, gamma, y0, y1, interval, h, eps):
    a, b = interval[0], interval[1]
    n_prev, n = 1.0, 0.8
    y_der = (y0 - alpha * n_prev) / beta
    x_prev, y_prev = runge_kutta_method(f, g, n_prev, y_der, (a, b), h)
    y_der = (y0 - alpha * n) / beta
    x, y = runge_kutta_method(f, g, n, y_der, (a, b), h)
    iterations = 0

    while check_finish(x, y, b, delta, gamma, y1, eps):
        n, n_prev = get_n(n_prev, n, (x_prev, y_prev), (x, y), b, delta, gamma, y1), n
        x_prev, y_prev = x, y
        y_der = (y0 - alpha * n) / beta
        x, y = runge_kutta_method(f, g, n, y_der, (a, b), h)
        iterations += 1

    return x, y, iterations


def finite_difference_method(p, q, f, y0, yn, alpha, beta, delta, gamma, interval, h):
    A = []
    B = []
    rows = []
    a, b = interval
    x = np.arange(a, b + h, h)
    n = len(x)

    for i in range(n):
        if i == 0:
            rows.append(alpha * h - beta)
        elif i == 1:
            rows.append(beta)
        else:
            rows.append(0)
    A.append(rows)
    B.append(y0 * h)

    for i in range(1, n - 1):
        rows = []
        B.append(f(x[i]) * h ** 2)
        for j in range(n):
            if j == i - 1:
                rows.append(1 - p(x[i]) * h / 2)
            elif j == i:
                rows.append(q(x[i]) * h ** 2 - 2)
            elif j == i + 1:
                rows.append(1 + p(x[i]) * h / 2)
            else:
                rows.append(0)
        A.append(rows)

    rows = []
    B.append(yn * h)
    for i in range(n):
        if i == n - 1:
            rows.append(delta * h + gamma)
        elif i == n - 2:
            rows.append(-gamma)
        else:
            rows.append(0)

    A.append(rows)
    y = tridiagonal_solve(A, B)
    return x, y


if __name__ == '__main__':
    interval = (0.000001, 1)
    y0 = 0
    y1 = -0.75
    h = 0.1
    eps = 0.001
    alpha, beta, delta, gamma = 0, 1, 1, 1

    x_shooting, y_shooting, iters_shooting = shooting_method(f, g, alpha, beta, delta, gamma, y0, y1, interval, h, eps)
    plt.plot(x_shooting, y_shooting, label=f'Метод стрельбы, шаг={h}')
    x_shooting2, y_shooting2, iters_shooting2 = shooting_method(f, g, alpha, beta, delta, gamma, y0, y1, interval, h / 2, eps)
    plt.plot(x_shooting2, y_shooting2, label=f'Метод стрельбы, шаг={h / 2}')

    x_fd, y_fd = finite_difference_method(p_fd, q_fd, f_fd, y0, y1, alpha, beta, delta, gamma, interval, h)
    plt.plot(x_fd, y_fd, label=f'finite difference method, шаг={h}')
    x_fd2, y_fd2 = finite_difference_method(p_fd, q_fd, f_fd, y0, y1, alpha, beta, delta, gamma, interval, h / 2)
    plt.plot(x_fd2, y_fd2, label=f'finite difference method, шаг={h / 2}')

    x_exact = [i for i in np.arange(interval[0], interval[1] + h, h)]
    x_exact2 = [i for i in np.arange(interval[0], interval[1] + h / 2, h / 2)]
    y_exact = [exact_solution(x_i) for x_i in x_exact]
    y_exact2 = [exact_solution(x_i) for x_i in x_exact2]
    plt.plot(x_exact, y_exact, label='Точное решение')

    print('Итерации')
    print(f'шаг = {h}')
    print('Метод стрельбы: ', iters_shooting)
    print(f'шаг = {h / 2}')
    print('Метод стрельбы: ', iters_shooting2)
    print()

    print('Значения абсолютных ошибок')
    print(f'шаг = {h}')
    print('Конечно-разностный метод решения: ', mae(y_fd, y_exact))
    print(f'шаг = {h / 2}')
    print('Конечно-разностный метод решения: ', mae(y_fd2, y_exact2))
    print(f'шаг = {h}')
    print('Метод стрельбы: ', mae(y_shooting, y_exact))
    print(f'шаг = {h / 2}')
    print('Метод стрельбы: ', mae(y_shooting2, y_exact2))
    print()

    print('Оценка погрешности методом Рунге-Румберга')
    print('Метод стрельбы: ', runge_romberg_method(h, h / 2, y_shooting, y_shooting2, 1))
    print('Конечно-разностный метод решения: ', runge_romberg_method(h, h / 2, y_fd, y_fd2, 4))

    plt.legend()
    plt.show()
