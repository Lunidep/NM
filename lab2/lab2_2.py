import math
import numpy as np

from lab1.lab1_1 import inverse_matrix


def f1(X):
    return 3 * X[0] - math.cos(X[1])


def f2(X):
    return 3 * X[1] - (math.e ** X[0])


def df1_dx1(X):
    return 3


def df1_dx2(X):
    return math.sin(X[1])


def df2_dx1(X):
    return -(math.e ** X[0])


def df2_dx2(X):
    return 3


def phi1(X):
    return math.cos(X[1]) / 3


def phi2(X):
    return (math.e ** X[0]) / 3


def dphi1_dx1(X):
    return 0


def dphi1_dx2(X):
    return -math.sin(X[1]) / 3


def dphi2_dx1(X):
    return (math.e ** X[0]) / 3


def dphi2_dx2(X):
    return 0


def L_inf_norm(a):
    abs_a = [abs(i) for i in a]
    return max(abs_a)


def get_q(interval1, interval2):
    l1, r1 = interval1
    l2, r2 = interval2
    x1 = (l1 + r1) / 2
    x2 = (l2 + r2) / 2
    max1 = abs(dphi1_dx1([x1, x2])) + abs(dphi1_dx2([x1, x2]))
    max2 = abs(dphi2_dx1([x1, x2])) + abs(dphi2_dx2([x1, x2]))
    return max(max1, max2)


def iteration_method(phi1, phi2, intervals, eps):
    l1, r1 = intervals[0][0], intervals[0][1]
    l2, r2 = intervals[1][0], intervals[1][1]
    x_prev = [(l1 + r1) * 0.5, (l2 + r2) * 0.5]
    q = get_q(intervals[0], intervals[1])
    iters = 0
    while True:
        iters += 1
        x = [phi1(x_prev), phi2(x_prev)]
        if q / (1 - q) * L_inf_norm([(x[i] - x_prev[i]) for i in range(len(x))]) < eps:
            break
        x_prev = x

    return x, iters


def newton_method(f1, f2, df1_dx1, df1_dx2, df2_dx1, df2_dx2, intervals, eps):
    l1, r1 = intervals[0][0], intervals[0][1]
    l2, r2 = intervals[1][0], intervals[1][1]
    x_prev = [(l1 + r1) / 2, (l2 + r2) / 2]
    jacobi = []
    jacobi.append([df1_dx1(x_prev), df1_dx2(x_prev)])
    jacobi.append([df2_dx1(x_prev), df2_dx2(x_prev)])
    jacobi_inversed = inverse_matrix(jacobi)
    iters = 0
    while True:
        iters += 1
        x = x_prev - jacobi_inversed @ np.array([f1(x_prev), f2(x_prev)])
        if L_inf_norm([(x[i] - x_prev[i]) for i in range(len(x))]) < eps:
            break
        x_prev = x

    return x, iters


if __name__ == "__main__":
    l1, r1 = 0, 1
    l2, r2 = 0, 1
    eps = 0.0000001

    print('Метод итераций')
    x_iter, i_iter = iteration_method(phi1, phi2, [(l1, r1), (l2, r2)], eps)
    print('x =', x_iter, '; f1(x) =', f1(x_iter), '; f2(x) =', f2(x_iter))
    print('Кол-во итераций', i_iter)

    print('Метод Ньютона')
    x_newton, i_newton = newton_method(f1, f2, df1_dx1, df1_dx2, df2_dx1, df2_dx2, [(l1, r1), (l2, r2)], eps)
    print('x =', x_newton, '; f1(x) =', f1(x_newton), '; f2(x)=', f2(x_newton))
    print('Кол-во итераций:', i_newton)

