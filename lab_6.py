from enum import Enum

import numpy as np
import matplotlib.pyplot as plt

A = 1
B = 0
C = -5

l = 1
Function = lambda: 0
alpha = 1
beta = -2
gamma = 1
delta = -2

Psi1 = lambda x: np.exp(2 * x)
Psi2 = lambda x: 0
Psi1_der1 = lambda x: 2 * np.exp(2 * x)
Psi1_der2 = lambda x: 4 * np.exp(2 * x)
Phi0 = lambda t: 0
PhiL = lambda t: 0

AnaliticalSolution = lambda x, t: np.exp(2 * x) * np.cos(t)


class Approximation(Enum):
    FIRST_DEGREE_TWO_DOTS = 1,
    SECOND_DEGREE_TWO_DOTS = 2,
    SECOND_DEGREE_THREE_DOTS = 3,


APPROXIMATION = Approximation.FIRST_DEGREE_TWO_DOTS

T, K, N = 10, 2000, 30
h = l / N
tau = T / K
sigma = (A * tau / h) ** 2


def three_diag(a, b, c, d):
    size = len(a)
    p = np.zeros(size)
    q = np.zeros(size)
    p[0] = (-c[0] / b[0])
    q[0] = (d[0] / b[0])

    for i in range(1, size):
        p[i] = -c[i] / (b[i] + a[i] * p[i - 1])
        q[i] = (d[i] - a[i] * q[i - 1]) / (b[i] + a[i] * p[i - 1])

    x = np.zeros(size)
    x[-1] = q[-1]

    for i in range(size - 2, -1, -1):
        x[i] = p[i] * x[i + 1] + q[i]

    return x


def explicit_step(h, k, tau, u):
    if APPROXIMATION == Approximation.FIRST_DEGREE_TWO_DOTS:
        u[k][0] = (Phi0(k * tau) - u[k][1] * alpha / h) / (beta - alpha / h)
        u[k][-1] = (PhiL(k * tau) + u[k][-2] * gamma / h) / (delta + gamma / h)

    elif APPROXIMATION == Approximation.SECOND_DEGREE_TWO_DOTS:
        u[k][0] = (Phi0(k * tau) + u[k][2] * alpha / (2 * h) -
                   u[k][1] * 2 * alpha / h) / (-3 * alpha / (2 * h) + beta)

        u[k][-1] = (PhiL(k * tau) - u[k][-3] * alpha / (2 * h) +
                    u[k][-2] * 2 * alpha / h) / (3 * alpha / (2 * h) + beta)

    elif APPROXIMATION == Approximation.SECOND_DEGREE_THREE_DOTS:
        u[k][0] = (Phi0(k * tau) - u[k - 1][0] * alpha * h / (tau * 2) -
                   u[k][1] * alpha * 2 * A / (h * 2)) / (alpha * (-2 * A / (h * 2) -
                                                                  h / (tau * 2) + C * h / 2) + beta)

        u[k][-1] = (PhiL(k * tau) + alpha * (h * u[k - 1][-1] / (2 * tau) +
                                             u[k][-2] * 2 * A / (h * 2))) / \
                   (alpha * (2 * A / (h * 2) + h / (2 * tau) - C * h / 2) + beta)


def implicit_step(a, b, c, d, h, k, tau, u):
    if APPROXIMATION == Approximation.FIRST_DEGREE_TWO_DOTS:
        b[0] = beta - alpha / h
        c[0] = alpha / h
        d[0] = Phi0(k * tau)

        a[-1] = -gamma / h
        b[-1] = delta + gamma / h
        d[-1] = PhiL(k * tau)

    elif APPROXIMATION == Approximation.SECOND_DEGREE_TWO_DOTS:
        c[0] = 2 * alpha / h + b[1] * (alpha / (2 * h * c[1]))
        b[0] = (-3 * alpha / (2 * h) + beta) + a[1] * (alpha / (2 * h * c[1]))
        d[0] = Phi0(k * tau) + d[1] * (alpha / (2 * h * c[1]))

        a[-1] = (-2 * alpha / h) + b[-2] * (-(alpha / (h * 2)) / a[-2])
        b[-1] = (3 * alpha / (2 * h) + beta) + c[-2] * (-(alpha / (h * 2)) / a[-2])
        d[-1] = PhiL((k + 1) * tau) + d[-2] * (-(alpha / (h * 2)) / a[-2])

    elif APPROXIMATION == Approximation.SECOND_DEGREE_THREE_DOTS:
        b[0] = (alpha * (-2 * A / (h * 2) -
                         h / (tau * 2) + C * h / 2) + beta)
        c[0] = alpha * 2 * A / (h * 2)
        d[0] = (Phi0(k * tau) -
                u[k - 1][0] * alpha * h / (tau * 2))

        a[-1] = -alpha * 2 * A / (h * 2)
        b[-1] = alpha * (2 * A / (h * 2) +
                         h / (tau * 2) - C * h / 2) + beta
        d[-1] = (PhiL(k * tau) +
                 u[k - 1][-1] * alpha * h / (tau * 2))


def explicit(N, K, T, h, tau):
    u = np.zeros((K + 1, N + 1))
    for j in range(N + 1):
        u[0][j] = Psi1(j * h)
        u[1][j] = u[0][j] + Psi2(j * h) * tau + A ** 2 * Psi1_der2(j * h) * ((tau ** 2) / 2)

    for k in range(2, K + 1):
        for j in range(1, N):
            u[k][j] = u[k - 1][j + 1] * ((A * tau / h) ** 2 + B * tau ** 2 / (2 * h)) \
                      + u[k - 1][j] * (-2 * (A * tau / h) ** 2 + 2 + C * tau ** 2) \
                      + u[k - 1][j - 1] * ((A * tau / h) ** 2 - B * tau ** 2 / (2 * h)) \
                      - u[k - 2][j] + tau ** 2 * Function()

        explicit_step(h, k, tau, u)

    return u


def implicit(N, K, T, h, tau):
    u = np.zeros((K + 1, N + 1))
    for j in range(N + 1):
        u[0][j] = Psi1(j * h)
        u[1][j] = u[0][j] + Psi2(j * h) * tau + A ** 2 * Psi1_der2(j * h) * ((tau ** 2) / 2)

    for k in range(2, K + 1):
        a = np.zeros(N + 1)
        b = np.zeros(N + 1)
        c = np.zeros(N + 1)
        d = np.zeros(N + 1)

        for j in range(1, N):
            a[j] = (A * tau / h) ** 2
            b[j] = -(1 + 2 * (A * tau / h) ** 2)
            c[j] = (A * tau / h) ** 2
            d[j] = u[k - 2][j] - (C * tau ** 2 + 2) * u[k - 1][j]

        implicit_step(a, b, c, d, h, k, tau, u)

        u[k] = three_diag(a, b, c, d)

    return u


def analytical_solution(N, K, T):
    h = l / N
    tau = T / K
    u = np.zeros((K + 1, N + 1))
    for k in range(K + 1):
        for j in range(N + 1):
            u[k][j] = AnaliticalSolution(j * h, k * tau)
    return u


def mae(numeric, analytic):
    err = []
    for i in range(len(numeric)):
        err.append(abs(numeric[i] - analytic[i]).mean())
    return np.array(err)


def draw_charts(answers, N, K, T, time=5):
    x = np.linspace(0, 1, num=N + 1)
    t = np.linspace(0, T, num=K + 1)
    z1 = np.array(answers['Analytic'])
    z2 = np.array(answers['Explicit'])
    z3 = np.array(answers['Implicit'])

    plt.title('U от x')
    plt.plot(x, z1[time], color='r', label='Analytic')
    plt.plot(x, z2[time], color='g', label='Explicit')
    plt.plot(x, z3[time], color='b', label='Implicit')
    plt.legend(loc='best')
    plt.ylabel('U')
    plt.xlabel('x')
    plt.grid()
    plt.show()

    plt.title('Зависимость ошибки от времени')
    plt.plot(t, mae(answers['Explicit'], answers['Analytic']), label='Explicit')
    plt.plot(t, mae(answers['Implicit'], answers['Analytic']), label='Implicit')
    plt.legend(loc='best')
    plt.ylabel('Ошибка')
    plt.xlabel('t')
    plt.grid()
    plt.show()


answers = dict()

answers['Analytic'] = analytical_solution(N, K, T)
answers['Explicit'] = explicit(N, K, T, h, tau)
answers['Implicit'] = implicit(N, K, T, h, tau)

draw_charts(answers, N, K, T)
