from enum import Enum

import numpy as np
import matplotlib.pyplot as plt

coef_a = 0.5
Psi = lambda x: np.sin(x)
Phi0 = lambda t: np.exp(-coef_a * t)
PhiL = lambda t: -np.exp(-coef_a * t)
TrueSolution = lambda x, t: np.exp(-coef_a * t) * np.sin(x)
l = np.pi

T, K, N = 10, 2000, 30
h = l / N
tau = T / K
sigma = tau * coef_a ** 2 / h ** 2
assert sigma <= 0.5, sigma


class Approximation(Enum):
    FIRST_DEGREE_TWO_DOTS = 1,
    SECOND_DEGREE_TWO_DOTS = 2,
    SECOND_DEGREE_THREE_DOTS = 3,


APPROXIMATION = Approximation.SECOND_DEGREE_TWO_DOTS


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


def explicit_step(h, k, tau, u, u_explicit):
    if APPROXIMATION == Approximation.FIRST_DEGREE_TWO_DOTS:
        u_explicit[0] = u[k][1] - h * Phi0((k + 1) * tau)
        u_explicit[-1] = u[k][-2] + h * PhiL((k + 1) * tau)
    elif APPROXIMATION == Approximation.SECOND_DEGREE_TWO_DOTS:
        u_explicit[0] = (Phi0(k * tau) + u[k][2] / (2 * h) - 2 * u[k][1] / h) * ((2 * h) / -3)
        u_explicit[-1] = (PhiL(k * tau) - u[k][-3] / (2 * h) + 2 * u[k][-2] / h) * ((2 * h) / 3)
    elif APPROXIMATION == Approximation.SECOND_DEGREE_THREE_DOTS:
        u_explicit[0] = (u[k][1] - h * Phi0(k * tau) + (h ** 2 / 2 / tau) * u[k - 1][0]) / (1 + h ** 2 / 2 / tau)
        u_explicit[-1] = (u[k][-2] + h * PhiL(k * tau) + (h ** 2 / 2 / tau) * u[k - 1][-1]) / (1 + h ** 2 / 2 / tau)


def implicit_step(N, h, k, sigma, tau, u):
    a = np.zeros(N + 1)
    b = np.zeros(N + 1)
    c = np.zeros(N + 1)
    d = np.zeros(N + 1)
    for j in range(1, N):
        a[j] = sigma
        b[j] = -(1 + 2 * sigma)
        c[j] = sigma
        d[j] = -u[k][j]
    if APPROXIMATION == Approximation.FIRST_DEGREE_TWO_DOTS:
        b[0] = -1 / h
        c[0] = 1 / h
        d[0] = -PhiL((k + 1) * tau)

        a[-1] = -1 / h
        b[-1] = 1 / h
        d[-1] = PhiL((k + 1) * tau)

    elif APPROXIMATION == Approximation.SECOND_DEGREE_TWO_DOTS:
        b[0] = (-3 / (h * 2)) + a[1] / (2 * h) / c[1]
        c[0] = 2 / h + b[1] / (2 * h) / c[1]
        d[0] = -PhiL((k + 1) * tau) + d[1] / (2 * h) / c[1]

        a[-1] = (-2 / h) - b[-2] / (h * 2) / a[-2]
        b[-1] = (3 / (h * 2)) - c[-2] / (h * 2) / a[-2]
        d[-1] = PhiL((k + 1) * tau) - d[-2] / (h * 2) / a[-2]

    elif APPROXIMATION == Approximation.SECOND_DEGREE_THREE_DOTS:
        b[0] = 2 * coef_a ** 2 / h + h / tau
        c[0] = - 2 * coef_a ** 2 / h
        d[0] = (h / tau) * u[k - 1][0] + PhiL((k + 1) * tau) * 2 * coef_a ** 2

        a[-1] = -2 * coef_a ** 2 / h
        b[-1] = 2 * coef_a ** 2 / h + h / tau
        d[-1] = (h / tau) * u[k - 1][-1] + PhiL((k + 1) * tau) * 2 * coef_a ** 2
    return a, b, c, d


def explicit(N, K, T, h, tau, sigma):
    u = np.zeros((K + 1, N + 1))
    for j in range(N + 1):
        u[0][j] = Psi(j * h)
    for k in range(1, K + 1):
        for j in range(1, N):
            u[k][j] = sigma * (u[k - 1][j + 1] + u[k - 1][j - 1]) + (1 - 2 * sigma) * u[k - 1][j]
        explicit_step(h, k, tau, u, u[k])
    return u


def implicit(N, K, T, h, tau, sigma):
    u = np.zeros((K + 1, N + 1))
    for j in range(N + 1):
        u[0][j] = Psi(j * h)

    for k in range(K):
        a, b, c, d = implicit_step(N, h, k, sigma, tau, u)

        u[k + 1] = three_diag(a, b, c, d)
    return u


def crank_nicholson(N, K, T, h, tau, sigma):
    theta = 0.5
    u = np.zeros((K + 1, N + 1))

    for j in range(1, N):
        u[0][j] = Psi(j * h)

    for k in range(K):
        a, b, c, d = implicit_step(N, h, k, sigma, tau, u)

        u_implisit = three_diag(a, b, c, d)
        u_explicit = np.zeros(N + 1)

        for j in range(1, N):
            u_explicit[j] = sigma * (u[k][j + 1] + u[k][j - 1]) + (1 - 2 * sigma) * u[k][j]

        explicit_step(h, k, tau, u, u_explicit)

        for j in range(N):
            u[k + 1][j] = theta * u_implisit[j] + (1 - theta) * u_explicit[j]

    return u


def analytical_solution(N, K, T):
    h = l / N
    tau = T / K
    u = np.zeros((K + 1, N + 1))
    for k in range(K + 1):
        for j in range(N + 1):
            u[k][j] = TrueSolution(j * h, k * tau)
    return u


def mae(numeric, analytic):
    err = []
    for i in range(len(numeric)):
        err.append(abs(numeric[i] - analytic[i]).mean())
    return np.array(err)


def draw_charts(answers, N, K, T, time=8):
    x = np.linspace(0, l, num=N + 1)
    t = np.linspace(0, T, num=K + 1)
    z1 = np.array(answers['Analytic'])
    z2 = np.array(answers['uExplicit'])
    z3 = np.array(answers['Implicit'])
    z4 = np.array(answers['Crank Nicholson'])
    plt.title('U от x')
    plt.plot(x, z1[time], color='r', label='Analytic')
    plt.plot(x, z2[time], color='g', label='uExplicit')
    plt.plot(x, z3[time], color='b', label='Implicit')
    plt.plot(x, z4[time], color='y', label='Crank Nicholson')
    plt.grid()
    plt.legend(loc='best')
    plt.ylabel('U')
    plt.xlabel('x')
    plt.show()

    plt.title('Зависимость ошибки от времени')
    plt.plot(t, mae(answers['uExplicit'], answers['Analytic']), label='uExplicit')
    plt.plot(t, mae(answers['Implicit'], answers['Analytic']), label='Implicit')
    plt.plot(t, mae(answers['Crank Nicholson'], answers['Analytic']), label='Crank Nicholson')
    plt.legend(loc='best')
    plt.ylabel('Ошибка')
    plt.xlabel('t')
    plt.grid()
    plt.show()


answers = dict()

answers['Analytic'] = analytical_solution(N, K, T)
answers['Implicit'] = implicit(N, K, T, h, tau, sigma)
answers['uExplicit'] = explicit(N, K, T, h, tau, sigma)
answers['Crank Nicholson'] = crank_nicholson(N, K, T, h, tau, sigma)

draw_charts(answers, N, K, T)
