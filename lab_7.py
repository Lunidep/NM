import numpy as np
import copy
import matplotlib.pyplot as plt

A = 0
B = 0
C = 0
D = 1

lx = np.pi
ly = 1

phi_1 = lambda y: np.exp(y)
phi_2 = lambda y: -np.exp(y)
phi_3 = lambda x: np.sin(x)
phi_4 = lambda x: np.e * np.sin(x)

analytic_solution = lambda x, y: np.sin(x) * np.exp(y)
ANALYTIC_SOLUTION_METHOD_NAME = 'Analytic'
free_member = lambda x, y: 0

nx = 10
ny = 10
hx = lx / nx
hy = ly / ny
MAX_ITERS_COUNT = 10000
eps = 1e-6


def get_accurancy(u, L):
    return np.max([abs(u[i][j] - L[i][j]) for i in range(nx + 1) for j in range(ny + 1)])


def simple_iteration_method():
    x = np.arange(0, lx + hx, hx)
    y = np.arange(0, ly + hy, hy)
    u = np.zeros((len(x), len(y)))

    for i in range(len(y)):
        u[0][i] = phi_1(y[i])
        u[-1][i] = phi_2(y[i])
    for i in range(len(x)):
        u[i][0] = phi_3(x[i])
        u[i][-1] = phi_4(x[i])

    iterations = 0
    while iterations < MAX_ITERS_COUNT:
        L = copy.deepcopy(u)
        iterations += 1

        for j in range(1, len(u[0]) - 1):
            u[0][j] = u[1][j] - hx * phi_1(y[j])
            u[-1][j] = u[-2][j] + hx * phi_2(y[j])

        for i in range(1, len(u) - 1):
            for j in range(1, len(u[0]) - 1):
                u[i][j] = (hx ** 2 * free_member(x[i], y[j]) - (
                        L[i + 1][j] + L[i - 1][j]) - D * hx ** 2 * (
                                   L[i][j + 1] + L[i][j - 1]) / (hy ** 2) - A * hx * 0.5 * (
                                   L[i + 1][j] - L[i - 1][j]) - B * hx ** 2 * (
                                   L[i][j + 1] - L[i][j - 1]) / (2 * hy)) / (
                                  C * hx ** 2 - 2 * (hy ** 2 + D * hx ** 2) / (
                                  hy ** 2))

        if get_accurancy(u, L) <= eps:
            break
    return u.transpose(), iterations


def simple_iteration_relaxation_method(param_relax=1.5):
    x = np.arange(0, lx + hx, hx)
    y = np.arange(0, ly + hy, hy)
    u = np.zeros((len(x), len(y)))

    for i in range(len(y)):
        u[0][i] = phi_1(y[i])
        u[-1][i] = phi_2(y[i])
    for i in range(len(x)):
        u[i][0] = phi_3(x[i])
        u[i][-1] = phi_4(x[i])

    iterations = 0
    while iterations < MAX_ITERS_COUNT:
        L = copy.deepcopy(u)
        iterations += 1

        for j in range(1, len(y) - 1):
            u[0][j] = u[1][j] - hx * phi_1(y[j])
            u[-1][j] = u[-2][j] + hx * phi_2(y[j])

        for i in range(1, len(u) - 1):
            for j in range(1, len(u[0]) - 1):
                new_u = (hx ** 2 * free_member(x[i], y[j]) - (L[i + 1][j] + u[i - 1][j]) - D * hx ** 2 * (
                        L[i][j + 1] + u[i][j - 1]) / (hy ** 2) - A * hx * 0.5 * (
                                 L[i + 1][j] - u[i - 1][j]) - B * hx ** 2 * (
                                 L[i][j + 1] - u[i][j - 1]) / (2 * hy)) / (
                                C * hx ** 2 - 2 * (hy ** 2 + D * hx ** 2) / (
                                hy ** 2))
                u[i][j] = (1 - param_relax) * L[i][j] + param_relax * new_u

        if get_accurancy(u, L) <= eps:
            break
    return u.transpose(), iterations


# def seidel_method():
#     return simple_iteration_relaxation_method(param_relax = 1.0)

def seidel_method():
    hx = lx / nx
    hy = ly / ny
    x = np.arange(0, lx + hx, hx)
    y = np.arange(0, ly + hy, hy)
    u = np.zeros((len(x), len(y)))

    for i in range(len(y)):
        u[0][i] = phi_1(y[i])
        u[-1][i] = phi_2(y[i])
    for i in range(len(x)):
        u[i][0] = phi_3(x[i])
        u[i][-1] = phi_4(x[i])

    iteration = 0
    while iteration < 10000:
        L = copy.deepcopy(u)
        iteration += 1

        for j in range(1, len(y) - 1):
            u[0][j] = u[1][j] - hx * phi_1(y[j])
            u[-1][j] = u[-2][j] + hx * phi_2(y[j])

        for i in range(1, len(u) - 1):
            for j in range(1, len(u[0]) - 1):
                u[i][j] = (hx ** 2 * free_member(x[i], y[j]) - (
                        L[i + 1][j] + u[i - 1][j]) - D * hx ** 2 * (
                                   L[i][j + 1] + u[i][j - 1]) / (hy ** 2) - A * hx * 0.5 * (
                                   L[i + 1][j] - u[i - 1][j]) - B * hx ** 2 * (
                                   L[i][j + 1] - u[i][j - 1]) / (2 * hy)) / (
                                  C * hx ** 2 - 2 * (hy ** 2 + D * hx ** 2) / (
                                  hy ** 2))
        if get_accurancy(u, L) <= eps:
            break
    return u.transpose(), iteration


def AnalyticSolution():
    return [[analytic_solution(xi, yi) for xi in np.arange(0, lx + hx, hx)] for yi in np.arange(0, ly + hy, hy)], 0


answers = {
    ANALYTIC_SOLUTION_METHOD_NAME: AnalyticSolution(),
    'Simple Iterations': simple_iteration_method(),
    'Zeidel': seidel_method(),
    'Simple Iterations Relaxation': simple_iteration_relaxation_method(),
}


def get_error(numeric):
    analytic_solution, iterations = answers[ANALYTIC_SOLUTION_METHOD_NAME]
    return np.array(
        [np.abs(numeric_el - analytic_el).mean() for numeric_el, analytic_el in zip(numeric, analytic_solution)]).max()


x = np.arange(0, lx + hx, hx)
y = np.arange(0, ly + hy, hy)


def plot_results_y(ax, cur_y):
    cur_y_id = abs(y - cur_y).argmin()

    for method_name, answer in answers.items():
        solution, iterations = answer
        ax.plot(x, [solution[i][cur_y_id] for i in range(len(x))], label=method_name)

    ax.set_xlabel('x')
    ax.set_ylabel('U')
    ax.grid()
    ax.legend(loc='best')


def plot_results_x(ax, cur_x):
    cur_x_id = abs(y - cur_x).argmin()

    for method_name, answer in answers.items():
        solution, iterations = answer
        ax.plot(x, solution[cur_x_id], label=method_name)

    ax.set_xlabel('y')
    ax.set_ylabel('U')
    ax.grid()
    ax.legend(loc='best')


def plot_errors_from_y(ax):
    analytic_solution, iterations = answers[ANALYTIC_SOLUTION_METHOD_NAME]
    analytic_xn = np.array([[analytic_solution[j][i] for j in range(len(x))] for i in range(len(y))])

    for method_name, answer in answers.items():
        if method_name == ANALYTIC_SOLUTION_METHOD_NAME:
            continue

        solution, iterations = answer
        solution_xn = np.array([[solution[j][i] for j in range(len(x))] for i in range(len(y))])
        max_abs_errors = np.array([
            np.abs(solution_i - analytic_i).max()
            for solution_i, analytic_i in zip(solution_xn, analytic_xn)
        ])
        ax.plot(y, max_abs_errors, label=method_name)

    ax.grid()
    ax.legend(loc='best')
    ax.set_xlabel('y')
    ax.set_ylabel('Max abs error')


def plot_errors_from_x(ax):
    analytic_solution, iterations = answers[ANALYTIC_SOLUTION_METHOD_NAME]
    analytic_yn = np.array([analytic_solution[i] for i in range(len(x))])

    for method_name, answer in answers.items():
        if method_name == ANALYTIC_SOLUTION_METHOD_NAME:
            continue

        solution, iterations = answer
        solution_yn = np.array([solution[i] for i in range(len(x))])
        max_abs_errors = np.array([
            np.abs(solution_i - analytic_i).max()
            for solution_i, analytic_i in zip(solution_yn, analytic_yn)
        ])
        ax.plot(x, max_abs_errors, label=method_name)

    ax.grid()
    ax.legend(loc='best')
    ax.set_xlabel('x')
    ax.set_ylabel('Max abs error')


fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

plot_results_y(ax[0][0], cur_y=ly / 2)
plot_results_x(ax[0][1], cur_x=lx / 2)
plot_errors_from_x(ax[1][0])
plot_errors_from_y(ax[1][1])

plt.show()

for method_name, answer in answers.items():
    if method_name == ANALYTIC_SOLUTION_METHOD_NAME:
        continue

    solution, iterations = answer
    print(f'{method_name}:\n \titerations {iterations}\n \terror: {get_error(solution)}')
