import matplotlib.pyplot as plt

from lab1.lab1_1 import LU_decompose, solve_system


def least_squares(x, y, n):
    assert len(x) == len(y)
    A = []
    b = []
    for k in range(n + 1):
        A.append([sum(map(lambda x: x ** (i + k), x)) for i in range(n + 1)])
        b.append(sum(map(lambda x: x[0] * x[1] ** k, zip(y, x))))
    L, U = LU_decompose(A)
    return solve_system(L, U, b)


def P(coefs, x):
    return sum([c * x**i for i, c in enumerate(coefs)])


def sum_squared_errors(x, y, ls_coefs):
    y_ls = [P(ls_coefs, x_i) for x_i in x]
    return sum((y_i - y_ls_i)**2 for y_i, y_ls_i in zip(y, y_ls))


if __name__ == '__main__':
    x = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]
    y = [-2.9502, -1.8647, -0.63212, 1.0, 3.7183, 9.3891]
    plt.scatter(x, y, color='r')
    plt.plot(x, y, color='c', label='original')

    print('МНК степень 1')
    ls1 = least_squares(x, y, 1)
    print(f'P(x) = {ls1[0]} + {ls1[1]}x')
    plt.plot(x, [P(ls1, x_i) for x_i in x], color='b', label='степень = 1')
    print(f'сумма квадратов ошибок = {sum_squared_errors(x, y, ls1)}')

    print('МНК степень = 2')
    ls2 = least_squares(x, y, 2)
    print(f'P(x) = {ls2[0]} + {ls2[1]}x + {ls2[2]}x^2')
    plt.plot(x, [P(ls2, x_i) for x_i in x], color='g', label='степень = 2')
    print(f'сумма квадратов ошибок = {sum_squared_errors(x, y, ls2)}')

    plt.legend()
    plt.show()
