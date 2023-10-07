import numpy as np


def L1_norm(X):
    n = X.shape[0]
    if type(X[0]) == np.ndarray:
        l2_norm = abs(X[0][0])
        for i in range(n):
            for j in range(n):
                l2_norm = max(abs(X[i][j]), l2_norm)
    else:
        l2_norm = abs(X[0])
        for i in range(n):
            l2_norm = max(abs(X[i]), l2_norm)
    return l2_norm


def solve_iterative(A, b, eps):
    n = A.shape[0]
    alpha = np.zeros_like(A, dtype='float')
    beta = np.zeros_like(b, dtype='float')

    for i in range(n):
        for j in range(n):
            if i == j:
                alpha[i][j] = 0
            else:
                alpha[i][j] = -A[i][j] / A[i][i]

        beta[i] = b[i] / A[i][i]

    iterations = 0
    cur_x = np.copy(beta)
    converge = False
    while not converge:
        prev_x = np.copy(cur_x)

        # Ax=b -> x = alpha * x + beta
        cur_x = alpha @ prev_x + beta
        iterations += 1
        # условие сходимости
        if L1_norm(alpha) < 1:
            # используем коррекцию для оценки сходимости
            converge = L1_norm(alpha) / (1 - L1_norm(alpha)) * L1_norm(cur_x - prev_x) <= eps
        else:
            # используем условие сходимости
            converge = L1_norm(cur_x - prev_x) <= eps
    return cur_x, iterations

# шаг итерации метода Зейделя
def seidel_multiplication(alpha, x, beta):
    res = np.copy(x)
    c = np.copy(alpha)
    for i in range(alpha.shape[0]):
        res[i] = beta[i]
        for j in range(alpha.shape[1]):
            res[i] += alpha[i][j] * res[j]
            if j < i:
                c[i][j] = 0
    return res, c


def solve_seidel(A, b, eps):
    n = A.shape[0]

    alpha = np.zeros_like(A, dtype='float')
    beta = np.zeros_like(b, dtype='float')
    for i in range(n):
        for j in range(n):
            if i == j:
                alpha[i][j] = 0
            else:
                alpha[i][j] = -A[i][j] / A[i][i]

        beta[i] = b[i] / A[i][i]

    iterations = 0
    cur_x = np.copy(beta)
    converge = False
    while not converge:
        prev_x = np.copy(cur_x)

        # Ax = b -> x = alpha * x + beta
        cur_x, c = seidel_multiplication(alpha, prev_x, beta)
        iterations += 1
        if L1_norm(alpha) < 1:
            converge = L1_norm(c) / (1 - L1_norm(alpha)) * L1_norm(cur_x - prev_x) <= eps
        else:
            converge = L1_norm(prev_x - cur_x) <= eps
    return cur_x, iterations


if __name__ == '__main__':
    A = [
        [-19, 2, -1, -8],
        [2, 14, 0, -4],
        [6, -5, -20, -6],
        [-6, 4, -2, 15]
    ]
    A = np.array(A, dtype='float')
    b = [38, 20, 52, 43]
    eps = 0.000000001

    print('Метод простых итераций')
    x_iter, i_iter = solve_iterative(A, b, eps)
    print(x_iter)
    print('Кол-во итераций:', i_iter)
    print()

    print('метод Зейделя')
    x_seidel, i_seidel = solve_seidel(A, b, eps)
    print(x_seidel)
    print('Кол-во итераций:', i_seidel)
