import numpy as np
import copy

EPS = 0.000000000000000001


# находит позицию максимального по модулю элемента в верхнем треугольнике матрицы
def find_max_upper_element(X):
    n = X.shape[0]
    i_max, j_max = 0, 1
    max_elem = abs(X[0][1])

    for i in range(n):
        for j in range(i + 1, n):
            if abs(X[i][j]) > max_elem:
                max_elem = abs(X[i][j])
                i_max = i
                j_max = j

    return i_max, j_max


# норма матрицы
def matrix_norm(X):
    norm = 0

    for i in range(len(X[0])):
        for j in range(i + 1, len(X[0])):
            norm += X[i][j] * X[i][j]

    return np.sqrt(norm)


# вычисляет СЗ и СВ с помощью метода вращений
def rotation_method(A):
    n = A.shape[0]
    A_i = np.copy(A)
    eigen_vectors = np.eye(n)
    iterations = 0

    while matrix_norm(A_i) > EPS:
        i_max, j_max = find_max_upper_element(A_i)
        if A_i[i_max][i_max] - A_i[j_max][j_max] == 0:
            phi = np.pi / 4
        else:
            phi = 0.5 * np.arctan(2 * A_i[i_max][j_max] / (A_i[i_max][i_max] - A_i[j_max][j_max]))

        # матрица ротаций
        U = np.eye(n)
        U[i_max][j_max] = -np.sin(phi)
        U[j_max][i_max] = np.sin(phi)
        U[i_max][i_max] = np.cos(phi)
        U[j_max][j_max] = np.cos(phi)

        A_i = U.T @ A_i @ U
        eigen_vectors = eigen_vectors @ U
        iterations += 1

    eigen_values = np.array([A_i[i][i] for i in range(n)])
    return eigen_values, eigen_vectors, iterations


if __name__ == '__main__':
    A = [
        [-7, -9, 1],
        [-9, 7, 2],
        [1, 2, 9]
    ]
    A = np.array(A, dtype='float')
    # X = copy.deepcopy(A)

    values, vector, iters = rotation_method(A)
    print('Собственные значения:', values)
    print('Собственный вектор:')
    print(vector)

    print('Итерации:', iters)
    n = len(A)
    Q = np.array([vector[i][0] for i in range(n)])
    print(Q)
    print(A @ Q)
    print(Q * values[0])