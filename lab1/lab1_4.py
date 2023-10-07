import numpy as np


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

# L2 норма для элементов выше главной диагонали
def matrix_norm(X):
    n = len(X[0])
    norm = 0
    for i in range(n):
        for j in range(i + 1, n):
            norm += X[i][j] * X[i][j]
    return np.sqrt(norm)


def rotation_method(A, eps):
    n = A.shape[0]
    A_i = np.copy(A)
    eigen_vectors = np.eye(n)
    iterations = 0

    while matrix_norm(A_i) > eps:
        i_max, j_max = find_max_upper_element(A_i)
        if A_i[i_max][i_max] - A_i[j_max][j_max] == 0:
            phi = np.pi / 4
        else:
            phi = 0.5 * np.arctan(2 * A_i[i_max][j_max] / (A_i[i_max][i_max] - A_i[j_max][j_max]))

        # матрица вращения
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
    A = [[5, -3, -4],
         [-3. -3, 4],
         [-4, 4, 0]]
    A = np.array(A, dtype='float')
    eps = 0.000000000000000001

    eig_values, eig_vectors, iters = rotation_method(A, eps)
    print('Eigen values:', eig_values)
    print('Eigen vectors')
    print(eig_vectors)
    print('Iterations:', iters)
