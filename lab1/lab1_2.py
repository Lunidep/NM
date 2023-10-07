def tridiagonal_solve(A, b):
    n = len(A)
    # Forward
    v = [0 for _ in range(n)]
    u = [0 for _ in range(n)]
    v[0] = A[0][1] / -A[0][0]
    u[0] = b[0] / A[0][0]
    for i in range(1, n-1):
        v[i] = A[i][i+1] / (-A[i][i] - A[i][i-1] * v[i-1])
        u[i] = (A[i][i-1] * u[i-1] - b[i]) / (-A[i][i] - A[i][i-1] * v[i-1])
    v[n-1] = 0
    u[n-1] = (A[n-1][n-2] * u[n-2] - b[n-1]) / (-A[n-1][n-1] - A[n-1][n-2] * v[n-2])

    # Backward
    x = [0 for _ in range(n)]
    x[n-1] = u[n-1]
    for i in range(n-1, 0, -1):
        x[i-1] = v[i-1] * x[i] + u[i-1]
    return x


if __name__ == "__main__":
    A = [
        [-6, 5, 0, 0, 0],
        [-1, 13, 6, 0, 0],
        [0, -9, -15, -4, 0],
        [0, 0, -1, -7, 1],
        [0, 0, 0, 9, -18]
    ]

    b = [51, 100, -12, 47, -90]

    x = tridiagonal_solve(A, b)
    print(x)
