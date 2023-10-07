def df(x_test, x, y):
    assert len(x) == len(y)
    visited = False
    for interval in range(len(x) - 1):
        if x[interval] <= x_test < x[interval+1]:
            i = interval
            visited = True
            break

    if not visited or (visited and i > len(x) - 3):
        i = len(x) - 3
    a1 = (y[i+1] - y[i]) / (x[i+1] - x[i])
    a2 = ((y[i+2] - y[i+1]) / (x[i+2] - x[i+1]) - a1) / (x[i+2] - x[i]) * (2*x_test - x[i] - x[i+1])
    return a1 + a2


def d2f(x_test, x, y):
    assert len(x) == len(y)
    for interval in range(len(x) - 1):
        if x[interval] <= x_test < x[interval+1]:
            i = interval
            break

    num = (y[i+2] - y[i+1]) / (x[i+2] - x[i+1]) - (y[i+1] - y[i]) / (x[i+1] - x[i])
    return 2 * num / (x[i+2] - x[i])


if __name__ == '__main__':
    x = [-0.2, 0.0, 0.2, 0.4, 0.6]
    y = [-0.40136, 0.0, 0.40136, 0.81152, 1.2435]
    x_test = 0.2

    print('Первая производная')
    print(f'df({x_test}) = {df(x_test, x, y)}')

    print('Вторая производная')
    print(f'd2f({x_test}) = {d2f(x_test, x, y)}')
