import math

def f(x):
    return (math.e ** x) + x

def lagrange_interpolation(x, y, test_point):
    assert len(x) == len(y)
    polynom_str = 'L(x) ='
    polynom_test_value = 0
    for i in range(len(x)): # сумма
        cur_enum_str = ''
        cur_enum_test = 1
        cur_denom = 1
        for j in range(len(x)): # произведение
            if i == j:
                continue
            cur_enum_str += f'(x-{x[j]:.2f})' # строка многочлена для текущего х
            cur_enum_test *= (test_point[0] - x[j])
            cur_denom *= (x[i] - x[j])

        polynom_str += f' + {(y[i] / cur_denom):.2f}*' + cur_enum_str
        polynom_test_value += y[i] * cur_enum_test / cur_denom
    return polynom_str, abs(polynom_test_value - test_point[1])

def newton_interpolation(x, y, test_point):
    assert len(x) == len(y)

    # Вычисляем разделенные разности
    n = len(x)
    coefs = [y[i] for i in range(n)]
    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            coefs[j] = float(coefs[j] - coefs[j - 1]) / float(x[j] - x[j - i])

    # Вычисление многочлена
    polynom_str = 'P(x) = '
    polynom_test_value = 0  # P(x*)

    cur_multipliers_str = ''
    cur_multipliers = 1
    for i in range(n):
        polynom_test_value += cur_multipliers * coefs[i]
        if i == 0:
            polynom_str += f'{coefs[i]:.2f}'
        else:
            polynom_str += ' + ' + cur_multipliers_str + '*' + f'{coefs[i]:.2f}'

        cur_multipliers *= (test_point[0] - x[i])
        cur_multipliers_str += f'(x-{x[i]:.2f})'
    return polynom_str, abs(polynom_test_value - test_point[1])

if __name__ == '__main__':
    x_a = [-2, -1, 0, 1]
    x_b = [-2, -1, 0.2, 1]
    y_a = [f(x) for x in x_a]
    y_b = [f(x) for x in x_b]

    x_test = -0.5
    y_test = f(x_test)

    print('Интерполяционный многочлен Лагранжа')
    print('Точки A')
    lagrange_polynom_a, lagrange_error_a = lagrange_interpolation(x_a, y_a, (x_test, y_test))
    print('Полином')
    print(lagrange_polynom_a)
    print('Значение погрешности интерполяции =', lagrange_error_a)

    print('Точки B')
    lagrange_polynom_b, lagrange_error_b = lagrange_interpolation(x_b, y_b, (x_test, y_test))
    print('Полином')
    print(lagrange_polynom_b)
    print('Значение погрешности интерполяции =', lagrange_error_b)
    print()

    print('Интерполяционный многочлен Ньютона')
    print('Точки A')
    newton_polynom_a, newton_error_a = newton_interpolation(x_a, y_a, (x_test, y_test))
    print('Полином')
    print(newton_polynom_a)
    print('Значение погрешности интерполяции =', newton_error_a)

    print('Точки B')
    newton_polynom_b, newton_error_b = newton_interpolation(x_b, y_b, (x_test, y_test))
    print('Полином')
    print(newton_polynom_b)
    print('Значение погрешности интерполяции =', newton_error_b)
