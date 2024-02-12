def f(x):
    return 1 / (256 - (x ** 4))


def integrate_rectangle_method(f, l, r, h):
    result = 0
    cur_x = l
    while cur_x < r:
        result += h * f((cur_x + cur_x + h) * 0.5)
        cur_x += h
    return result


def integrate_trapeze_method(f, l, r, h):
    result = 0
    cur_x = l
    while cur_x < r:
        result += h * 0.5 * (f(cur_x + h) + f(cur_x))
        cur_x += h
    return result


def integrate_simpson_method(f, l, r, h):
    result = 0
    cur_x = l + h
    while cur_x < r:
        result += f(cur_x - h) + 4*f(cur_x) + f(cur_x + h)
        cur_x += 2 * h
    return result * h / 3


def runge_romberg_method(h1, h2, integral1, integral2, p):
    return integral1 + (integral1 - integral2) / ((h2 / h1)**p - 1)


if __name__ == '__main__':
    l, r = -2, 2
    h1, h2 = 1.0, 0.5

    print('Метод пряоугольников')
    int_rectangle_h1 = integrate_rectangle_method(f, l, r, h1)
    int_rectangle_h2 = integrate_rectangle_method(f, l, r, h2)
    print(f'Шаг = {h1}: занчение интеграла = {int_rectangle_h1}')
    print(f'Шаг = {h2}: занчение интеграла = {int_rectangle_h2}')

    print('Метод трапеций')
    int_trapeze_h1 = integrate_trapeze_method(f, l, r, h1)
    int_trapeze_h2 = integrate_trapeze_method(f, l, r, h2)
    print(f'Шаг = {h1}: занчение интеграла = {int_trapeze_h1}')
    print(f'Шаг = {h2}: занчение интеграла = {int_trapeze_h2}')

    print('Метод Симпсона')
    int_simpson_h1 = integrate_simpson_method(f, l, r, h1)
    int_simpson_h2 = integrate_simpson_method(f, l, r, h2)
    print(f'Шаг = {h1}: занчение интеграла = {int_simpson_h1}')
    print(f'Шаг = {h2}: занчение интеграла = {int_simpson_h2}')

    print('Метод Рунге-Ромберга')
    print(f'Погрешность прямоугольников = {runge_romberg_method(h1, h2, int_rectangle_h1, int_rectangle_h2, 3)}')//уточненное значение
    print(f'Погрешность трапеций = {runge_romberg_method(h1, h2, int_trapeze_h1, int_trapeze_h2, 3)}')
    print(f'Погрешность Симпсона = {runge_romberg_method(h1, h2, int_simpson_h1, int_simpson_h2, 3)}')
