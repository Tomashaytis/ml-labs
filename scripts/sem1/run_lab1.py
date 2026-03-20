import numpy as np

from src.generate import generate_2_samples, generate_3_samples
from src.utils.sem1.lab1 import calculate_covariance2, stat_estimates

'''
Генерация данных - вариант 3
'''

if __name__ == '__main__':
    _ = generate_2_samples(should_visualize=True, should_print=True)
    sample1, sample2, sample3 = generate_3_samples(should_visualize=True, should_print=True)

    sample_mean, sample_covariance = stat_estimates(sample1.data)
    print(f'Выборка 1: ')
    print(f'Мат. ожидание: {sample_mean}')
    print(f'Ковариационная матрица:')
    print(sample_covariance)
    print()

    print(f'Выборка 2: ')
    print(f'Мат. ожидание: {np.average(sample2.data, 0)}')
    print(f'Ковариационная матрица:')
    print(calculate_covariance2(sample2.data))
    print()

    print(f'Выборка 2: ')
    print(f'Мат. ожидание: {np.average(sample3.data, 0)}')
    print(f'Ковариационная матрица:')
    print(calculate_covariance2(sample3.data))
    print()
