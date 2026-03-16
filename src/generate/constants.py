import numpy as np

# Цвета
COLORS = {
    (0, 1): 'red',
    (1, 2): 'blue',
    (2, 0): 'green'
}

# Общие константы
SIZE = 200

# Математические ожидания
MEAN0 = np.array([0, 0])
MEAN1 = np.array([-1, 1])
MEAN2 = np.array([-1, -1])
MEAN3 = np.array([1, 1])
MEAN4 = np.array([1, -1])

# Ковариационные матрицы
COV0 = np.array([
    [0.15, 0.00],
    [0.00, 0.15],
])
COV1 = np.array([
    [ 0.15, -0.07],
    [-0.07,  0.15],
])
COV2 = np.array([
    [ 0.2, -0.1],
    [-0.1,  0.2],
])
COV3 = np.array([
    [0.05, 0.00],
    [0.00, 0.05],
])
'''
COV1 = np.array([
    [0.15, 0.00],
    [0.00, 0.15],
])
COV2 = np.array([
    [ 0.15, -0.10],
    [-0.10,  0.15],
])
COV3 = np.array([
    [ 0.2, -0.1],
    [-0.1,  0.2],
])
'''