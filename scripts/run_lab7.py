import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

from src.draw import ImagePlotter

N_EPOCHS = 10
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

OPTIMIZER = 'adam'
LOSS = 'sparse_categorical_crossentropy'
METRICS = [
    'accuracy'
]

INPUT_UNITS = 28 * 28
HIDDEN_ACTIVATION = 'relu'
HIDDEN_UNITS = 2 * INPUT_UNITS + 1
OUTPUT_ACTIVATION = 'softmax'
OUTPUT_UNITS = 10


if __name__ == '__main__':
    plotter = ImagePlotter(cmap='gray')

    # Загрузка и отображение датасета
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    plotter.imshow_mnist(x_train, y_train)
    print(f'Train sample length: {len(x_train)}')
    print(f'Test sample length: {len(x_test)}')

    print()
    train_classes = np.unique(y_train, return_counts=True)
    print("Train classes distribution:")
    for digit, count in zip(train_classes[0], train_classes[1]):
        print(f"Digit {digit}: {count} ({count / len(y_train) * 100:.2f}%)")

    # Разворачивание в строку
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    # Создание модели с одним скрытым слоем
    nn_model = Sequential([
        # Входной слой
        Input(shape=(INPUT_UNITS,), name='input_layer'),

        # Скрытый слой
        Dense(HIDDEN_UNITS, activation=HIDDEN_ACTIVATION, name='hidden'),

        # Выходной слой
        Dense(OUTPUT_UNITS, activation=OUTPUT_ACTIVATION, name='output')
    ])

    # Компиляция модели
    nn_model.compile(
        optimizer=OPTIMIZER,
        loss=LOSS,
        metrics=METRICS
    )

    # Обучение модели
    history = nn_model.fit(
        x_train, y_train,
        epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT
    )
    print(history.history.keys())





