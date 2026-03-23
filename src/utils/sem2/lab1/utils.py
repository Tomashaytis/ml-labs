import time
import numpy as np

from keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout


def create_nn_model(hidden_units: int) -> Sequential:
    nn_model = Sequential([
        # Входной слой
        Input(shape=(28 * 28,), name='input'),

        # Скрытый слой
        Dense(hidden_units, activation='softmax', name='hidden'),
        Dropout(0.2, name='hidden_dropout'),

        # Выходной слой
        Dense(10, activation='softmax', name='output')
    ])

    # Компиляция модели
    nn_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return nn_model


def grid_search_nn(x_train: np.ndarray, y_train: np.ndarray, hidden_neurons: list, batch_sizes: list, params: dict):
    """Поиск оптимальных параметров по сетке для сети прямого распространения с одним скрытым слоем"""
    results = []
    best_metric = 0
    best_params = {}

    print("=" * 70)
    print("GRID SEARCH")
    print("=" * 70)

    total_combinations = len(hidden_neurons) * len(batch_sizes)
    current = 0

    for neurons in hidden_neurons:
        for batch_size in batch_sizes:
            current += 1
            print(f"\n[{current}/{total_combinations}] Test: units={neurons}, batch_size={batch_size}")

            # Создание модели
            model = Sequential([
                Input(shape=(params['input_units'],), name='input'),
                Dense(neurons, activation=params['hidden_activation'], name='hidden'),
                Dense(params['output_units'], activation=params['output_activation'], name='output')
            ])

            # Компиляция модели
            model.compile(
                optimizer=params['optimizer'],
                loss=params['loss'],
                metrics=[params['metric']]
            )

            # Обучение модели
            start_time = time.time()
            history = model.fit(
                x_train, y_train,
                epochs=params['epochs'],
                batch_size=batch_size,
                validation_split=params['validation_split'],
                verbose=0
            )
            train_time = time.time() - start_time

            # Результаты обучения
            val_metric = max(history.history[f"val_{params['metric']}"])
            train_metric = max(history.history[params['metric']])

            results.append({
                'neurons': neurons,
                'batch_size': batch_size,
                'train_acc': train_metric,
                'val_acc': val_metric,
                'time': train_time
            })

            print(f"  Train acc: {train_metric:.4f}, Val acc: {val_metric:.4f}, Time: {train_time:.2f}s")

            # Запоминаем лучший результат
            if val_metric > best_metric:
                best_metric = val_metric
                best_params = {'neurons': neurons, 'batch_size': batch_size}

    return results, best_params, best_metric