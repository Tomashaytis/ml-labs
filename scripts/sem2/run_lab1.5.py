import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.datasets import mnist
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout

EPOCHS = 15
VALIDATION_SPLIT = 0.2
OPTIMIZER = 'adam'
LOSS = 'sparse_categorical_crossentropy'
METRIC = 'accuracy'
INPUT_UNITS = 28 * 28
OUTPUT_UNITS = 10

# Формулы для отчета (включая bias, как считает Keras count_params)
def params_one_hidden(h1: int) -> int:
    return INPUT_UNITS * h1 + h1 + h1 * OUTPUT_UNITS + OUTPUT_UNITS


def params_two_hidden(h1: int, h2: int) -> int:
    return (
        INPUT_UNITS * h1 + h1 +
        h1 * h2 + h2 +
        h2 * OUTPUT_UNITS + OUTPUT_UNITS
    )


def build_model(hidden_layers: list[int], dropout_rate: float = 0.0) -> Sequential:
    layers = [Input(shape=(INPUT_UNITS,), name='input')]

    for i, units in enumerate(hidden_layers, start=1):
        layers.append(Dense(units, activation='relu', name=f'hidden_{i}'))
        if dropout_rate > 0:
            layers.append(Dropout(dropout_rate, name=f'dropout_{i}'))

    layers.append(Dense(OUTPUT_UNITS, activation='softmax', name='output'))

    model = Sequential(layers)
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[METRIC])
    return model


def train_config(name: str, hidden_layers: list[int], batch_size: int, dropout_rate: float,
                 x_train: np.ndarray, y_train: np.ndarray) -> dict:
    model = build_model(hidden_layers, dropout_rate)
    print('=' * 80)
    print(f'{name}')
    print(f'hidden_layers={hidden_layers}, batch_size={batch_size}, dropout={dropout_rate}')
    print(f'params (keras) = {model.count_params()}')

    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=batch_size,
        validation_split=VALIDATION_SPLIT,
        verbose=1,
    )

    best_epoch = int(np.argmax(history.history['val_accuracy'])) + 1
    best_val_acc = float(np.max(history.history['val_accuracy']))
    best_val_loss = float(np.min(history.history['val_loss']))

    print(f'best epoch: {best_epoch}')
    print(f'best val_accuracy: {best_val_acc:.4f}')
    print(f'best val_loss: {best_val_loss:.4f}')

    return {
        'name': name,
        'hidden_layers': hidden_layers,
        'batch_size': batch_size,
        'dropout_rate': dropout_rate,
        'params': int(model.count_params()),
        'history': history.history,
        'best_epoch': best_epoch,
        'best_val_accuracy': best_val_acc,
        'best_val_loss': best_val_loss,
    }


def plot_metric(results: list[dict], metric_name: str, title: str):
    if not results:
        return

    plt.figure(figsize=(10, 6))
    for result in results:
        plt.plot(result['history'][metric_name], label=result['name'])
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    tf.get_logger().setLevel('ERROR')

    # 1 скрытый слой: 795*h1 + 10
    h1 = 100
    print('Формула для 1 скрытого слоя (с bias):')
    print('N1(h1) = 784*h1 + h1 + 10*h1 + 10 = 795*h1 + 10')
    print(f'N1({h1}) = {params_one_hidden(h1)}')
    print()

    # 2 скрытых слоя: 784*h1 + h1 + h1*h2 + h2 + 10*h2 + 10
    h1_2, h2_2 = 96, 32
    print('Формула для 2 скрытых слоев (с bias):')
    print('N2(h1, h2) = 784*h1 + h1 + h1*h2 + h2 + 10*h2 + 10')
    print(f'N2({h1_2}, {h2_2}) = {params_two_hidden(h1_2, h2_2)}')
    print()
    print('Обе конфигурации попадают в диапазон 70 000 - 100 000 параметров.')
    print()

    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_train = x_train.reshape(-1, INPUT_UNITS)

    configs_dropout = [
        {
            'name': '1 hidden, dropout=0.0, bs=32',
            'hidden_layers': [100],
            'batch_size': 32,
            'dropout_rate': 0.0,
        },
        {
            'name': '1 hidden, dropout=0.1, bs=32',
            'hidden_layers': [100],
            'batch_size': 32,
            'dropout_rate': 0.1,
        },
        {
            'name': '1 hidden, dropout=0.2, bs=32',
            'hidden_layers': [100],
            'batch_size': 32,
            'dropout_rate': 0.2,
        },
        {
            'name': '1 hidden, dropout=0.3, bs=32',
            'hidden_layers': [100],
            'batch_size': 32,
            'dropout_rate': 0.3,
        },
    ]

    configs_bs = [
        {
            'name': '1 hidden, dropout=0.2, bs=8',
            'hidden_layers': [100],
            'batch_size': 8,
            'dropout_rate': 0.2,
        },
        {
            'name': '1 hidden, dropout=0.2, bs=16',
            'hidden_layers': [100],
            'batch_size': 16,
            'dropout_rate': 0.2,
        },
        {
            'name': '1 hidden, dropout=0.2, bs=32',
            'hidden_layers': [100],
            'batch_size': 32,
            'dropout_rate': 0.2,
        },
        {
            'name': '1 hidden, dropout=0.2, bs=64',
            'hidden_layers': [100],
            'batch_size': 64,
            'dropout_rate': 0.2,
        },
        {
            'name': '1 hidden, dropout=0.2, bs=128',
            'hidden_layers': [100],
            'batch_size': 128,
            'dropout_rate': 0.2,
        },
    ]

    configs_layer = [
        {
            'name': '1 hidden, dropout=0.2, bs=32',
            'hidden_layers': [100],
            'batch_size': 32,
            'dropout_rate': 0.2,
        },
        {
            'name': '2 hidden, dropout=0.2, bs=32',
            'hidden_layers': [96, 32],
            'batch_size': 32,
            'dropout_rate': 0.2,
        },
    ]

    results = []
    for cfg in configs_layer:
        results.append(train_config(
            cfg['name'],
            cfg['hidden_layers'],
            cfg['batch_size'],
            cfg['dropout_rate'],
            x_train,
            y_train,
        ))

    print('\n' + '=' * 80)
    print('ИТОГОВАЯ ТАБЛИЦА')
    print('=' * 80)
    for r in results:
        print(
            f"{r['name']:<32} | params={r['params']:<6} | "
            f"best_epoch={r['best_epoch']:<2} | "
            f"best_val_acc={r['best_val_accuracy']:.4f} | "
            f"best_val_loss={r['best_val_loss']:.4f}"
        )

    plot_metric(results, 'val_loss', 'Task 5: validation loss')
    plot_metric(results, 'val_accuracy', 'Task 5: validation accuracy')


if __name__ == '__main__':
    main()
