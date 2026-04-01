import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import tensorflow as tf

from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import (Dense, Input, Dropout, Conv2D, Flatten, MaxPooling2D,
                          RandomFlip, RandomRotation, RandomZoom, RandomTranslation, RandomContrast)
from sklearn.metrics import confusion_matrix

from src.draw import ImagePlotter

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
EPOCHS = 30
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
SEED = 42

OPTIMIZER = 'adam'
LOSS = 'sparse_categorical_crossentropy'
METRIC = 'accuracy'

INPUT_SHAPE = (32, 32, 3)
CONV_1_FILTERS = 32
CONV_2_FILTERS = 64
CONV_3_FILTERS = 128
CONV_KERNEL_SIZE = (3, 3)
CONV_ACTIVATION = 'relu'
CONV_PADDING = 'same'
POOL_SIZE = (2, 2)
DROPOUT_RATE = 0.5
HIDDEN_UNITS = 256
HIDDEN_ACTIVATION = 'relu'
OUTPUT_UNITS = 10
OUTPUT_ACTIVATION = 'softmax'

CHECKPOINT_MONITOR = 'val_loss'
CHECKPOINT_MODE = 'min'


OPTION = 'eval' # train, eval
MODEL_PATH = os.path.join('models', 'cnn_model.keras')
SHOW_DATASET_STATS = True


if __name__ == '__main__':
    plotter = ImagePlotter(cmap='gray', heatmap_cmap='Reds')
    tf.get_logger().setLevel('ERROR')
    tf.random.set_seed(SEED)

    # Загрузка и отображение датасета
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train, y_test = y_train.flatten(), y_test.flatten()

    if SHOW_DATASET_STATS:
        plotter.imshow_cifer10(x_train, y_train, CLASSES)
        print(f'Train sample length: {len(x_train)}')
        print(f'Test sample length: {len(x_test)}')

        print()
        train_classes = np.unique(y_train, return_counts=True)
        print("Train classes distribution:")
        for class_label, count in zip(train_classes[0], train_classes[1]):
            print(f"Class {class_label} ({CLASSES[class_label]}): {count} ({count / len(y_train) * 100:.2f}%)")

    # Нормировка
    x_train, x_test = x_train / 255.0, x_test / 255.0

    if OPTION == 'train':
        # Перемешивание train-выборки перед разбиением на train/val
        rng = np.random.default_rng(SEED)
        train_indices = rng.permutation(len(x_train))
        x_train = x_train[train_indices]
        y_train = y_train[train_indices]

        # Слой аугментации
        data_augmentation = Sequential([
            RandomFlip("horizontal", name='random_flip'),
            RandomRotation(0.03, name='random_rotation'),
            RandomZoom(0.03, name='random_zoom'),
        ], name='data_augmentation')

        # Создание модели
        cnn_model = Sequential([
            # Входной слой
            Input(shape=INPUT_SHAPE, name='input'),

            # Слой аугментации
            data_augmentation,

            # Первый свёрточный слой
            Conv2D(CONV_1_FILTERS, CONV_KERNEL_SIZE, activation=CONV_ACTIVATION, padding=CONV_PADDING, name='conv-1'),
            MaxPooling2D(POOL_SIZE, name='max-pooling-1'),

            # Второй свёрточный слой
            Conv2D(CONV_2_FILTERS, CONV_KERNEL_SIZE, activation=CONV_ACTIVATION, padding=CONV_PADDING, name='conv-2'),
            MaxPooling2D(POOL_SIZE, name='max-pooling-2'),

            # Третий свёрточный слой
            Conv2D(CONV_3_FILTERS, CONV_KERNEL_SIZE, activation=CONV_ACTIVATION, padding=CONV_PADDING, name='conv-3'),
            MaxPooling2D(POOL_SIZE, name='max-pooling-3'),

            # Слой развертки
            Flatten(name='flatten'),

            # Dropout
            Dropout(DROPOUT_RATE, name='dropout'),

            # Полносвязный слой
            Dense(HIDDEN_UNITS, activation=HIDDEN_ACTIVATION, name='hidden'),

            # Выходной слой
            Dense(OUTPUT_UNITS, activation=OUTPUT_ACTIVATION, name='output')
        ])

        # Компиляция модели
        cnn_model.compile(
            optimizer=OPTIMIZER,
            loss=LOSS,
            metrics=[METRIC]
        )

        # Callback для сохранения лучшей модели на Val
        checkpoint_callback = ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor=CHECKPOINT_MONITOR,
            mode=CHECKPOINT_MODE,
            save_best_only=True,
            save_weights_only=False
        )

        # Обучение модели
        print()
        print('Train model...')
        val_size = int(VALIDATION_SPLIT * len(x_train))
        x_val, y_val = x_train[:val_size], y_train[:val_size]
        x_train_fit, y_train_fit = x_train[val_size:], y_train[val_size:]

        history = cnn_model.fit(
            x_train_fit, y_train_fit,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(x_val, y_val),
            callbacks=[checkpoint_callback],
            shuffle=True
        )

        # Зависимость функции потерь и критерия качества от номера эпохи для Train и Val
        metric_names = list(history.history.keys())[:4]
        metric_values = [history.history[name] for name in metric_names]
        plotter.plot_four(
            'Metrics and Loss',
            metric_values[0], metric_names[0],
            metric_values[1], metric_names[1],
            metric_values[2], metric_names[2],
            metric_values[3], metric_names[3],
            xlabel='Epoch',
            ylabel='Value'
        )

    elif OPTION == 'eval':
        # Перемешивание test-выборки
        rng = np.random.default_rng(SEED)
        test_indices = rng.permutation(len(x_test))
        x_test = x_test[test_indices]
        y_test = y_test[test_indices]

        # Загрузка модели
        print()
        print('Load model...')
        fnn_model = load_model(MODEL_PATH)

        # Качество работы модели на тестовой выборке
        print()
        print('Test model...')
        loss_and_metrics = fnn_model.evaluate(x_test, y_test)
        test_loss = loss_and_metrics[0]
        test_accuracy = loss_and_metrics[1]
        print(f'Test loss: {test_loss:.3f}')
        print(f'Test {METRIC}: {test_accuracy:.3f}')

        # Confusion Matrix
        print()
        print('Predict...')
        y_pred_proba = fnn_model.predict(x_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        cm = confusion_matrix(y_test, y_pred)
        plotter.plot_confusion_matrix(cm, CLASSES)

        # Отображение ошибок по классам
        print()
        print('Accuracy per classes:')
        for class_label in range(10):
            true_class_mask = (y_test == class_label)
            total_true = np.sum(true_class_mask)

            correct_predictions = np.sum((y_pred == class_label) & true_class_mask)

            accuracy = correct_predictions / total_true if total_true > 0 else 0
            print(f'Test {METRIC} ({CLASSES[class_label]}): {accuracy:.3f} ({correct_predictions}/{total_true})')

        # Пример неверно проклассифицированных объектов по каждому классу
        incorrect_mask = (y_pred != y_test)
        incorrect_indices = np.where(incorrect_mask)[0]

        examples_per_class = 4
        selected_indices = []

        for class_label in range(10):
            class_incorrect_indices = incorrect_indices[y_pred[incorrect_indices] == class_label]
            num_examples = min(examples_per_class, len(class_incorrect_indices))
            selected_indices.extend(class_incorrect_indices[:num_examples])

        incorrect_images = (x_test[selected_indices] * 255).astype(np.uint8)
        incorrect_labels = y_pred[selected_indices]

        plotter.imshow_cifer10(
            incorrect_images,
            incorrect_labels,
            CLASSES,
            title="False predictions per class"
        )
