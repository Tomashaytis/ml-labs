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
                          RandomFlip, RandomRotation, RandomZoom)
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

OPTION = 'train'  # train, eval, transfer
MODEL_PATH = os.path.join('models', 'cnn_model.keras')
MODEL_8_CLASSES_PATH = os.path.join('models', 'cnn_model_8_classes.keras')
TRANSFER_MODEL_PATH = os.path.join('models', 'cnn_model_transfer.keras')
SHOW_DATASET_STATS = True

# Классы, исключаемые при первом этапе transfer learning (п.3)
EXCLUDED_CLASSES = [0, 1]


def build_cnn_model(output_units: int = OUTPUT_UNITS, with_augmentation: bool = True):
    # Слой аугментации
    data_augmentation = Sequential([
        RandomFlip("horizontal", name='random_flip'),
        RandomRotation(0.03, name='random_rotation'),
        RandomZoom(0.03, name='random_zoom'),
    ], name='data_augmentation')

    # Входной слой
    layers = [Input(shape=INPUT_SHAPE, name='input')]

    # Слой аугментации
    if with_augmentation:
        layers.append(data_augmentation)

    layers.extend([
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
        Dense(output_units, activation=OUTPUT_ACTIVATION, name='output')
    ])

    return Sequential(layers)


def build_checkpoint_callback(model_path: str):
    return ModelCheckpoint(
        filepath=model_path,
        monitor=CHECKPOINT_MONITOR,
        mode=CHECKPOINT_MODE,
        save_best_only=True,
        save_weights_only=False
    )


def split_train_val(x_data: np.ndarray, y_data: np.ndarray, validation_split: float):
    val_size = int(validation_split * len(x_data))
    x_val, y_val = x_data[:val_size], y_data[:val_size]
    x_train_fit, y_train_fit = x_data[val_size:], y_data[val_size:]
    return x_train_fit, y_train_fit, x_val, y_val


def class_accuracy(model, x_data: np.ndarray, y_true: np.ndarray, class_label: int):
    y_pred = np.argmax(model.predict(x_data, verbose=0), axis=1)
    class_mask = (y_true == class_label)
    total = np.sum(class_mask)
    if total == 0:
        return 0.0
    correct = np.sum((y_pred == class_label) & class_mask)
    return correct / total


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

        # Создание модели
        cnn_model = build_cnn_model()

        # Компиляция модели
        cnn_model.compile(
            optimizer=OPTIMIZER,
            loss=LOSS,
            metrics=[METRIC]
        )

        # Callback для сохранения лучшей модели на Val
        checkpoint_callback = build_checkpoint_callback(MODEL_PATH)

        # Обучение модели
        print()
        print('Train model...')
        x_train_fit, y_train_fit, x_val, y_val = split_train_val(x_train, y_train, VALIDATION_SPLIT)

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
        plotter.plot_train(
            'Metrics and Loss',
            metric_values[0],
            metric_values[2], 'Loss',
            metric_values[1],
            metric_values[3], 'Accuracy',
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

    elif OPTION == 'transfer':
        print()
        print('Transfer learning mode...')
        print(f'Excluded classes on stage 1: {EXCLUDED_CLASSES} -> {[CLASSES[c] for c in EXCLUDED_CLASSES]}')

        # Этап 1: обучение на 8 классах
        train_mask_8 = ~np.isin(y_train, EXCLUDED_CLASSES)
        x_train_8 = x_train[train_mask_8]
        y_train_8_orig = y_train[train_mask_8]

        remaining_classes = [c for c in range(OUTPUT_UNITS) if c not in EXCLUDED_CLASSES]
        class_to_new = {old_class: new_class for new_class, old_class in enumerate(remaining_classes)}
        y_train_8 = np.array([class_to_new[label] for label in y_train_8_orig], dtype=np.int32)

        rng = np.random.default_rng(SEED)
        train_indices_8 = rng.permutation(len(x_train_8))
        x_train_8 = x_train_8[train_indices_8]
        y_train_8 = y_train_8[train_indices_8]

        model_8 = build_cnn_model(output_units=len(remaining_classes), with_augmentation=True)
        model_8.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[METRIC])

        checkpoint_8 = build_checkpoint_callback(MODEL_8_CLASSES_PATH)

        x_train_8_fit, y_train_8_fit, x_val_8, y_val_8 = split_train_val(x_train_8, y_train_8, VALIDATION_SPLIT)

        print()
        print('Stage 1/2: train base model on 8 classes...')
        model_8.fit(
            x_train_8_fit, y_train_8_fit,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(x_val_8, y_val_8),
            callbacks=[checkpoint_8],
            shuffle=True
        )

        # Загружаем лучшую модель этапа 1
        model_8 = load_model(MODEL_8_CLASSES_PATH)

        # Удаляем 2 последних полносвязных слоя и замораживаем базу
        model_8.pop()
        model_8.pop()
        model_8.trainable = False

        # Этап 2: новая голова на 10 классов
        transfer_model = Sequential([
            model_8,
            Dense(HIDDEN_UNITS, activation=HIDDEN_ACTIVATION, name='transfer_hidden'),
            Dense(OUTPUT_UNITS, activation=OUTPUT_ACTIVATION, name='transfer_output')
        ], name='transfer_cnn')

        transfer_model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[METRIC])
        checkpoint_transfer = build_checkpoint_callback(TRANSFER_MODEL_PATH)

        train_indices = rng.permutation(len(x_train))
        x_train_full = x_train[train_indices]
        y_train_full = y_train[train_indices]
        x_train_fit, y_train_fit, x_val, y_val = split_train_val(x_train_full, y_train_full, VALIDATION_SPLIT)

        print()
        print('Stage 2/2: train transfer head on 10 classes...')
        transfer_model.fit(
            x_train_fit, y_train_fit,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(x_val, y_val),
            callbacks=[checkpoint_transfer],
            shuffle=True
        )

        # Сравнение точности по всем классам с моделью из п.2
        print()
        print('Compare classes accuracy with base model from task 2...')
        transfer_model = load_model(TRANSFER_MODEL_PATH)

        if os.path.exists(MODEL_PATH):
            base_model = load_model(MODEL_PATH)
            for class_label in range(OUTPUT_UNITS):
                base_acc = class_accuracy(base_model, x_test, y_test, class_label)
                transfer_acc = class_accuracy(transfer_model, x_test, y_test, class_label)

                print(
                    f'Class {class_label} ({CLASSES[class_label]}): '
                    f'base={base_acc:.3f}, transfer={transfer_acc:.3f}, '
                    f'delta={transfer_acc - base_acc:+.3f}'
                )
        else:
            print(f'Base model is not found: {MODEL_PATH}')
            print('Run task 2 training first to compare with transfer learning result.')
