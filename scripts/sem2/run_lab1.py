import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import tensorflow as tf

from keras.datasets import mnist
from keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier

from src.draw import ImagePlotter
from src.utils.sem2.lab1 import grid_search_nn, create_nn_model

EPOCHS = 10
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
SEED = 42

OPTIMIZER = 'adam'
LOSS = 'sparse_categorical_crossentropy'
METRIC = 'accuracy'

INPUT_UNITS = 28 * 28
HIDDEN_ACTIVATION = 'relu'
HIDDEN_UNITS = 2 * INPUT_UNITS + 1
HIDDEN_DROPOUT_RATE = 0.2
OUTPUT_ACTIVATION = 'softmax'
OUTPUT_UNITS = 10

GRID_HIDDEN_NEURONS = [1500, 1750]
GRID_BATCH_SIZES = [16, 32]

OPTION = 'train'  # 'train', 'eval', 'grid', 'grid-cv'
MODEL_PATH = os.path.join('models', 'nn_model.keras')
SHOW_DATASET_STATS = True


if __name__ == '__main__':
    plotter = ImagePlotter(cmap='gray', heatmap_cmap='Reds')
    tf.get_logger().setLevel('ERROR')
    tf.random.set_seed(SEED)

    # Загрузка и отображение датасета
    print()
    print('Load dataset...')
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if SHOW_DATASET_STATS:
        plotter.imshow_mnist(x_train, y_train)
        print(f'Train sample length: {len(x_train)}')
        print(f'Test sample length: {len(x_test)}')

        print()
        train_classes = np.unique(y_train, return_counts=True)
        print("Train classes distribution:")
        for digit, count in zip(train_classes[0], train_classes[1]):
            print(f"Digit {digit}: {count} ({count / len(y_train) * 100:.2f}%)")

    # Разворачивание в строку
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    # Перемешивание train-выборки перед разбиением на train/val
    rng = np.random.default_rng(SEED)
    train_indices = rng.permutation(len(x_train))
    x_train = x_train[train_indices]
    y_train = y_train[train_indices]

    if OPTION == 'train':
        # Создание модели с одним скрытым слоем
        nn_model = Sequential([
            # Входной слой
            Input(shape=(INPUT_UNITS,), name='input'),

            # Скрытый слой
            Dense(HIDDEN_UNITS, activation=HIDDEN_ACTIVATION, name='hidden'),
            Dropout(HIDDEN_DROPOUT_RATE, name='hidden_dropout'),

            # Выходной слой
            Dense(OUTPUT_UNITS, activation=OUTPUT_ACTIVATION, name='output')
        ])

        # Компиляция модели
        nn_model.compile(
            optimizer=OPTIMIZER,
            loss=LOSS,
            metrics=[METRIC]
        )

        # Обучение модели
        print()
        print('Train model...')
        val_size = int(VALIDATION_SPLIT * len(x_train))
        x_val, y_val = x_train[:val_size], y_train[:val_size]
        x_train_fit, y_train_fit = x_train[val_size:], y_train[val_size:]

        history = nn_model.fit(
            x_train_fit, y_train_fit,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(x_val, y_val),
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

        # Сохранение модели
        print()
        print('Save model...')
        nn_model.save(MODEL_PATH)

    elif OPTION == 'eval':
        # Загрузка модели
        print()
        print('Load model...')
        nn_model = load_model(MODEL_PATH)

        # Качество работы модели на тестовой выборке
        print()
        print('Test model...')
        loss_and_metrics = nn_model.evaluate(x_test, y_test)
        test_loss = loss_and_metrics[0]
        test_accuracy = loss_and_metrics[1]
        print(f'Test loss: {test_loss:.3f}')
        print(f'Test {METRIC}: {test_accuracy:.3f}')

        # Confusion Matrix
        print()
        print('Predict...')
        y_pred_proba = nn_model.predict(x_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        cm = confusion_matrix(y_test, y_pred)
        plotter.plot_confusion_matrix(cm)

        # Вычисление критерия качества по Confusion Matrix
        correct_predictions = np.trace(cm)
        total_predictions = np.sum(cm)
        test_accuracy_cm = correct_predictions / total_predictions
        print(f'Test {METRIC} (CM): {test_accuracy_cm:.3f}')

        # Пример неверно проклассифицированных объектов по каждому классу
        incorrect_mask = (y_pred != y_test)
        incorrect_indices = np.where(incorrect_mask)[0]

        examples_by_class = {}

        for digit in range(10):
            digit_predicted = incorrect_indices[y_pred[incorrect_indices] == digit]

            if len(digit_predicted) > 0:
                idx = digit_predicted[0]
                examples_by_class[digit] = idx

        incorrect_images = np.array([x_test[idx].reshape(28, 28) for idx in examples_by_class.values()])
        incorrect_labels = np.array([y_pred[idx] for idx in examples_by_class.values()])

        plotter.imshow_mnist(incorrect_images, incorrect_labels, title="False predictions per class")

    elif OPTION == 'grid':
        # Перебор параметров по сетке
        params = {
            'optimizer': OPTIMIZER,
            'loss': LOSS,
            'metric': METRIC,
            'input_units': INPUT_UNITS,
            'hidden_activation': HIDDEN_ACTIVATION,
            'output_activation': OUTPUT_ACTIVATION,
            'hidden_dropout_rate': HIDDEN_DROPOUT_RATE,
            'output_units': OUTPUT_UNITS,
            'epochs': EPOCHS,
            'validation_split': VALIDATION_SPLIT,
        }
        _, best_params, best_metric = grid_search_nn(x_train, y_train, GRID_HIDDEN_NEURONS, GRID_BATCH_SIZES, params)

        print()
        print(f'Best {METRIC}: {best_metric:.3f}')
        print(f'Best params: {best_params}')

    elif OPTION == 'grid-cv':
        # Keras обёртка для модели
        nn_model = KerasClassifier(
            model=create_nn_model,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            verbose=0
        )

        # Сетка параметров
        param_grid = {
            'model__hidden_units': GRID_HIDDEN_NEURONS,
            'batch_size': GRID_BATCH_SIZES
        }

        # Инициализация GridSearchCV
        grid_search = GridSearchCV(
            estimator=nn_model,
            param_grid=param_grid,
            cv=5,
            scoring=METRIC,
            n_jobs=-1,
            verbose=2
        )

        # Поиск по сетке GridSearchCV
        print()
        grid_search.fit(x_train, y_train)

        print('Grid Search CV results:')
        print(f'Best {METRIC}: {grid_search.best_score_:.3f}')
        print(f'Best params: {grid_search.best_params_}')

        # Оценка на тестовой выборке
        best_nn_model = grid_search.best_estimator_
        test_metric = best_nn_model.score(x_test, y_test)
        print(f'Test {METRIC}: {test_metric:.3f}')

