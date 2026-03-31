import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import tensorflow as tf

from src.draw import ImagePlotter

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
SEED = 42

SHOW_DATASET_STATS = True


if __name__ == '__main__':
    plotter = ImagePlotter(cmap='gray', heatmap_cmap='Reds')
    tf.get_logger().setLevel('ERROR')
    tf.random.set_seed(SEED)

    # Загрузка и отображение датасета
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
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

    x_train, x_test = x_train / 255.0, x_test / 255.0
