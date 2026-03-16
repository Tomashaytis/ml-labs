import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict
from collections import Counter

from src.core import SampleParams


class KNNClassifier:
    def __init__(self, sample_params: List[SampleParams], n_neighbours = 1, distance: str = 'euclidean', params: dict = None):
        if len(sample_params) < 1:
            raise ValueError('Sample is empty')
        self._sample_params: Dict[int, SampleParams] = {}

        for sample_param in sample_params:
            self._sample_params[sample_param.class_label] = sample_param

        self._n_neighbours = n_neighbours
        self._distance_type = distance
        self._params = { 'p': 3 } if params is None else params
        self._train_sample = np.array([])
        self._train_labels = np.array([])

    @property
    def trainable(self):
        return True

    @property
    def name(self):
        return "KNN"

    def fit(self, sample: np.array, true_labels: np.ndarray):
        self._train_sample = sample.copy()
        self._train_labels = true_labels.copy()

    def predict(self, point: np.ndarray) -> int:
        """
        d_i(x) = K_i
        """
        distances = self.find_distances(self._train_sample, point)

        k_nearest_indices = np.argpartition(distances, self._n_neighbours)[:self._n_neighbours]
        nearest_labels = self._train_labels[k_nearest_indices]

        d_values = Counter(nearest_labels)
        return max(d_values, key=d_values.get)

    def find_distances(self, sample: np.ndarray, point: np.ndarray) -> np.float64:
        if self._distance_type == 'euclidean':
            return np.linalg.norm(sample - point, axis=1)

        elif self._distance_type == 'manhattan':
            return np.sum(np.abs(sample - point), axis=1)

        elif self._distance_type == 'chebyshev':
            return np.max(np.abs(sample - point), axis=1)

        elif self._distance_type == 'minkowski':
            p = self._params['p']
            diff = sample - point
            return np.sum(np.abs(diff) ** p, axis=1) ** (1 / p)

        elif self._distance_type == 'cosine':
            norm_train = np.linalg.norm(sample, axis=1)
            norm_point = np.linalg.norm(point)
            dot_products = np.dot(sample, point)

            with np.errstate(divide='ignore', invalid='ignore'):
                similarities = dot_products / (norm_train * norm_point)
                similarities = np.nan_to_num(similarities, nan=0.0)

            return 1 - similarities

        raise ValueError(f'Unknown distance type \'{self._distance_type}\'')

    @staticmethod
    def scott_bandwidth(sample: np.ndarray) -> float:
        n, d = sample.shape
        return n ** (-1 / (d + 4))

    @staticmethod
    def silverman_bandwidth(sample: np.ndarray) -> float:
        n, d = sample.shape
        return (n * (d + 2) / 4) ** (-1 / (d + 4))

    def calculate_real_errors(self, sample: np.array, true_labels: np.ndarray, should_print = False) -> dict:
        predictions = np.array([self.predict(point) for point in sample])

        errors = {}
        for class_label_i in self._sample_params.keys():
            for class_label_j in self._sample_params.keys():
                if class_label_i != class_label_j:
                    mask = (true_labels == class_label_i) | (true_labels == class_label_j)
                    pred_pair = predictions[mask]
                    true_pair = true_labels[mask]

                    p_i_error = np.mean((pred_pair == class_label_j) & (true_pair == class_label_i))

                    errors[(class_label_i, class_label_j)] = p_i_error

        if should_print:
            for labels, p_error in errors.items():
                print(f'Ошибки для классов {labels[0]} и {labels[1]}: p_{labels[0]}{labels[1]} = {p_error:3f}')

        return errors

    def calculate_risk(self, sample: np.ndarray, true_labels: np.ndarray) -> float:
        errors = self.calculate_real_errors(sample, true_labels)
        risk = 0

        for i in range(len(self._sample_params)):
            for j in range(len(self._sample_params)):
                if i != j:
                    risk += errors[i, j] * self._sample_params[i].prior

        return risk

    def plot(self, sample: np.array, true_labels: np.ndarray = None):
        x_min, x_max = np.min(sample[:,0]) - 1, np.max(sample[:,0]) + 1
        y_min, y_max = np.min(sample[:,1]) - 1, np.max(sample[:,1]) + 1

        h = 0.05
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_predictions = np.array([self.predict(point) for point in grid_points])
        z = grid_predictions.reshape(xx.shape)

        plt.contour(xx, yy, z, colors='black', linewidths=0.5)

        predictions = np.array([self.predict(point) for point in sample])

        if true_labels is None:
            true_labels = predictions

        correct_samples_by_class = {}
        incorrect_samples_by_class = {}
        for class_label in np.unique(predictions):
            correct_samples_by_class[class_label] = []
            incorrect_samples_by_class[class_label] = []

        for i in range(len(predictions)):
            if predictions[i] == true_labels[i]:
                correct_samples_by_class[predictions[i]].append(sample[i])
            else:
                incorrect_samples_by_class[predictions[i]].append(sample[i])

        for class_label, class_sample in correct_samples_by_class.items():
            if len(class_sample) == 0:
                continue
            class_sample = np.array(class_sample)

            plt.scatter(
                class_sample[:, 0],
                class_sample[:, 1],
                marker='.',
                color=self._sample_params[class_label].class_color,
                label=self._sample_params[class_label].class_name
            )

        for class_label, class_sample in incorrect_samples_by_class.items():
            if len(class_sample) == 0:
                continue
            class_sample = np.array(class_sample)

            plt.scatter(
                class_sample[:, 0],
                class_sample[:, 1],
                marker='x',
                color=self._sample_params[class_label].class_color,
                label=f'{self._sample_params[class_label].class_name} (err)'
            )

        plt.title('Классификатор K ближайших соседей')
        plt.legend()
        plt.show()