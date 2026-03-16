import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict

from src.core import SampleParams


class KDEClassifier:
    def __init__(self, sample_params: List[SampleParams], bandwidth: str | float = 'default',
                 kernel: str = 'gaussian', kernel_construction: str = 'product', params: dict = None):
        if len(sample_params) < 1:
            raise ValueError('Sample is empty')
        self._sample_params: Dict[int, SampleParams] = {}

        for sample_param in sample_params:
            self._sample_params[sample_param.class_label] = sample_param

        self._bandwidth = bandwidth
        self._kernel_type = kernel
        self._kernel_construction = kernel_construction
        if params is None:
            self._params = {
                'k': 0.25
            }

        self._train_samples: Dict[int, np.ndarray] = {}
        self._covariances: Dict[int, np.ndarray] = {}

    @property
    def trainable(self):
        return True

    @property
    def name(self):
        return "KDE"

    def fit(self, sample: np.array, true_labels: np.ndarray):
        if self._bandwidth == 'default':
            self._bandwidth = self.default_bandwidth(sample, self._params['k'])
        elif self._bandwidth == 'scott':
            self._bandwidth = self.scott_bandwidth(sample)
        elif self._bandwidth == 'silverman':
            self._bandwidth = self.silverman_bandwidth(sample)
        else:
            raise ValueError(f'Unknown bandwidth type \'{self._bandwidth}\'')

        for class_label in self._sample_params.keys():
            self._train_samples[class_label] = sample[true_labels == class_label].copy()
            if self._kernel_construction == 'gaussian':
                sample_centered = self._train_samples[class_label] - np.mean(self._train_samples[class_label], axis=0)
                self._covariances[class_label] = 1 / (len(self._train_samples[class_label]) - 1) * sample_centered.T @ sample_centered

    def predict(self, point: np.ndarray) -> int:
        """
        d_i(x) = f(x|Ω_i)P(Ω_i)
        ln(d_i(x)) = ln(f(x|Ω_i)) + ln(P(Ω_i))
        """
        d_values = {}

        for class_label, train_samples in self._train_samples.items():
            covariance = self._covariances[class_label] if self._kernel_construction == 'gaussian' else None
            likelihood = self.estimate_density(train_samples, point, covariance)

            d_i = np.log(likelihood) + np.log(self._sample_params[class_label].prior)
            d_values[class_label] = d_i

        return max(d_values, key=d_values.get)

    def estimate_density(self, sample: np.ndarray, x: np.ndarray, covariance = None) -> np.float64:
        N, n = sample.shape
        h = self._bandwidth

        if self._kernel_construction == 'gaussian':
            inv_cov = np.linalg.inv(covariance)
            det_cov = np.linalg.det(covariance)

            coef = 1 / np.sqrt((2 * np.pi) ** n * det_cov)
            diffs = sample - x

            # (x-x_i).T * B^(-1) * (x-x_i)
            mahalanobis_sq = np.einsum('ij,jk,ik->i', diffs, inv_cov, diffs)

            kernels = np.exp(-1 / (2 * h ** 2) * mahalanobis_sq)

            return 1 / (N * h ** n) * coef * np.sum(kernels)

        elif self._kernel_construction == 'product':
            U = (x - sample) / h
            kernels = np.prod(self.kernel_func(U), axis=1)

        elif self._kernel_construction == 'radial':
            U = (x - sample) / h
            r = np.linalg.norm(U, axis=1)
            kernels = self.kernel_func(r.reshape(-1, 1)).flatten()

        else:
            raise ValueError(f'Unknown kernel construction type \'{self._kernel_construction}\'')

        kernel_sum = np.sum(kernels)
        return 1 / (N * h ** n) * kernel_sum

    def kernel_func(self, u: np.ndarray) -> np.ndarray:
        if self._kernel_type == 'rectangular':
            return np.where(np.abs(u) <= 1, 0.5, 0.0)

        elif self._kernel_type == 'triangular':
            return np.where(np.abs(u) <= 1, 1 - np.abs(u), 0.0)

        elif self._kernel_type == 'gaussian':
            return 1 / np.sqrt(2 * np.pi) * np.exp(-u ** 2 / 2)

        elif self._kernel_type == 'laplacian':
            return 1 / 2 * np.exp(-np.abs(u))

        elif self._kernel_type == 'cauchy':
            return 1 / np.pi * 1 / (1 + u ** 2)

        elif self._kernel_type == 'sinc':
            return 1 / np.pi * np.sinc(u / 2) ** 2

        raise ValueError(f'Unknown kernel type \'{self._kernel_type}\'')

    def decision_function(self, point):
        d_values = {}
        for class_label, train_samples in self._train_samples.items():
            covariance = self._covariances[class_label] if self._kernel_construction == 'gaussian' else None
            likelihood = self.estimate_density(train_samples, point, covariance)
            d_values[class_label] = self._sample_params[class_label].prior * likelihood

        if len(d_values) == 2:
            classes = list(d_values.keys())
            return d_values[classes[1]] - d_values[classes[0]]
        else:
            return max(d_values.values())

    @staticmethod
    def default_bandwidth(sample: np.ndarray, k: float) -> float:
        if not (0 < k < 1 / 2):
            raise ValueError('k should be between 0 and 1 / 2')
        n, d = sample.shape
        return n ** (- k / d)

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
        grid_decisions = np.array([self.decision_function(point) for point in grid_points])
        z = grid_decisions.reshape(xx.shape)

        plt.contour(xx, yy, z, colors='black', levels=[0], linewidths=1)

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

        plt.title('Классификатор Парзена')
        plt.legend()
        plt.show()