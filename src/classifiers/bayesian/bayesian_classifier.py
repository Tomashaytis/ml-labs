import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from dataclasses import replace
from typing import List, Dict
from scipy.stats import norm
from src.core import NormalSampleParams
from src.utils import mahalanobis_distance


class BayesianClassifier:
    def __init__(self, sample_params: List[NormalSampleParams], is_equal_covariance = False):
        if len(sample_params) < 1:
            raise ValueError('Sample is empty')
        self._sample_params: Dict[int, NormalSampleParams] = {}
        self._common_covariance = sample_params[0].covariance if is_equal_covariance else None

        for sample_param in sample_params:
            self._sample_params[sample_param.class_label] = replace(sample_param,
                                                                    mean=sample_param.mean.copy(),
                                                                    covariance=sample_param.covariance.copy())

    @property
    def trainable(self):
        return False

    @property
    def name(self):
        return "Bayesian"

    def predict(self, point: np.ndarray) -> int:
        point_class_label = 0
        max_disc_diff = 0

        first = True
        for class_label in self._sample_params.keys():
            if first:
                point_class_label = class_label
                if self._common_covariance is None:
                    max_disc_diff = self.discriminant_diff(class_label, point)
                else:
                    max_disc_diff = self.linear_discriminant_diff(class_label, point)
                first = False
                continue

            if self._common_covariance is None:
                disc_diff = self.discriminant_diff(class_label, point)
            else:
                disc_diff = self.linear_discriminant_diff(class_label, point)

            if disc_diff > max_disc_diff:
                max_disc_diff = disc_diff
                point_class_label = class_label

        return point_class_label

    def discriminant_diff(self, class_label: int, point: np.array) -> float:
        sample_param = self._sample_params[class_label]
        return (np.log(sample_param.prior)
                - 1 / 2 * np.log(np.linalg.det(sample_param.covariance))
                - 1 / 2 * mahalanobis_distance(point, sample_param.mean, sample_param.covariance))

    def linear_discriminant_diff(self, class_label: int, point: np.ndarray) -> float:
        if self._common_covariance is None:
            raise ValueError('Common covariance is not set')

        sample_param = self._sample_params[class_label]
        return (np.log(sample_param.prior)
                + sample_param.mean.T @ np.linalg.inv(self._common_covariance) @ point
                - 1 / 2 * sample_param.mean.T @ np.linalg.inv(self._common_covariance) @ sample_param.mean)

    def decision_function(self, point):
        posteriors = {}
        for class_label in self._sample_params.keys():
            sample_param = self._sample_params[class_label]
            likelihood = 1 / np.sqrt(2 * np.pi * np.linalg.det(sample_param.covariance)) \
                         * np.exp(- 1 / 2 * (point - sample_param.mean).T \
                                  @ np.linalg.inv(sample_param.covariance) \
                                  @ (point - sample_param.mean))
            posteriors[class_label] = sample_param.prior * likelihood

        if len(posteriors) == 2:
            classes = list(posteriors.keys())
            return posteriors[classes[1]] - posteriors[classes[0]]
        else:
            return max(posteriors.values())

    def calculate_theoretical_errors(self, should_print = False) -> dict:
        if self._common_covariance is None:
            raise NotImplemented('calculate_errors not implemented in general case')

        errors = {}
        for class_label_i in self._sample_params.keys():
            for class_label_j in self._sample_params.keys():
                if class_label_i != class_label_j and (class_label_j, class_label_i) not in errors:
                    likelihood_ratio = np.log(self._sample_params[class_label_i].prior / self._sample_params[class_label_j].prior)
                    p = mahalanobis_distance(
                        self._sample_params[class_label_i].mean,
                        self._sample_params[class_label_j].mean,
                        self._common_covariance
                    )
                    p_i_error = norm.cdf(likelihood_ratio - 1 / 2 * p / np.sqrt(p))
                    p_j_error = 1 - norm.cdf(likelihood_ratio + 1 / 2 * p / np.sqrt(p))
                    errors[(class_label_i, class_label_j)] = p_i_error, p_j_error

        if should_print:
            for labels, p_errors in errors.items():
                print(f'Ошибки для классов {labels[0]} и {labels[1]}: p_{labels[0]} = {p_errors[0]:3f}, p_{labels[1]} = {p_errors[1]:3f}')

        return errors

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

        plt.title('Байесовский классификатор')
        plt.legend()
        plt.show()

    def find_line(self, label1: int, label2: int, matrix_sym):
        if self._common_covariance is None:
            cov1_inv = np.linalg.inv(self._sample_params[label1].covariance)
            cov2_inv = np.linalg.inv(self._sample_params[label2].covariance)
        else:
            cov1_inv = np.linalg.inv(self._common_covariance)
            cov2_inv = np.linalg.inv(self._common_covariance)

        ln_det_ratio = np.log(np.linalg.det(cov2_inv) / np.linalg.det(cov1_inv))

        def mahalanobis_sym(x, m, cov_inv):
            diff = x - sp.Matrix(m)
            return (diff.T * sp.Matrix(cov_inv) * diff)[0, 0]

        p1 = mahalanobis_sym(matrix_sym, self._sample_params[label1].mean, cov1_inv)
        p2 = mahalanobis_sym(matrix_sym, self._sample_params[label2].mean, cov2_inv)

        return sp.simplify(p1 - p2 - ln_det_ratio)

    def plot_with_lines(self, sample: np.array, colors: dict, true_labels: np.ndarray = None):
        x_min, x_max = np.min(sample[:, 0]) - 1, np.max(sample[:, 0]) + 1
        y_min, y_max = np.min(sample[:, 1]) - 1, np.max(sample[:, 1]) + 1

        processed = []
        for class_label_i in self._sample_params.keys():
            for class_label_j in self._sample_params.keys():
                if class_label_i != class_label_j and (class_label_j, class_label_i) not in processed:
                    x_sym, y_sym = sp.symbols("x y")
                    matrix_sym = sp.Matrix([x_sym, y_sym])
                    equation = self.find_line(class_label_i, class_label_j, matrix_sym)
                    f_border = sp.lambdify((x_sym, y_sym), equation, 'numpy')

                    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                                         np.linspace(y_min, y_max, 200))

                    plt.contour(xx, yy, f_border(xx, yy), levels=[0], colors=colors[(class_label_i, class_label_j)], linewidths=2)
                    plt.plot([0, 0], [0, 0], color=colors[(class_label_i, class_label_j)], linewidth=3,
                             label=f'классы {class_label_i} и {class_label_j}')

                    processed.append((class_label_i, class_label_j))

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

        plt.title('Байесовский классификатор')
        plt.legend()
        plt.show()




