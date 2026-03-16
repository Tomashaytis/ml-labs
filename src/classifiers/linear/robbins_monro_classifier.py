import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from src.core import SampleParams


class RobbinsMonroClassifier:
    def __init__(self, sample_param0: SampleParams, sample_param1: SampleParams,
                 is_min_std = True, b: float = 0.75, start_weights = None):
        self._sample_param0 = sample_param0
        self._sample_param1 = sample_param1
        self._is_min_std = is_min_std
        self._k = 1
        self._b = b

        self._weights = np.zeros(sample_param0.dimensional) if start_weights is None else start_weights[:-1]
        self._threshold = 0 if start_weights is None else start_weights[-1]

    @property
    def coef_(self):
        return self._weights.reshape(1, -1)

    @property
    def intercept_(self):
        return np.array([self._threshold])

    @property
    def trainable(self):
        return True

    @property
    def name(self):
        return "Robbins-Monro"

    def fit(self, sample: np.array, true_labels: np.ndarray,
                      epochs: int = 100, epsilon: float = 0.001, should_plot: bool = False) -> int:
        self._k = 1
        w = []

        for epoch in range(epochs):
            for i in range(len(sample)):
                w.append(self._weights.copy())
                self.fit_one(sample[i], int(true_labels[i]))
                if should_plot and len(w) % 15 == 0:
                    self.plot_line(sample)
                if len(w) < 100:
                    continue
                if np.abs(self.check_stop(w[-100], self._weights)) > 1 - epsilon:
                    return self._k

        return self._k

    @staticmethod
    def check_stop(w_prev: np.ndarray, w_next: np.ndarray) -> float:
        if np.linalg.norm(w_prev) == 0 or np.linalg.norm(w_next) == 0:
            return 0
        return np.sum(w_prev * w_next) / (np.linalg.norm(w_prev) * np.linalg.norm(w_next))

    def fit_one(self, point: np.ndarray, true_label: int):
        w = np.append(self._weights, self._threshold).reshape(-1, 1)
        z = np.append(point, 1).reshape(-1, 1)
        r = -1 if true_label == self._sample_param0.class_label else 1
        if self._is_min_std:
            w = w + self._get_a() * z * (r - w.T @ z)
            self._weights = w[:-1]
            self._threshold = w[-1]
        else:
            if r >= w.T @ z:
                w = w + self._get_a() * z
            else:
                w = w - self._get_a() * z
            self._weights = w[:-1]
            self._threshold = w[-1]

    def _get_a(self) -> float:
        a = 1 / (self._k ** self._b)
        self._k += 1
        return a

    def predict(self, point: np.array) -> int:
        d = self._weights.T @ point + self._threshold
        return self._sample_param1.class_label if d > 0 else self._sample_param0.class_label

    def calculate_real_errors(self, sample: np.array, true_labels: np.ndarray) -> dict:
        predictions = np.array([self.predict(point) for point in sample])
        class_label_0 = self._sample_param0.class_label
        class_label_1 = self._sample_param1.class_label

        mask = (true_labels == class_label_0) | (true_labels == class_label_1)
        pred_pair = predictions[mask]
        true_pair = true_labels[mask]

        p0_error = np.mean((pred_pair == class_label_1) & (true_pair == class_label_0))
        p1_error = np.mean((pred_pair == class_label_0) & (true_pair == class_label_1))

        errors = {(0, 1): p0_error, (1, 0): p1_error}

        return errors

    def calculate_risk(self, sample: np.ndarray, true_labels: np.ndarray) -> float:
        errors = self.calculate_real_errors(sample, true_labels)
        risk = self._sample_param0.prior * errors[(0, 1)] + self._sample_param1.prior * errors[(1, 0)]
        return risk

    def plot(self, sample: np.array, true_labels: np.ndarray = None):
        x_min, x_max = np.min(sample[:, 0]) - 1, np.max(sample[:, 0]) + 1
        y_min, y_max = np.min(sample[:, 1]) - 1, np.max(sample[:, 1]) + 1

        h = 0.02
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_predictions = np.array([self.predict(point) for point in grid_points])
        z = grid_predictions.reshape(xx.shape)

        plt.contourf(xx, yy, z, alpha=0.4, cmap=plt.cm.RdYlBu)
        plt.contour(xx, yy, z, colors='black', linewidths=0.5)

        predictions = np.array([self.predict(point) for point in sample])

        if true_labels is None:
            true_labels = predictions

        correct_samples_by_class = {}
        incorrect_samples_by_class = {}

        unique_labels = np.unique(true_labels)
        for class_label in unique_labels:
            correct_samples_by_class[class_label] = []
            incorrect_samples_by_class[class_label] = []

        for i in range(len(predictions)):
            predicted_class = predictions[i]
            true_class = true_labels[i]

            if predicted_class == true_class:
                if predicted_class in correct_samples_by_class:
                    correct_samples_by_class[predicted_class].append(sample[i])
                else:
                    if 'other' not in correct_samples_by_class:
                        correct_samples_by_class['other'] = []
                    correct_samples_by_class['other'].append(sample[i])
            else:
                if predicted_class in incorrect_samples_by_class:
                    incorrect_samples_by_class[predicted_class].append(sample[i])
                else:
                    if 'other' not in incorrect_samples_by_class:
                        incorrect_samples_by_class['other'] = []
                    incorrect_samples_by_class['other'].append(sample[i])

        def get_class_params(class_label):
            if class_label == self._sample_param0.class_label:
                return self._sample_param0
            elif class_label == self._sample_param1.class_label:
                return self._sample_param1
            else:
                return SampleParams(
                    prior=0,
                    dimensional=2,
                    class_label=class_label,
                    class_name=f'Class {class_label}',
                    class_color='gray'
                )

        for class_label, class_sample in correct_samples_by_class.items():
            if len(class_sample) == 0:
                continue
            class_sample = np.array(class_sample)

            if class_label == 'other':
                plt.scatter(
                    class_sample[:, 0],
                    class_sample[:, 1],
                    marker='.',
                    color='gray',
                    label='Other classes'
                )
            else:
                params = get_class_params(class_label)
                plt.scatter(
                    class_sample[:, 0],
                    class_sample[:, 1],
                    marker='.',
                    color=params.class_color,
                    label=params.class_name
                )

        for class_label, class_sample in incorrect_samples_by_class.items():
            if len(class_sample) == 0:
                continue
            class_sample = np.array(class_sample)

            if class_label == 'other':
                plt.scatter(
                    class_sample[:, 0],
                    class_sample[:, 1],
                    marker='x',
                    color='gray',
                    label='Other classes (err)'
                )
            else:
                params = get_class_params(class_label)
                plt.scatter(
                    class_sample[:, 0],
                    class_sample[:, 1],
                    marker='x',
                    color=params.class_color,
                    label=f'{params.class_name} (err)'
                )

        plt.title(f"Классификатор Роббинса-Монро ({'НСКО-алгоритм' if self._is_min_std else 'АКП-алгоритм'})")
        plt.legend()
        plt.show()

    def find_line(self, m_sym):
        w_vector = sp.Matrix(self._weights)
        discriminant = w_vector.T.dot(m_sym) + self._threshold
        return sp.simplify(discriminant[0])

    def plot_line(self, sample: np.array):
        x_min, x_max = np.min(sample[:, 0]) - 1, np.max(sample[:, 0]) + 1
        y_min, y_max = np.min(sample[:, 1]) - 1, np.max(sample[:, 1]) + 1

        x_sym, y_sym = sp.symbols("x y")
        matrix_sym = sp.Matrix([x_sym, y_sym])
        equation = self.find_line(matrix_sym)
        f_border = sp.lambdify([x_sym, y_sym], equation, 'numpy')

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))

        plt.contour(xx, yy, f_border(xx, yy), levels=[0], colors='green', linewidths=1)

    def plot_with_lines(self, sample: np.array, colors: dict, true_labels: np.ndarray = None):
        x_min, x_max = np.min(sample[:, 0]) - 1, np.max(sample[:, 0]) + 1
        y_min, y_max = np.min(sample[:, 1]) - 1, np.max(sample[:, 1]) + 1

        x_sym, y_sym = sp.symbols("x y")
        matrix_sym = sp.Matrix([x_sym, y_sym])
        equation = self.find_line(matrix_sym)
        f_border = sp.lambdify([x_sym, y_sym], equation, 'numpy')

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))

        class_label_0 = self._sample_param0.class_label
        class_label_1 = self._sample_param1.class_label

        plt.contour(xx, yy, f_border(xx, yy), levels=[0], colors=colors[(class_label_0, class_label_1)], linewidths=2)
        plt.plot([0, 0], [0, 0], color=colors[(class_label_0, class_label_1)], linewidth=3,
                 label=f'классы {class_label_0} и {class_label_1}')

        predictions = np.array([self.predict(point) for point in sample])

        if true_labels is None:
            true_labels = predictions

        correct_samples_by_class = {}
        incorrect_samples_by_class = {}

        unique_labels = np.unique(true_labels)
        for class_label in unique_labels:
            correct_samples_by_class[class_label] = []
            incorrect_samples_by_class[class_label] = []

        for i in range(len(predictions)):
            predicted_class = predictions[i]
            true_class = true_labels[i]

            if predicted_class == true_class:
                if predicted_class in correct_samples_by_class:
                    correct_samples_by_class[predicted_class].append(sample[i])
                else:
                    if 'other' not in correct_samples_by_class:
                        correct_samples_by_class['other'] = []
                    correct_samples_by_class['other'].append(sample[i])
            else:
                if predicted_class in incorrect_samples_by_class:
                    incorrect_samples_by_class[predicted_class].append(sample[i])
                else:
                    if 'other' not in incorrect_samples_by_class:
                        incorrect_samples_by_class['other'] = []
                    incorrect_samples_by_class['other'].append(sample[i])

        def get_class_params(class_label):
            if class_label == self._sample_param0.class_label:
                return self._sample_param0
            elif class_label == self._sample_param1.class_label:
                return self._sample_param1
            else:
                return SampleParams(
                    prior=0,
                    dimensional=2,
                    class_label=class_label,
                    class_name=f'Class {class_label}',
                    class_color='gray'
                )

        for class_label, class_sample in correct_samples_by_class.items():
            if len(class_sample) == 0:
                continue
            class_sample = np.array(class_sample)

            if class_label == 'other':
                plt.scatter(
                    class_sample[:, 0],
                    class_sample[:, 1],
                    marker='.',
                    color='gray',
                    label='Other classes'
                )
            else:
                params = get_class_params(class_label)
                plt.scatter(
                    class_sample[:, 0],
                    class_sample[:, 1],
                    marker='.',
                    color=params.class_color,
                    label=params.class_name
                )

        for class_label, class_sample in incorrect_samples_by_class.items():
            if len(class_sample) == 0:
                continue
            class_sample = np.array(class_sample)

            if class_label == 'other':
                plt.scatter(
                    class_sample[:, 0],
                    class_sample[:, 1],
                    marker='x',
                    color='gray',
                    label='Other classes (err)'
                )
            else:
                params = get_class_params(class_label)
                plt.scatter(
                    class_sample[:, 0],
                    class_sample[:, 1],
                    marker='x',
                    color=params.class_color,
                    label=f'{params.class_name} (err)'
                )

        plt.title(f"Классификатор Роббинса-Монро ({'НСКО-алгоритм' if self._is_min_std else 'АКП-алгоритм'})")
        plt.legend()
        plt.show()
