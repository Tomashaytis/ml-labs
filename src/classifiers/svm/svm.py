import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from qpsolvers import solve_qp
from src.core import SampleParams


class SVM:
    def __init__(self, sample_param0: SampleParams, sample_param1: SampleParams, C: float = np.inf):
        self._sample_param0 = sample_param0
        self._sample_param1 = sample_param1

        self._C = C
        self._support_vectors = np.array([])
        self._weights = np.zeros(sample_param0.dimensional)
        self._threshold = 0

    @property
    def coef_(self):
        return self._weights.reshape(1, -1)

    @property
    def intercept_(self):
        return np.array([self._threshold])

    @property
    def support_vectors_(self):
        return self._support_vectors

    @property
    def trainable(self):
        return True

    @property
    def name(self):
        return "SVM"

    def fit(self, sample: np.array, true_labels: np.ndarray, solver: str = "proxqp"):
        """
        (1/2)αᵀPa + qᵀα -> min
        Aa = b
        Gα ≤ h
        """
        n_samples = len(true_labels)
        X = sample
        r = np.ones(true_labels.shape[0])
        r[true_labels == self._sample_param0.class_label] = -1

        P = np.outer(r, r) * (X @ X.T)                  # матрица P: P_ij = r_i * r_j * x_iᵀ * x_j
        q = -np.ones(n_samples)                         # вектор q: q_i = -1

        A = r.reshape(1, -1)                            # вектор A: A_i = r_i
        b = np.array([0.0])                             # скаляр b: b = 0

        G = np.vstack([                                 # матрица G: G = [-I, I]ᵀ
            -np.eye(n_samples),                         # матрица -I: a ≥ 0 -> -a ≤ 0
            np.eye(n_samples)                           # матрица I: a ≤ C
        ])
        h = np.hstack([                                 # вектор h: [0, ..., 0, C, ..., C]
            np.zeros(n_samples),
            self._C * np.ones(n_samples)
        ])

        a = solve_qp(P, q, G, h, A, b, solver=solver)   # вектор a: a_i = λ_i

        if a is None or len(a) == 0:
            return

        self._weights = a * r @ X
        self._support_vectors = X[a > 1e-5]
        sv_r = r[np.where(a > 1e-5)]
        self._threshold = 0 if len(self._support_vectors) == 0 else np.mean(sv_r - self._support_vectors @ self._weights)

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

        plt.title('Классификатор SVM')
        plt.legend()
        plt.show()

    def find_line(self, m_sym):
        w_vector = sp.Matrix(self._weights)
        discriminant = w_vector.T.dot(m_sym) + self._threshold
        return sp.simplify(discriminant)

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

        plt.title('Классификатор SVM')
        plt.legend()
        plt.show()
