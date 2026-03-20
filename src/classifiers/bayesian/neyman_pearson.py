import numpy as np

from scipy.stats import norm
from dataclasses import replace

from src.core import NormalSampleParams
from src.utils.sem1.lab1 import mahalanobis_distance
from .bayesian import Bayesian


class NeymanPearson:
    def __init__(self, sample_param0: NormalSampleParams, sample_param1: NormalSampleParams, max_error_prob, is_equal_covariance = False):
        self._sample_param0 = replace(sample_param0, mean=sample_param0.mean.copy(),
                                      covariance=sample_param0.covariance.copy())
        self._sample_param1 = replace(sample_param1, mean=sample_param1.mean.copy(),
                                      covariance=sample_param0.covariance.copy())
        self._max_error_prob = max_error_prob
        self._common_covariance = sample_param0.covariance if is_equal_covariance else None

        likelihood_ratio = self.find_likelihood_ratio()

        self._sample_param0.prior = likelihood_ratio / (likelihood_ratio + 1)
        self._sample_param1.prior = 1 / (likelihood_ratio + 1)

        self._bayesian_classifier = Bayesian(
            [self._sample_param0, self._sample_param1],
            is_equal_covariance
        )

    @property
    def trainable(self):
        return False

    @property
    def name(self):
        return "Neyman-Pearson"

    def find_likelihood_ratio(self) -> float:
        if self._common_covariance is None:
            raise NotImplemented('find_likelihood_ratio not implemented in general case')

        p = mahalanobis_distance(
            self._sample_param0.mean,
            self._sample_param1.mean,
            self._common_covariance
        )

        return np.exp(-1 / 2 * p + np.sqrt(p) * norm.ppf(1 - self._max_error_prob))

    def predict(self, point: np.ndarray) -> int:
        return self._bayesian_classifier.predict(point)

    def calculate_theoretical_errors(self, should_print: bool = False) -> dict:
        return self._bayesian_classifier.calculate_theoretical_errors(should_print)

    def calculate_real_errors(self, sample: np.array, true_labels: np.ndarray = None, should_print = False) -> dict:
        return self._bayesian_classifier.calculate_real_errors(sample, true_labels, should_print)

    def plot(self, sample: np.array, true_labels: np.ndarray = None):
        return self._bayesian_classifier.plot(sample, true_labels)

    def plot_with_lines(self, sample: np.array, colors: dict, true_labels: np.ndarray = None):
        return self._bayesian_classifier.plot_with_lines(sample, colors, true_labels)
