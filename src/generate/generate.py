import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, List
from random import shuffle

from src.core import Sample, NormalSampleParams
from src.utils.sem1.lab1 import normal2, mahalanobis_distance, bhattacharyya_distance
from .constants import SIZE, MEAN0, MEAN1, MEAN2, MEAN3, MEAN4, COV0, COV1, COV2, COV3


def generate_2_samples(size: int = SIZE, should_visualize: bool = False, should_print: bool = False) -> Tuple[Sample, Sample]:
    sample0 = np.array([normal2(MEAN0, COV0) for _ in range(size)])
    sample1 = np.array([normal2(MEAN1, COV0) for _ in range(size)])

    if should_print:
        print(f'Расстояние Махаланобиса выборки 0 и 1: {mahalanobis_distance(MEAN0, MEAN1, COV0)}')
        print()

    if should_visualize:
        plt.scatter(sample0[:, 0], sample0[:, 1], marker='x', color='red', label='класс 0')
        plt.scatter(sample1[:, 0], sample1[:, 1], marker='x', color='blue', label='класс 1')
        plt.title('Два класса')
        plt.legend()
        plt.show()

    sample_params0 = NormalSampleParams(prior=1 / 2, mean=MEAN0, covariance=COV0,
                                        class_label=0, class_name='класс 0', class_color='red', dimensional=2)
    sample_params1 = NormalSampleParams(prior=1 / 2, mean=MEAN1, covariance=COV0,
                                        class_label=1, class_name='класс 1', class_color='blue', dimensional=2)

    return (Sample(data=sample0, params=sample_params0),
            Sample(data=sample1, params=sample_params1))


def generate_3_samples(size: int = SIZE, should_visualize: bool = False, should_print: bool = False) \
        -> Tuple[Sample, Sample, Sample]:
    sample0 = np.array([normal2(MEAN0, COV0) for _ in range(size)])
    sample1 = np.array([normal2(MEAN1, COV1) for _ in range(size)])
    sample2 = np.array([normal2(MEAN2, COV2) for _ in range(size)])

    if should_print:
        print(f'Расстояние Бхаттачария выборки 0 и 1: {bhattacharyya_distance(MEAN0, MEAN1, COV0, COV1)}')
        print(f'Расстояние Бхаттачария выборки 1 и 2: {bhattacharyya_distance(MEAN1, MEAN2, COV1, COV2)}')
        print(f'Расстояние Бхаттачария выборки 0 и 2: {bhattacharyya_distance(MEAN0, MEAN2, COV0, COV2)}')
        print()

    if should_visualize:
        plt.scatter(sample0[:, 0], sample0[:, 1], marker='x', color='red', label='класс 0')
        plt.scatter(sample1[:, 0], sample1[:, 1], marker='x', color='blue', label='класс 1')
        plt.scatter(sample2[:, 0], sample2[:, 1], marker='x', color='green', label='класс 2')
        plt.title('Три класса')
        plt.legend()
        plt.show()

    sample_params0 = NormalSampleParams(prior=1 / 3, mean=MEAN0, covariance=COV0,
                                        class_label=0, class_name='класс 0', class_color='red', dimensional=2)
    sample_params1 = NormalSampleParams(prior=1 / 3, mean=MEAN1, covariance=COV1,
                                        class_label=1, class_name='класс 1', class_color='blue', dimensional=2)
    sample_params2 = NormalSampleParams(prior=1 / 3, mean=MEAN2, covariance=COV2,
                                        class_label=2, class_name='класс 2', class_color='green', dimensional=2)

    return (Sample(data=sample0, params=sample_params0),
            Sample(data=sample1, params=sample_params1),
            Sample(data=sample2, params=sample_params2))


def generate_5_samples(size: int = SIZE, should_visualize: bool = False) \
        -> Tuple[Sample, Sample, Sample, Sample, Sample]:
    sample0 = np.array([normal2(MEAN0, COV3) for _ in range(size)])
    sample1 = np.array([normal2(MEAN1, COV3) for _ in range(size)])
    sample2 = np.array([normal2(MEAN2, COV3) for _ in range(size)])
    sample3 = np.array([normal2(MEAN3, COV3) for _ in range(size)])
    sample4 = np.array([normal2(MEAN4, COV3) for _ in range(size)])

    if should_visualize:
        plt.scatter(sample0[:, 0], sample0[:, 1], marker='x', color='red', label='класс 0')
        plt.scatter(sample1[:, 0], sample1[:, 1], marker='x', color='blue', label='класс 1')
        plt.scatter(sample2[:, 0], sample2[:, 1], marker='x', color='green', label='класс 2')
        plt.scatter(sample3[:, 0], sample3[:, 1], marker='x', color='cyan', label='класс 3')
        plt.scatter(sample4[:, 0], sample4[:, 1], marker='x', color='yellow', label='класс 4')
        plt.title('Пять классов')
        plt.legend()
        plt.show()

    sample_params0 = NormalSampleParams(prior=1 / 2, mean=MEAN0, covariance=COV3,
                                        class_label=0, class_name='класс 0', class_color='red', dimensional=2)
    sample_params1 = NormalSampleParams(prior=1 / 2, mean=MEAN1, covariance=COV3,
                                        class_label=1, class_name='класс 1', class_color='blue', dimensional=2)
    sample_params2 = NormalSampleParams(prior=1 / 2, mean=MEAN2, covariance=COV3,
                                        class_label=2, class_name='класс 2', class_color='green', dimensional=2)
    sample_params3 = NormalSampleParams(prior=1 / 2, mean=MEAN3, covariance=COV3,
                                        class_label=3, class_name='класс 3', class_color='cyan', dimensional=2)
    sample_params4 = NormalSampleParams(prior=1 / 2, mean=MEAN4, covariance=COV3,
                                        class_label=4, class_name='класс 4', class_color='yellow', dimensional=2)

    return (
        Sample(data=sample0, params=sample_params0),
        Sample(data=sample1, params=sample_params1),
        Sample(data=sample2, params=sample_params2),
        Sample(data=sample3, params=sample_params3),
        Sample(data=sample4, params=sample_params4),
    )


def randomize_sample(samples: List[Sample]) -> tuple:
    main_sample = []
    main_labels = []

    for sample in samples:
        main_sample.extend(sample.data)
        main_labels.extend([sample.params.class_label] * len(sample.data))

    combined = list(zip(main_sample, main_labels))
    shuffle(combined)

    main_sample, main_labels = zip(*combined)
    main_sample = np.array(main_sample)
    main_labels = np.array(main_labels)

    return main_sample, main_labels

