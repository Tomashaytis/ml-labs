import matplotlib.pyplot as plt
import numpy as np

from tabulate import tabulate

from src.generate import generate_2_samples, generate_3_samples, randomize_sample, COLORS
from src.classifiers.bayesian import Bayesian
from src.classifiers.linear import Fisher, STD, RobbinsMonro
from src.utils.sem1.lab2 import complex_plot
from src.utils.sem1.lab3 import classifiers_stats

CLASSIFIER = 'robins-monro'
IS_MIN_STD = False
B = 0.75
ROBBINS_MONRO_CHECK = True
EPSILON = 0.000000001


if __name__ == '__main__':
    sample0, sample1 = generate_2_samples()
    main_sample, main_labels = randomize_sample([sample0, sample1])

    if CLASSIFIER == 'bayesian':
        classifier_bc = Bayesian([sample0.params, sample1.params], is_equal_covariance=True)
        classifier_bc.plot(main_sample, main_labels)
        classifier_bc.plot_with_lines(main_sample, COLORS, main_labels)
    elif CLASSIFIER == 'fisher':
        classifier_f = Fisher(sample0.params, sample1.params, is_equal_covariance=True)
        classifier_f.plot(main_sample, main_labels)
        classifier_f.plot_with_lines(main_sample, COLORS, main_labels)
    elif CLASSIFIER == 'std':
        classifier_std = STD(sample0.params, sample1.params)
        classifier_std.fit(main_sample, main_labels)
        classifier_std.plot(main_sample, main_labels)
        classifier_std.plot_with_lines(main_sample, COLORS, main_labels)
    elif CLASSIFIER == 'robins-monro':
        classifier_rm = RobbinsMonro(sample0.params, sample1.params,
                                               is_min_std=IS_MIN_STD, b=B)
        classifier_rm.fit(main_sample, main_labels, epsilon=EPSILON, should_plot=True)
        classifier_rm.plot(main_sample, main_labels)
        classifier_rm.plot_with_lines(main_sample, COLORS, main_labels)
        plt.show()
    elif CLASSIFIER == 'all':
        classifier_bc = Bayesian([sample0.params, sample1.params], is_equal_covariance=True)
        classifier_f = Fisher(sample0.params, sample1.params, is_equal_covariance=True)
        classifier_std = STD(sample0.params, sample1.params)
        classifier_std.fit(main_sample, main_labels)
        classifier_rm = RobbinsMonro(sample0.params, sample1.params,
                                               is_min_std=IS_MIN_STD, b=B)
        classifier_rm.fit(main_sample, main_labels, epsilon=EPSILON)

        classifiers = [classifier_bc, classifier_f, classifier_std, classifier_rm]
        line_colors = ['red', 'green', 'cyan', 'magenta']
        names = ['Bayesian', 'Fisher', 'STD', 'Robbins-Monro']

        print('Classifier stats (equal covariance)')
        classifiers_stats(classifiers, names, main_sample, main_labels)
        classifier_bc.plot_with_lines(main_sample, COLORS, main_labels)
        complex_plot(classifiers, line_colors, names, main_sample, sample0, sample1)
        print()

    sample0, sample1, _ = generate_3_samples()
    main_sample, main_labels = randomize_sample([sample0, sample1])

    if CLASSIFIER == 'bayesian':
        classifier_bc = Bayesian([sample0.params, sample1.params], is_equal_covariance=False)
        classifier_bc.plot(main_sample, main_labels)
        classifier_bc.plot_with_lines(main_sample, COLORS, main_labels)
    elif CLASSIFIER == 'fisher':
        classifier_f = Fisher(sample0.params, sample1.params, is_equal_covariance=False)
        classifier_f.plot(main_sample, main_labels)
        classifier_f.plot_with_lines(main_sample, COLORS, main_labels)
    elif CLASSIFIER == 'std':
        classifier_std = STD(sample0.params, sample1.params)
        classifier_std.fit(main_sample, main_labels)
        classifier_std.plot(main_sample, main_labels)
        classifier_std.plot_with_lines(main_sample, COLORS, main_labels)
    elif CLASSIFIER == 'robins-monro':
        classifier_rm = RobbinsMonro(sample0.params, sample1.params,
                                               is_min_std=IS_MIN_STD, b=B)
        classifier_rm.fit(main_sample, main_labels)
        classifier_rm.plot(main_sample, main_labels)
        classifier_rm.plot_with_lines(main_sample, COLORS, main_labels)
    elif CLASSIFIER == 'all':
        classifier_bc = Bayesian([sample0.params, sample1.params], is_equal_covariance=False)
        classifier_f = Fisher(sample0.params, sample1.params, is_equal_covariance=False)
        classifier_std = STD(sample0.params, sample1.params)
        classifier_std.fit(main_sample, main_labels)
        classifier_rm = RobbinsMonro(sample0.params, sample1.params,
                                               is_min_std=IS_MIN_STD, b=B)
        classifier_rm.fit(main_sample, main_labels)

        classifiers = [classifier_bc, classifier_f, classifier_std, classifier_rm]
        line_colors = ['red', 'green', 'cyan', 'magenta']
        names = ['Bayesian', 'Fisher', 'STD', 'Robbins-Monro']

        print('Classifier stats (different covariance)')
        classifiers_stats(classifiers, names, main_sample, main_labels)
        complex_plot(classifiers, line_colors, names, main_sample, sample0, sample1)
        print()

    if ROBBINS_MONRO_CHECK:
        classifier_rm = RobbinsMonro(sample0.params, sample1.params,
                                               is_min_std=IS_MIN_STD, b=B)

        risk_stats = []
        for i in range(len(main_sample)):
            classifier_rm.fit_one(main_sample[i], int(main_labels[i]))
            risk_stats.append(classifier_rm.calculate_risk(main_sample, main_labels))

        plt.plot(range(len(main_sample)), risk_stats)
        plt.xlabel('Iterations')
        plt.ylabel('Risk')
        plt.title('Robbins-Monro classifier')
        plt.show()

        start_weights = [
            [-3, -3, -3],
            [-2, -2, -2],
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [1, 7, 8],
            [-5, 0, -6],
            [-3, 9, 9],
            [5, 7, -3],
            [-5, 6, -6],
            [-9, -9, 9],
        ]

        headers = list(np.linspace(0.5, 1, 10))
        data = []
        for weights in start_weights:
            data.append([weights])
            for b in np.linspace(0.5, 1, 10):
                classifier_rm = RobbinsMonro(sample0.params, sample1.params, is_min_std=IS_MIN_STD,
                                             b=b, start_weights=weights)
                classifier_rm.fit(main_sample, main_labels, epsilon=EPSILON)
                data[-1].append(1 - classifier_rm.calculate_risk(main_sample, main_labels))

        print('Зависимость точности классификации от параметров классификатора Роббинса-Монро')
        print(tabulate(data, headers=headers, tablefmt="grid"))
        print()

        headers = list(np.linspace(0.5, 1, 10))
        data = []
        for weights in start_weights:
            data.append([weights])
            for b in np.linspace(0.5, 1, 10):
                classifier_rm = RobbinsMonro(sample0.params, sample1.params, is_min_std=IS_MIN_STD,
                                             b=b, start_weights=weights)
                k = classifier_rm.fit(main_sample, main_labels, epsilon=EPSILON)
                data[-1].append(k)

        print('Зависимость числа итераций обучения от параметров классификатора Роббинса-Монро (мин. риск - 0.05)')
        print(tabulate(data, headers=headers, tablefmt="grid"))
        print()

