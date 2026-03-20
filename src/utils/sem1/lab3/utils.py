import numpy as np

from tabulate import tabulate


def classifiers_stats(classifiers: list, names: list, sample: np.ndarray, true_labels):
    data = []
    for name, classifier in zip(names, classifiers):
        errors = classifier.calculate_real_errors(sample, true_labels)
        data.append([name, errors[(0, 1)], errors[(1, 0)], classifier.calculate_risk(sample, true_labels)])

    headers = ['classifier', 'p0_error', 'p1_error', 'risk']

    print(tabulate(data, headers=headers, tablefmt="grid"))