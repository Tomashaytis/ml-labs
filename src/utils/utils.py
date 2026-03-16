import numpy as np
import matplotlib.pyplot as plt

from tabulate import tabulate


def standard(size: int = 1, dist_range: int = 100, accuracy: int = 20) -> float:
    rand = np.zeros(size)
    for i in range(accuracy):
        rand += np.random.uniform(-dist_range, dist_range, size)

    sigma = dist_range / np.sqrt(3)
    rand = rand / (sigma * np.sqrt(accuracy))
    return rand


def normal2(mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    a = np.zeros_like(covariance)

    a[0, 0] = np.sqrt(covariance[0, 0])
    a[1, 0] = covariance[0, 1] / a[0, 0]
    a[1, 1] = np.sqrt(covariance[1, 1] - covariance[0, 1] ** 2 / covariance[0, 0])

    e = standard(2)
    x = a @ e + mean

    return x


def calculate_covariance2(sample: np.ndarray):
    covariance2 = np.zeros((2, 2))
    covariance2[0, 0] = np.std(sample[:, 0]) ** 2
    covariance2[1, 1] = np.std(sample[:, 1]) ** 2
    covariance2[0, 1] = np.average((sample[:, 0] - np.average(sample[:, 0])) * (sample[:, 1] - np.average(sample[:, 1])))
    covariance2[1, 0] = covariance2[0, 1]
    return covariance2


def stat_estimates(sample: np.ndarray) -> tuple:
    mean = np.mean(sample)

    sample_copy = (sample - mean) @ (sample - mean).T
    covariance = np.mean(sample_copy)

    return mean, covariance


def mahalanobis_distance(mean1: np.ndarray, mean2: np.ndarray, covariance: np.ndarray):
    mean_diff = mean2 - mean1
    return mean_diff.T @ np.linalg.inv(covariance) @ mean_diff


def bhattacharyya_distance(mean1: np.ndarray, mean2: np.ndarray, covariance1: np.ndarray, covariance2: np.ndarray):
    mean_diff = mean2 - mean1
    avg_cov = (covariance1 + covariance2) / 2
    return (1 / 4 * mean_diff.T @ np.linalg.inv(avg_cov) @ mean_diff
            + 1 / 2 * np.log(np.linalg.det(avg_cov) / (np.sqrt(np.linalg.det(covariance1) * np.linalg.det(covariance2)))))


def estimate_sample_volume(error_prob: float, e: float = 0.05):
    return (1 - error_prob) / (error_prob * e ** 2)


def complex_plot(classifiers: list, colors: list, names: list, sample: np.ndarray, sample0, sample1, scaler = None, size: int = 1):
    if scaler is not None:
        sample_scaled = scaler.transform(sample)
        sample0_data_scaled = scaler.transform(sample0.data)
        sample1_data_scaled = scaler.transform(sample1.data)
        plot_sample = sample_scaled
        plot_sample0 = sample0_data_scaled
        plot_sample1 = sample1_data_scaled
    else:
        plot_sample = sample
        plot_sample0 = sample0.data
        plot_sample1 = sample1.data

    x_min, x_max = np.min(plot_sample[:, 0]) - size, np.max(plot_sample[:, 0]) + size
    y_min, y_max = np.min(plot_sample[:, 1]) - size, np.max(plot_sample[:, 1]) + size

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    for classifier, color in zip(classifiers, colors):
        h = 0.02
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        is_sklearn = 'sklearn' in str(type(classifier))

        if is_sklearn:
            grid_predictions = classifier.predict(grid_points)
        else:
            grid_predictions = np.array([classifier.predict(point) for point in grid_points])

        z = grid_predictions.reshape(xx.shape)
        plt.contour(xx, yy, z, colors=color, linewidths=0.5)

    plt.scatter(plot_sample0[:, 0], plot_sample0[:, 1], marker='.', color='red', label=sample0.params.class_name)
    plt.scatter(plot_sample1[:, 0], plot_sample1[:, 1], marker='.', color='blue', label=sample1.params.class_name)

    for i in range(len(classifiers)):
        plt.plot([0, 0], [0, 0], color=colors[i], linewidth=3, label=names[i])

    plt.title('Классификаторы')
    plt.legend()
    plt.show()


def complex_plot_linear(classifiers: list, colors: list, names: list, sample: np.ndarray, sample0, sample1, scaler=None,
                        show_margin: bool = False):
    names = names.copy()
    if scaler is not None:
        sample_scaled = scaler.transform(sample)
        sample0_data_scaled = scaler.transform(sample0.data)
        sample1_data_scaled = scaler.transform(sample1.data)
        plot_sample = sample_scaled
        plot_sample0 = sample0_data_scaled
        plot_sample1 = sample1_data_scaled
    else:
        plot_sample = sample
        plot_sample0 = sample0.data
        plot_sample1 = sample1.data

    x_min, x_max = np.min(plot_sample[:, 0]) - 1, np.max(plot_sample[:, 0]) + 1
    y_min, y_max = np.min(plot_sample[:, 1]) - 1, np.max(plot_sample[:, 1]) + 1

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    i = 0
    for classifier, color in zip(classifiers, colors):
        margin = None
        is_sklearn = 'sklearn' in str(type(classifier))
        if show_margin:
            margin = 1 / np.linalg.norm(classifier.coef_[0]) if names[i] == 'LinearSVC' else 1
            names[i] = f'{names[i]} (margin = {margin})'

        boundary = get_decision_boundary(classifier.coef_[0], classifier.intercept_[0], (x_min, x_max), margin)

        if show_margin and margin is not None:
            x, y_main, y_upper, y_lower = boundary
            plt.plot(x, y_main, color=color, linewidth=2, linestyle='--')
            plt.plot(x, y_upper, color=color, linewidth=2)
            plt.plot(x, y_lower, color=color, linewidth=2)
            plt.fill_between(x, y_upper, y_lower, color=color, alpha=0.1)
            if names[i].find('LinearSVC') == -1:
                plt.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1], marker='o', color=color, alpha=0.5)
        else:
            x, y = boundary
            plt.plot(x, y, color=color, linewidth=2)
        i += 1

    plt.scatter(plot_sample0[:, 0], plot_sample0[:, 1], marker='.', color='red', label=sample0.params.class_name)
    plt.scatter(plot_sample1[:, 0], plot_sample1[:, 1], marker='.', color='blue', label=sample1.params.class_name)

    for i in range(len(classifiers)):
        plt.plot([0, 0], [0, 0], color=colors[i], linewidth=3, label=names[i])

    plt.title('Классификаторы')
    plt.legend()
    plt.show()


def complex_plot_kernel(classifiers: list, colors: list, names: list, sample: np.ndarray, sample0, sample1, scaler=None,
                        size: int = 1, should_plot=True, title='Classifiers'):
    if scaler is not None:
        sample_scaled = scaler.transform(sample)
        sample0_data_scaled = scaler.transform(sample0.data)
        sample1_data_scaled = scaler.transform(sample1.data)
        plot_sample = sample_scaled
        plot_sample0 = sample0_data_scaled
        plot_sample1 = sample1_data_scaled
    else:
        plot_sample = sample
        plot_sample0 = sample0.data
        plot_sample1 = sample1.data

    fig = plt.figure()

    x_min, x_max = np.min(plot_sample[:, 0]) - size, np.max(plot_sample[:, 0]) + size
    y_min, y_max = np.min(plot_sample[:, 1]) - size, np.max(plot_sample[:, 1]) + size

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    for classifier, color, name in zip(classifiers, colors, names):
        h = 0.05
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        if 'sklearn' in str(type(classifier)):
            if hasattr(classifier, 'decision_function'):
                grid_decisions = classifier.decision_function(grid_points)
                z = grid_decisions.reshape(xx.shape)

                plt.contour(xx, yy, z, levels=[-1, 0, 1], colors=color,
                            linewidths=[1, 2, 1], linestyles=['--', '-', '--'])

        else:
            grid_decisions = np.array([classifier.decision_function(point) for point in grid_points])
            z = grid_decisions.reshape(xx.shape)

            plt.contour(xx, yy, z, levels=[-1, 0, 1], colors=color,
                        linewidths=[1, 2, 1], linestyles=['--', '-', '--'])

        support_vectors = get_support_vectors(classifier)
        if len(support_vectors) > 0:
            plt.scatter(support_vectors[:, 0], support_vectors[:, 1],
                        marker='o', s=80, facecolors='none', edgecolors=color, linewidths=2,
                        label=f'{name} support vectors')

    plt.scatter(plot_sample0[:, 0], plot_sample0[:, 1], marker='.', color='red',
                label=sample0.params.class_name, alpha=0.7)
    plt.scatter(plot_sample1[:, 0], plot_sample1[:, 1], marker='.', color='blue',
                label=sample1.params.class_name, alpha=0.7)

    plt.title(title)
    plt.legend()
    if should_plot:
        plt.show()


def complex_plot_nonlinear(classifiers: list, colors: list, names: list, sample: np.ndarray, sample0, sample1, scaler=None,
                        size: int = 1, title='Classifiers'):
    if scaler is not None:
        sample_scaled = scaler.transform(sample)
        sample0_data_scaled = scaler.transform(sample0.data)
        sample1_data_scaled = scaler.transform(sample1.data)
        plot_sample = sample_scaled
        plot_sample0 = sample0_data_scaled
        plot_sample1 = sample1_data_scaled
    else:
        plot_sample = sample
        plot_sample0 = sample0.data
        plot_sample1 = sample1.data

    fig = plt.figure()

    x_min, x_max = np.min(plot_sample[:, 0]) - size, np.max(plot_sample[:, 0]) + size
    y_min, y_max = np.min(plot_sample[:, 1]) - size, np.max(plot_sample[:, 1]) + size

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    for classifier, color, name in zip(classifiers, colors, names):
        h = 0.05
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        if hasattr(classifier, 'decision_function'):
            grid_decisions = np.array([classifier.decision_function(point) for point in grid_points])
            z = grid_decisions.reshape(xx.shape)
            plt.contour(xx, yy, z, levels=[0], colors=color,
                        linewidths=[2], linestyles=['-'])
        else:
            if 'sklearn' in str(type(classifier)):
                grid_predictions = classifier.predict(grid_points)
            else:
                grid_predictions = np.array([classifier.predict(point) for point in grid_points])
            z = grid_predictions.reshape(xx.shape)
            plt.contour(xx, yy, z, colors=color, linewidths=0.5)

    plt.scatter(plot_sample0[:, 0], plot_sample0[:, 1], marker='.', color='red',
                label=sample0.params.class_name, alpha=0.7)
    plt.scatter(plot_sample1[:, 0], plot_sample1[:, 1], marker='.', color='blue',
                label=sample1.params.class_name, alpha=0.7)

    for i in range(len(classifiers)):
        plt.plot([0, 0], [0, 0], color=colors[i], linewidth=3, label=names[i])

    plt.title(title)
    plt.legend()
    plt.show()


def classifiers_stats(classifiers: list, names: list, sample: np.ndarray, true_labels):
    data = []
    for name, classifier in zip(names, classifiers):
        errors = classifier.calculate_real_errors(sample, true_labels)
        data.append([name, errors[(0, 1)], errors[(1, 0)], classifier.calculate_risk(sample, true_labels)])

    headers = ['classifier', 'p0_error', 'p1_error', 'risk']

    print(tabulate(data, headers=headers, tablefmt="grid"))


def get_decision_boundary(weights: np.ndarray, threshold: float, x_limits: tuple, margin=None):
    if len(weights) != 2:
        return None

    w1, w2 = weights
    b = threshold
    x_min, x_max = x_limits

    x_plot = np.array([x_min, x_max])
    y_plot = (-w1 * x_plot - b) / w2

    if margin is None:
        return x_plot, y_plot

    y_plot_upper = (-w1 * x_plot - b + margin) / w2
    y_plot_lower = (-w1 * x_plot - b - margin) / w2

    return x_plot, y_plot, y_plot_upper, y_plot_lower

def get_support_vectors(classifier):
    if hasattr(classifier, 'support_vectors_'):
        return classifier.support_vectors_
    elif hasattr(classifier, '_support_vectors'):
        return classifier._support_vectors
    elif hasattr(classifier, 'support_vectors'):
        return classifier.support_vectors
    else:
        return np.array([])




