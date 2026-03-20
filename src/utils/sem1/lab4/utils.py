import numpy as np
import matplotlib.pyplot as plt


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

    _ = plt.figure()

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