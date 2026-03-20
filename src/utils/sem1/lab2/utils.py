import numpy as np
import matplotlib.pyplot as plt


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