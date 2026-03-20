import numpy as np
import matplotlib.pyplot as plt


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