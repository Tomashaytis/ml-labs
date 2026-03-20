from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from src.generate import generate_2_samples, generate_3_samples, randomize_sample
from src.classifiers.bayesian import Bayesian
from src.classifiers.neighbour import KDEClassifier, KNNClassifier
from src.utils.sem1.lab3 import classifiers_stats
from src.utils.sem1.lab5 import complex_plot_nonlinear

TRAIN_SIZE = 200
TEST_SIZE = 200

CLASSIFIER = 'all'
BANDWIDTH = 'default'
KERNEL = 'gaussian'
KERNEL_CONSTRUCTION = 'gaussian'
N_NEIGHBOURS = 5
DISTANCE = 'euclidean'


if __name__ == '__main__':
    # Одинаковая матрица ковариации
    train_sample0, train_sample1 = generate_2_samples(TRAIN_SIZE)
    test_sample0, test_sample1 = generate_2_samples(TEST_SIZE)

    train_sample, train_labels = randomize_sample([train_sample0, train_sample1])
    test_sample, test_labels = randomize_sample([test_sample0, test_sample1])

    if CLASSIFIER == 'bayesian':
        classifier_bc = Bayesian([train_sample0.params, train_sample1.params])
        classifier_bc.plot(test_sample, test_labels)

    elif CLASSIFIER == 'kde':
        classifier_kde = KDEClassifier([train_sample0.params, train_sample1.params],
                                       BANDWIDTH, KERNEL, KERNEL_CONSTRUCTION)
        classifier_kde.fit(train_sample, train_labels)
        classifier_kde.plot(test_sample, test_labels)

    elif CLASSIFIER == 'knn':
        classifier_knn = KNNClassifier([train_sample0.params, train_sample1.params], N_NEIGHBOURS, DISTANCE)
        classifier_knn.fit(train_sample, train_labels)
        classifier_knn.plot(test_sample, test_labels)

    elif CLASSIFIER == 'knn-sklearn':
        classifier_knn = KNNClassifier([train_sample0.params, train_sample1.params], N_NEIGHBOURS, DISTANCE)
        classifier_knn.fit(train_sample, train_labels)

        classifier_sklearn_knn = KNeighborsClassifier(n_neighbors=N_NEIGHBOURS)
        classifier_sklearn_knn.fit(train_sample, train_labels)

        classifiers_kde = [classifier_knn, classifier_sklearn_knn]
        line_colors = ['pink', 'green']
        names = ['KNN', 'KNearestClassifier']

        complex_plot_nonlinear(classifiers_kde, line_colors, names, test_sample, test_sample0, test_sample1)

        for i in range(len(classifiers_kde)):
            complex_plot_nonlinear([classifiers_kde[i]], [line_colors[i]], [names[i]],
                                   test_sample, test_sample0, test_sample1, title=names[i])

    elif CLASSIFIER == 'knn-test':
        classifier_knn1 = KNNClassifier([train_sample0.params, train_sample1.params], 1, DISTANCE)
        classifier_knn1.fit(train_sample, train_labels)

        classifier_knn2 = KNNClassifier([train_sample0.params, train_sample1.params], 3, DISTANCE)
        classifier_knn2.fit(train_sample, train_labels)

        classifier_knn3 = KNNClassifier([train_sample0.params, train_sample1.params], 5, DISTANCE)
        classifier_knn3.fit(train_sample, train_labels)

        classifiers_kde = [classifier_knn1, classifier_knn2, classifier_knn3]
        line_colors = ['pink', 'green', 'cyan']
        names = ['KNN (K = 1)', 'KNN (K = 3)', 'KNN (K = 5)']

        classifiers_stats(classifiers_kde, names, test_sample, test_labels)
        complex_plot_nonlinear(classifiers_kde, line_colors, names, test_sample, test_sample0, test_sample1)

        for i in range(len(classifiers_kde)):
            complex_plot_nonlinear([classifiers_kde[i]], [line_colors[i]], [names[i]],
                                   test_sample, test_sample0, test_sample1, title=names[i])

    elif CLASSIFIER == 'all':
        classifier_bc = Bayesian([train_sample0.params, train_sample1.params])

        classifier_kde = KDEClassifier([train_sample0.params, train_sample1.params],
                                       BANDWIDTH, KERNEL, KERNEL_CONSTRUCTION)
        classifier_kde.fit(train_sample, train_labels)

        classifier_kde1 = KDEClassifier([train_sample0.params, train_sample1.params],
                                        BANDWIDTH, KERNEL, 'product')
        classifier_kde1.fit(train_sample, train_labels)

        classifier_knn = KNNClassifier([train_sample0.params, train_sample1.params], N_NEIGHBOURS, DISTANCE)
        classifier_knn.fit(train_sample, train_labels)

        classifiers_kde = [classifier_bc, classifier_kde, classifier_kde1, classifier_knn]
        line_colors = ['pink', 'green', 'cyan', 'red']
        names = ['Bayesian', 'KDE', 'KDE 1', 'KNN']

        classifiers_stats(classifiers_kde, names, test_sample, test_labels)
        complex_plot_nonlinear(classifiers_kde, line_colors, names, test_sample, test_sample0, test_sample1)

        for i in range(len(classifiers_kde)):
            complex_plot_nonlinear([classifiers_kde[i]], [line_colors[i]], [names[i]],
                                   test_sample, test_sample0, test_sample1, title=names[i])

    # Различная матрица ковариации
    train_sample0, train_sample1, _ = generate_3_samples(TRAIN_SIZE)
    test_sample0, test_sample1, _ = generate_3_samples(TEST_SIZE)

    train_sample, train_labels = randomize_sample([train_sample0, train_sample1])
    test_sample, test_labels = randomize_sample([test_sample0, test_sample1])

    if CLASSIFIER == 'bayesian':
        classifier_bc = Bayesian([train_sample0.params, train_sample1.params])
        classifier_bc.plot(test_sample, test_labels)

    elif CLASSIFIER == 'kde':
        classifier_kde = KDEClassifier([train_sample0.params, train_sample1.params],
                                       BANDWIDTH, KERNEL, KERNEL_CONSTRUCTION)
        classifier_kde.fit(train_sample, train_labels)
        classifier_kde.plot(test_sample, test_labels)

    elif CLASSIFIER == 'knn':
        classifier_knn = KNNClassifier([train_sample0.params, train_sample1.params], N_NEIGHBOURS, DISTANCE)
        classifier_knn.fit(train_sample, train_labels)
        classifier_knn.plot(test_sample, test_labels)

    elif CLASSIFIER == 'knn-sklearn':
        classifier_knn = KNNClassifier([train_sample0.params, train_sample1.params], N_NEIGHBOURS, DISTANCE)
        classifier_knn.fit(train_sample, train_labels)

        classifier_sklearn_knn = KNeighborsClassifier(n_neighbors=N_NEIGHBOURS)
        classifier_sklearn_knn.fit(train_sample, train_labels)

        classifiers_kde = [classifier_knn, classifier_sklearn_knn]
        line_colors = ['pink', 'green']
        names = ['KNN', 'KNearestClassifier']

        complex_plot_nonlinear(classifiers_kde, line_colors, names, test_sample, test_sample0, test_sample1)

        for i in range(len(classifiers_kde)):
            complex_plot_nonlinear([classifiers_kde[i]], [line_colors[i]], [names[i]],
                                   test_sample, test_sample0, test_sample1, title=names[i])

    elif CLASSIFIER == 'knn-test':
        classifier_knn1 = KNNClassifier([train_sample0.params, train_sample1.params], 1, DISTANCE)
        classifier_knn1.fit(train_sample, train_labels)

        classifier_knn2 = KNNClassifier([train_sample0.params, train_sample1.params], 3, DISTANCE)
        classifier_knn2.fit(train_sample, train_labels)

        classifier_knn3 = KNNClassifier([train_sample0.params, train_sample1.params], 5, DISTANCE)
        classifier_knn3.fit(train_sample, train_labels)

        classifiers_kde = [classifier_knn1, classifier_knn2, classifier_knn3]
        line_colors = ['pink', 'green', 'cyan']
        names = ['KNN (K = 1)', 'KNN (K = 3)', 'KNN (K = 5)']

        classifiers_stats(classifiers_kde, names, test_sample, test_labels)
        complex_plot_nonlinear(classifiers_kde, line_colors, names, test_sample, test_sample0, test_sample1)

        for i in range(len(classifiers_kde)):
            complex_plot_nonlinear([classifiers_kde[i]], [line_colors[i]], [names[i]],
                                   test_sample, test_sample0, test_sample1, title=names[i])

    elif CLASSIFIER == 'all':
        classifier_bc = Bayesian([train_sample0.params, train_sample1.params])

        classifier_kde = KDEClassifier([train_sample0.params, train_sample1.params],
                                       BANDWIDTH, KERNEL, KERNEL_CONSTRUCTION)
        classifier_kde.fit(train_sample, train_labels)

        classifier_kde1 = KDEClassifier([train_sample0.params, train_sample1.params],
                                        BANDWIDTH, KERNEL, 'product')
        classifier_kde1.fit(train_sample, train_labels)

        classifier_knn = KNNClassifier([train_sample0.params, train_sample1.params], N_NEIGHBOURS, DISTANCE)
        classifier_knn.fit(train_sample, train_labels)

        classifiers_kde = [classifier_bc, classifier_kde, classifier_kde1, classifier_knn]
        line_colors = ['pink', 'green', 'cyan', 'red']
        names = ['Bayesian', 'KDE', 'KDE 1', 'KNN']

        classifiers_stats(classifiers_kde, names, test_sample, test_labels)
        complex_plot_nonlinear(classifiers_kde, line_colors, names, test_sample, test_sample0, test_sample1)

        for i in range(len(classifiers_kde)):
            complex_plot_nonlinear([classifiers_kde[i]], [line_colors[i]], [names[i]],
                                   test_sample, test_sample0, test_sample1, title=names[i])

    if CLASSIFIER == 'kernels' or CLASSIFIER == 'kde-kernels' or CLASSIFIER == 'all':
        kernel_construction = ['product', 'radial']
        kernels = ['rectangular', 'triangular', 'gaussian', 'laplacian', 'cauchy', 'sinc']
        classifiers_kde = []
        names = []
        titles = []

        total_iterations = len(kernel_construction) * len(kernels)
        with tqdm(total=total_iterations, desc='Training KDE classifiers') as pbar:
            for kernel in kernels:
                for kc in kernel_construction:
                    classifier_kde = KDEClassifier([train_sample0.params, train_sample1.params],
                                                   kernel=kernel, kernel_construction=kc)
                    classifier_kde.fit(train_sample, train_labels)
                    classifiers_kde.append(classifier_kde)
                    names.append(f'KDE {kernel} ({kc})')
                    pbar.update(1)
                titles.append(f'KDE {kernel}')

        classifiers_stats(classifiers_kde, names, test_sample, test_labels)
        for i in range(0, len(classifiers_kde), 2):
            complex_plot_nonlinear([classifiers_kde[i], classifiers_kde[i + 1]], ['green', 'cyan'],
                                   [names[i], names[i + 1]], test_sample, test_sample0, test_sample1,
                                   title=titles[i // 2])

    if CLASSIFIER == 'kernels' or CLASSIFIER == 'knn-kernels' or CLASSIFIER == 'all':
        k_values = [1, 3, 5]
        distances = ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'cosine']
        classifiers_knn = []
        names = []
        titles = []

        total_iterations = len(k_values) * len(distances)
        with tqdm(total=total_iterations, desc='Training classifiers') as pbar:
            for distance in distances:
                classifiers_knn.append([])
                names.append([])
                for k in k_values:
                    classifier_knn = KNNClassifier([train_sample0.params, train_sample1.params],
                                                   n_neighbours=k, distance=distance)
                    classifier_knn.fit(train_sample, train_labels)
                    classifiers_knn[-1].append(classifier_knn)
                    names[-1].append(f'KNN {distance} (K = {k})')
                    pbar.update(1)

        line_colors = ['pink', 'green', 'cyan']

        classifiers_stats(classifiers_knn[0], names[0], test_sample, test_labels)
        classifiers_stats(classifiers_knn[1], names[1], test_sample, test_labels)
        classifiers_stats(classifiers_knn[2], names[2], test_sample, test_labels)
        classifiers_stats(classifiers_knn[3], names[3], test_sample, test_labels)
        classifiers_stats(classifiers_knn[4], names[4], test_sample, test_labels)

        for i in range(len(distances)):
            complex_plot_nonlinear(classifiers_knn[i], line_colors, names[i],
                                   test_sample, test_sample0, test_sample1,
                                   title=f'KNN {distances[i]}')


