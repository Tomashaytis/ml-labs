import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC, LinearSVC

from src.generate import generate_3_samples, randomize_sample, COLORS
from src.classifiers.svm import SVMClassifier, KernelSVMClassifier
from src.utils import complex_plot, complex_plot_linear, classifiers_stats, complex_plot_kernel

SIZE = 200

CLASSIFIER = 'all'
C = 1
SHOW_MARGIN = True
KERNEL = 'polynomial-homogeneous'


if __name__ == '__main__':
    sample0, sample1, sample2 = generate_3_samples(SIZE)

    # Линейно разделимые классы
    main_sample, main_labels = randomize_sample([sample1, sample2])

    if CLASSIFIER == 'svm':
        classifier_svm = SVMClassifier(sample1.params, sample2.params)
        classifier_svm.fit(main_sample, main_labels)
        classifier_svm.plot(main_sample, main_labels)
        classifier_svm.plot_with_lines(main_sample, COLORS, main_labels)

    elif CLASSIFIER == 'svc':
        classifier_svc = SVC(kernel='linear', C=np.inf)
        classifier_svc.fit(main_sample, main_labels)
        complex_plot_linear([classifier_svc], ['red'], ['SVC'], main_sample, sample1, sample2)

    elif CLASSIFIER == 'lsvc':
        classifier_lsvc = LinearSVC()
        classifier_lsvc.fit(main_sample, main_labels)
        complex_plot_linear([classifier_lsvc], ['red'], ['LinearSVC'], main_sample, sample1, sample2)

    elif CLASSIFIER == 'all':
        classifier_svm = SVMClassifier(sample1.params, sample2.params)
        classifier_svm.fit(main_sample, main_labels)
        classifier_svc = SVC(kernel='linear', C=np.inf)
        classifier_svc.fit(main_sample, main_labels)
        classifier_lsvc = LinearSVC()
        classifier_lsvc.fit(main_sample, main_labels)

        classifiers = [classifier_svm, classifier_svc, classifier_lsvc]
        line_colors = ['pink', 'green', 'cyan']
        names = ['SVM', 'SVC', 'LinearSVC']

        complex_plot(classifiers, line_colors, names, main_sample, sample1, sample2)

        complex_plot_linear(classifiers, line_colors, names, main_sample, sample1, sample2, show_margin=SHOW_MARGIN)
        complex_plot_linear(classifiers[:1], line_colors[:1], names[:1], main_sample, sample1, sample2, show_margin=SHOW_MARGIN)
        complex_plot_linear(classifiers[1:2], line_colors[1:2], names[1:2], main_sample, sample1, sample2, show_margin=SHOW_MARGIN)
        complex_plot_linear(classifiers[2:3], line_colors[2:3], names[2:3], main_sample, sample1, sample2,
                            show_margin=SHOW_MARGIN)

    # Линейно неразделимые классы
    main_sample, main_labels = randomize_sample([sample0, sample1])

    if CLASSIFIER == 'svm':
        classifier_svm = SVMClassifier(sample0.params, sample1.params, C=C)
        classifier_svm.fit(main_sample, main_labels)
        classifier_svm.plot(main_sample, main_labels)
        classifier_svm.plot_with_lines(main_sample, COLORS, main_labels)

    elif CLASSIFIER == 'svc':
        classifier_svc = SVC(kernel='linear', C=C)
        classifier_svc.fit(main_sample, main_labels)
        complex_plot_linear([classifier_svc], ['red'], ['SVC'], main_sample, sample0, sample1)

    elif CLASSIFIER == 'lsvc':
        classifier_lsvc = LinearSVC(C=C)
        classifier_lsvc.fit(main_sample, main_labels)
        complex_plot_linear([classifier_lsvc], ['red'], ['LinearSVC'], main_sample, sample0, sample1)

    elif CLASSIFIER == 'all':
        classifier_svm1 = SVMClassifier(sample0.params, sample1.params, C=1 / 10)
        classifier_svm1.fit(main_sample, main_labels)
        classifier_svm2 = SVMClassifier(sample0.params, sample1.params, C=1)
        classifier_svm2.fit(main_sample, main_labels)
        classifier_svm3 = SVMClassifier(sample0.params, sample1.params, C=10)
        classifier_svm3.fit(main_sample, main_labels)
        classifier_svm_best = SVMClassifier(sample0.params, sample1.params, C=C)
        classifier_svm_best.fit(main_sample, main_labels)
        classifier_svc = SVC(kernel='linear', C=C)
        classifier_svc.fit(main_sample, main_labels)

        classifiers = [classifier_svm1, classifier_svm2, classifier_svm3, classifier_svm_best, classifier_svc]
        line_colors = ['pink', 'green', 'cyan', 'magenta', 'gray']
        names = ['SVM (C=0.1)', 'SVM (C=1)', 'SVM (C=10)', f'best SVM (C={C})', 'SVC']

        complex_plot(classifiers, line_colors, names, main_sample, sample0, sample1)

        classifiers_stats(classifiers[:4], names, main_sample, main_labels)

        complex_plot_linear(classifiers, line_colors, names, main_sample, sample0, sample1, show_margin=True)
        complex_plot_linear(classifiers[:1], line_colors[:1], names[:1], main_sample, sample0, sample1,
                            show_margin=SHOW_MARGIN)
        complex_plot_linear(classifiers[1:2], line_colors[1:2], names[1:2], main_sample, sample0, sample1,
                            show_margin=SHOW_MARGIN)
        complex_plot_linear(classifiers[2:3], line_colors[2:3], names[2:3], main_sample, sample0, sample1,
                            show_margin=SHOW_MARGIN)
        complex_plot_linear(classifiers[3:4], line_colors[3:4], names[3:4], main_sample, sample0, sample1,
                            show_margin=SHOW_MARGIN)

    # Ядра
    if CLASSIFIER == 'svm':
        classifier_svm = KernelSVMClassifier(sample0.params, sample1.params, C=C, kernel=KERNEL)
        classifier_svm.fit(main_sample, main_labels)
        classifier_svm.plot(main_sample, main_labels)
    elif CLASSIFIER == 'all' or CLASSIFIER == 'kernels':
        classifiers = []
        names = []
        svc_classifiers = []
        svc_names = []
        c_values = [0.1, 1, 10]
        kernels = ['polynomial-homogeneous', 'polynomial-inhomogeneous', 'rbf', 'gaussian-rbf', 'sigmoid']

        kernel_mapping = {
            'scalar': 'linear',
            'polynomial-homogeneous': 'poly',
            'polynomial-inhomogeneous': 'poly',
            'rbf': 'rbf',
            'gaussian-rbf': 'rbf',
            'sigmoid': 'sigmoid'
        }

        sklearn_params = {
            'polynomial-homogeneous': { 'degree': 2, 'coef0': 0 },
            'polynomial-inhomogeneous': { 'degree': 2, 'coef0': 1 },
            'rbf': { 'gamma': 1 },
            'gaussian-rbf': { 'gamma': 1 / 8 },
            'sigmoid': { 'gamma': 0.1, 'coef0': -1.0 }
        }

        total_iterations = len(c_values) * len(kernels) * 2
        with tqdm(total=total_iterations, desc='Training classifiers') as pbar:
            for kernel in kernels:
                classifiers.append([])
                names.append([])
                svc_classifiers.append([])
                svc_names.append([])
                for c in c_values:
                    classifier_svm = KernelSVMClassifier(sample0.params, sample1.params, C=c, kernel=kernel)
                    classifier_svm.fit(main_sample, main_labels)
                    classifiers[-1].append(classifier_svm)
                    names[-1].append(f'SVM {kernel} (C = {c})')
                    pbar.update(1)

                    sklearn_kernel = kernel_mapping[kernel]
                    sklearn_clf = SVC(C=c, kernel=sklearn_kernel, **sklearn_params.get(kernel, {}))
                    sklearn_clf.fit(main_sample, main_labels)
                    svc_classifiers[-1].append(sklearn_clf)
                    svc_names[-1].append(f'SVC {kernel} (C = {c})')
                    pbar.update(1)

        line_colors = ['pink', 'green', 'cyan', 'magenta', 'gray']

        classifiers_stats(classifiers[0], names[0], main_sample, main_labels)
        classifiers_stats(classifiers[1], names[1], main_sample, main_labels)
        classifiers_stats(classifiers[2], names[2], main_sample, main_labels)
        classifiers_stats(classifiers[3], names[3], main_sample, main_labels)
        classifiers_stats(classifiers[4], names[4], main_sample, main_labels)

        for i in range(len(kernels)):
            for j in range(len(c_values)):
                complex_plot_kernel([classifiers[i][j]], [line_colors[j]], [names[i][j]], main_sample, sample0, sample1,
                                    size=3,
                                    should_plot=False, title=f'Kernels SVM ({kernels[i]} C = {c_values[j]})')
                complex_plot_kernel([svc_classifiers[i][j]], [line_colors[j]], [svc_names[i][j]], main_sample, sample0,
                                    sample1,
                                    size=3,
                                    title=f'Kernels SVC ({kernels[i]} C = {c_values[j]})')
