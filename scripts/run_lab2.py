import numpy as np

from src.generate import generate_2_samples, generate_3_samples, randomize_sample, COLORS
from src.classifiers.bayesian import BayesianClassifier, MinimaxClassifier, NeymanPearsonClassifier
from src.utils import estimate_sample_volume, complex_plot

CLASSIFIER = 'all'
IS_EQUAL_COVARIANCE = True
MAX_ERROR_PROB = 0.05
EPSILON = 0.05


if __name__ == '__main__':
    sample1, sample2 = generate_2_samples()
    main_sample, main_labels = randomize_sample([sample1, sample2])

    if CLASSIFIER == 'bayesian':
        classifier_bc = BayesianClassifier([sample1.params, sample2.params], IS_EQUAL_COVARIANCE)
        classifier_bc.plot(main_sample, main_labels)
        classifier_bc.plot_with_lines(main_sample, COLORS, main_labels)

        print('Bayesian errors')
        if IS_EQUAL_COVARIANCE:
            print('theory')
            classifier_bc.calculate_theoretical_errors(True)
        print('real')
        classifier_bc.calculate_real_errors(main_sample, main_labels, True)
        print()

    elif CLASSIFIER == 'minimax':
        classifier_mc = MinimaxClassifier(sample1.params, sample2.params, IS_EQUAL_COVARIANCE)
        classifier_mc.plot(main_sample, main_labels)
        classifier_mc.plot_with_lines(main_sample, COLORS, main_labels)

        print('Minimax errors')
        if IS_EQUAL_COVARIANCE:
            print('theory')
            classifier_mc.calculate_theoretical_errors(True)
        print('real')
        classifier_mc.calculate_real_errors(main_sample, main_labels, True)
        print()

    elif CLASSIFIER == 'neyman-pearson':
        classifier_npc = NeymanPearsonClassifier(sample1.params, sample2.params, MAX_ERROR_PROB, IS_EQUAL_COVARIANCE)
        classifier_npc.plot(main_sample, main_labels)
        classifier_npc.plot(main_sample, main_labels)
        classifier_npc.plot_with_lines(main_sample, COLORS, main_labels)

        print('Neyman-Pearson errors')
        if IS_EQUAL_COVARIANCE:
            print('theory')
            classifier_npc.calculate_theoretical_errors(True)
        print('real')
        classifier_npc.calculate_real_errors(main_sample, main_labels, True)
        print()

    elif CLASSIFIER == 'all':
        classifier_bc = BayesianClassifier([sample1.params, sample2.params], IS_EQUAL_COVARIANCE)
        classifier_mc = MinimaxClassifier(sample1.params, sample2.params, IS_EQUAL_COVARIANCE)
        classifier_npc = NeymanPearsonClassifier(sample1.params, sample2.params, MAX_ERROR_PROB, IS_EQUAL_COVARIANCE)

        print('Bayesian errors')
        if IS_EQUAL_COVARIANCE:
            print('theory')
            classifier_bc.calculate_theoretical_errors(True)
        print('real')
        classifier_bc.calculate_real_errors(main_sample, main_labels, True)
        print()

        print('Minimax errors')
        if IS_EQUAL_COVARIANCE:
            print('theory')
            classifier_mc.calculate_theoretical_errors(True)
        print('real')
        classifier_mc.calculate_real_errors(main_sample, main_labels, True)
        print()

        print('Neyman-Pearson errors')
        if IS_EQUAL_COVARIANCE:
            print('theory')
            classifier_npc.calculate_theoretical_errors(True)
        print('real')
        classifier_npc.calculate_real_errors(main_sample, main_labels, True)
        print()

        complex_plot([classifier_bc, classifier_mc, classifier_npc], ['red', 'green', 'yellow'], ['Bayesian', 'Minimax', 'Neyman-Pearson'], main_sample, sample1, sample2)

    sample1, sample2, sample3 = generate_3_samples()
    main_sample, main_labels = randomize_sample([sample1, sample2, sample3])

    classifier_bc = BayesianClassifier([sample1.params, sample2.params, sample3.params])
    classifier_bc.plot(main_sample, main_labels)
    classifier_bc.plot_with_lines(main_sample, COLORS, main_labels)
    print('Bayesian errors')
    print('real')
    errors = classifier_bc.calculate_real_errors(main_sample, main_labels, True)
    print()

    p1 = (errors[(1, 2)][0] + errors[(1, 2)][1]) / 2
    p2 = (errors[(1, 3)][0] + errors[(1, 3)][1]) / 2
    p3 = (errors[(2, 3)][0] + errors[(2, 3)][1]) / 2

    p = float(np.mean(np.array([p1, p2, p3])))

    print(f'Объём выборки, обеспечивающий погрешность {EPSILON}: {estimate_sample_volume(p, EPSILON)}')
