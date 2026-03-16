import os
import json
import numpy as np

from src.core import Sample
from src.generate import generate_5_samples, randomize_sample
from src.classifiers.cluster import LeaderClusterizer, MaximinClusterizer, KMedoidsClusterizer, KMeansClusterizer

SIZE = 50
GENERATE = False
SAVE = False
CLUSTERIZER = 'kmeans'
DISTANCE = 'euclidean'
DATA_DIR = os.path.join('data', 'lab6')

THRESHOLD = 1
COEF = 0.5
N_CLUSTERS = 5
GOOD_CENTER_INDICES = np.array([1, 75, 36, 117, 200])
BAD_CENTER_INDICES = np.array([1, 2, 3, 4, 5])

if __name__ == '__main__':
    sample0, sample1, sample2, sample3, sample4 = None, None, None, None, None
    main_sample, main_labels = None, None

    if GENERATE:
        sample0, sample1, sample2, sample3, sample4 = generate_5_samples(SIZE, should_visualize=True)

        main_sample, main_labels = randomize_sample([sample0, sample1, sample2, sample3, sample4])

        if SAVE:
            with open(os.path.join(DATA_DIR, 'sample0.json'), 'w', encoding='utf-8') as fp:
                json.dump(sample0.to_json(), fp)
            with open(os.path.join(DATA_DIR, 'sample1.json'), 'w', encoding='utf-8') as fp:
                json.dump(sample1.to_json(), fp)
            with open(os.path.join(DATA_DIR, 'sample2.json'), 'w', encoding='utf-8') as fp:
                json.dump(sample2.to_json(), fp)
            with open(os.path.join(DATA_DIR, 'sample3.json'), 'w', encoding='utf-8') as fp:
                json.dump(sample3.to_json(), fp)
            with open(os.path.join(DATA_DIR, 'sample4.json'), 'w', encoding='utf-8') as fp:
                json.dump(sample4.to_json(), fp)
            with open(os.path.join(DATA_DIR, 'main_sample.json'), 'w', encoding='utf-8') as fp:
                json.dump(main_sample.tolist(), fp)
            with open(os.path.join(DATA_DIR, 'main_labels.json'), 'w', encoding='utf-8') as fp:
                json.dump(main_labels.tolist(), fp)
    else:
        with open(os.path.join(DATA_DIR, 'sample0.json'), 'r', encoding='utf-8') as fp:
            sample0 = Sample.from_json(json.load(fp))
        with open(os.path.join(DATA_DIR, 'sample1.json'), 'r', encoding='utf-8') as fp:
            sample1 = Sample.from_json(json.load(fp))
        with open(os.path.join(DATA_DIR, 'sample2.json'), 'r', encoding='utf-8') as fp:
            sample2 = Sample.from_json(json.load(fp))
        with open(os.path.join(DATA_DIR, 'sample3.json'), 'r', encoding='utf-8') as fp:
            sample3 = Sample.from_json(json.load(fp))
        with open(os.path.join(DATA_DIR, 'sample4.json'), 'r', encoding='utf-8') as fp:
            sample4 = Sample.from_json(json.load(fp))
        with open(os.path.join(DATA_DIR, 'main_sample.json'), 'r', encoding='utf-8') as fp:
            main_sample = np.array(json.load(fp))
        with open(os.path.join(DATA_DIR, 'main_labels.json'), 'r', encoding='utf-8') as fp:
            main_labels = np.array(json.load(fp))

    if CLUSTERIZER == 'leader':
        clusterizer_leader = LeaderClusterizer(THRESHOLD, DISTANCE, covariance=sample0.params.covariance)
        clusterizer_leader.fit(main_sample)
        clusterizer_leader.plot(main_sample, main_labels)

    if CLUSTERIZER == 'maximin':
        clusterizer_maximin = MaximinClusterizer(N_CLUSTERS, COEF, DISTANCE, covariance=sample0.params.covariance)
        clusterizer_maximin.fit(main_sample, should_plot=True)
        clusterizer_maximin.plot(main_sample, main_labels)

        for n_clusters in range(2, N_CLUSTERS + 1):
            clusterizer_maximin = MaximinClusterizer(n_clusters, COEF, DISTANCE, covariance=sample0.params.covariance)
            clusterizer_maximin.fit(main_sample)
            clusterizer_maximin.plot(main_sample, main_labels)

    if CLUSTERIZER == 'kmedoids':
        clusterizer_kmedoids = KMedoidsClusterizer(N_CLUSTERS, DISTANCE, covariance=sample0.params.covariance)
        clusterizer_kmedoids.fit(main_sample, should_plot=True)
        clusterizer_kmedoids.plot(main_sample, main_labels)

        clusterizer_kmedoids = KMedoidsClusterizer(3, DISTANCE, covariance=sample0.params.covariance)
        clusterizer_kmedoids.fit(main_sample, should_plot=True)
        clusterizer_kmedoids.plot(main_sample, main_labels)

        clusterizer_kmedoids = KMedoidsClusterizer(N_CLUSTERS, DISTANCE, covariance=sample0.params.covariance)
        clusterizer_kmedoids.fit(main_sample, GOOD_CENTER_INDICES, should_plot=True)
        clusterizer_kmedoids.plot(main_sample, main_labels)

        clusterizer_kmedoids = KMedoidsClusterizer(N_CLUSTERS, DISTANCE, covariance=sample0.params.covariance)
        clusterizer_kmedoids.fit(main_sample, BAD_CENTER_INDICES, should_plot=True)
        clusterizer_kmedoids.plot(main_sample, main_labels)

    if CLUSTERIZER == 'kmeans':
        clusterizer_kmeans = KMeansClusterizer(N_CLUSTERS, DISTANCE, covariance=sample0.params.covariance)
        clusterizer_kmeans.fit(main_sample, should_plot=True)
        clusterizer_kmeans.plot(main_sample, main_labels)

        clusterizer_kmeans = KMeansClusterizer(3, DISTANCE, covariance=sample0.params.covariance)
        clusterizer_kmeans.fit(main_sample, should_plot=True)
        clusterizer_kmeans.plot(main_sample, main_labels)

        clusterizer_kmeans = KMeansClusterizer(N_CLUSTERS, DISTANCE, covariance=sample0.params.covariance)
        clusterizer_kmeans.fit(main_sample, GOOD_CENTER_INDICES, should_plot=True)
        clusterizer_kmeans.plot(main_sample, main_labels)

        clusterizer_kmeans = KMeansClusterizer(N_CLUSTERS, DISTANCE, covariance=sample0.params.covariance)
        clusterizer_kmeans.fit(main_sample, BAD_CENTER_INDICES, should_plot=True)
        clusterizer_kmeans.plot(main_sample, main_labels)

