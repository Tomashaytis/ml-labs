import numpy as np

from matplotlib import pyplot as plt


class MaximinClusterizer:
    def __init__(self, n_clusters = np.inf, coef: float = 0.5, distance: str = 'euclidean', params: dict = None,
                 covariance: np.ndarray = None):
        self._n_clusters = n_clusters
        self._labels = np.array([])
        self._cluster_centers = np.array([])
        self._distance_type = distance
        self._coef = coef
        self._params = { 'p': 3 } if params is None else params
        self._covariance = covariance

    @property
    def trainable(self):
        return True

    @property
    def name(self):
        return "Maximin"

    @property
    def labels_(self):
        return self._labels

    @property
    def cluster_centers_(self):
        return self._cluster_centers

    def fit(self, sample: np.array, should_plot: bool = False):
        self._cluster_centers = []
        self._labels = np.zeros_like(sample)

        distances = []
        mean = np.mean(sample, axis=0)
        for x in sample:
            distances.append(self.find_distance(x, mean))
        index = distances.index(max(distances))
        self._cluster_centers.append(sample[index])

        distances = []
        for x in sample:
            distances.append(self.find_distance(x, self._cluster_centers[0]))
        index = distances.index(max(distances))
        self._cluster_centers.append(sample[index])

        d_maximin_values = []
        d_typical_values = []

        while True:
            min_distances = []
            for x in sample:
                distances = []
                for center in self._cluster_centers:
                    distances.append(self.find_distance(x, center))
                min_distances.append(min(distances))
                
            s = 0
            for i in range(len(self._cluster_centers)):
                for j in range(i + 1, len(self._cluster_centers)):
                    s += self.find_distance(self._cluster_centers[i], self._cluster_centers[j])

            l = len(self._cluster_centers)
            d_typical = self._coef * 2 / (l * (l - 1)) * s
            d_typical_values.append(d_typical)

            d_maximin = max(min_distances)
            d_maximin_values.append(d_maximin)
            new_center = sample[min_distances.index(d_maximin)]

            if d_maximin > d_typical and len(self._cluster_centers) < self._n_clusters:
                self._cluster_centers.append(new_center)
            else:
                break

        self._cluster_centers = np.array(self._cluster_centers)

        self._labels = np.array([self.predict(x) for x in sample])

        if should_plot:
            x_values = range(3, len(self._cluster_centers) + 2)
            plt.plot(x_values, d_maximin_values, label='d_maximin')
            plt.plot(x_values, d_typical_values, label='d_typical')
            plt.xlabel('Число кластеров')
            plt.ylabel('Расстояние')
            plt.title('Зависимость расстояния от числа кластеров')
            plt.legend()
            plt.xticks(x_values)
            plt.show()

    def predict(self, point: np.ndarray) -> int:
        distances = []
        for center in self._cluster_centers:
            distances.append(self.find_distance(point, center))
        return distances.index(min(distances))

    def find_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        if self._distance_type == 'euclidean':
            return float(np.linalg.norm(point1 - point2))

        elif self._distance_type == 'mahalanobis':
            if self._covariance is None:
                self._covariance = np.eye(len(point1))
            return float((point1 - point2).T @ self._covariance @ (point1 - point2))

        elif self._distance_type == 'takimoto':
            prod1 = point1.T @ point2
            prod2 = point1.T @ point1
            prod3 = point2.T @ point2
            return 1.0 - float(prod1 / (prod2 + prod3 - prod1))

        elif self._distance_type == 'manhattan':
            return np.sum(np.abs(point1 - point2))

        elif self._distance_type == 'chebyshev':
            return np.max(np.abs(point1 - point2))

        elif self._distance_type == 'minkowski':
            p = self._params.get('p', 2)
            diff = point1 - point2
            return np.sum(np.abs(diff) ** p) ** (1 / p)

        elif self._distance_type == 'cosine':
            norm1 = np.linalg.norm(point1)
            norm2 = np.linalg.norm(point2)

            if norm1 == 0 or norm2 == 0:
                return 1.0

            dot_product = np.dot(point1, point2)
            similarity = dot_product / (norm1 * norm2)

            similarity = np.clip(similarity, -1.0, 1.0)
            return 1 - similarity

        raise ValueError(f'Unknown distance type \'{self._distance_type}\'')

    def plot(self, sample: np.ndarray, true_labels: np.ndarray):
        predictions = np.array([self.predict(point) for point in sample])

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        unique_true_labels = np.unique(true_labels)
        colors_true = plt.cm.tab10(np.linspace(0, 1, len(unique_true_labels)))
        color_dict_true = {label: colors_true[i] for i, label in enumerate(unique_true_labels)}

        for label in unique_true_labels:
            mask = true_labels == label
            class_points = sample[mask]

            if len(class_points) > 0:
                axes[0].scatter(
                    class_points[:, 0],
                    class_points[:, 1],
                    marker='.',
                    color=color_dict_true[label],
                    label=f'Класс {label}'
                )

        axes[0].set_title('Исходное распределение классов')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        unique_pred_labels = np.unique(predictions)
        colors_pred = plt.cm.tab10(np.linspace(0, 1, len(unique_pred_labels)))
        color_dict_pred = {label: colors_pred[i] for i, label in enumerate(unique_pred_labels)}

        for label in unique_pred_labels:
            mask = predictions == label
            class_points = sample[mask]

            if len(class_points) > 0:
                axes[1].scatter(
                    class_points[:, 0],
                    class_points[:, 1],
                    marker='.',
                    color=color_dict_pred[label],
                    label=f'Кластер {label}'
                )

        centers = np.array(self._cluster_centers)
        axes[1].scatter(
            centers[:, 0],
            centers[:, 1],
            marker='o',
            s=80,
            facecolors='none',
            edgecolors='magenta',
            linewidth=2
        )

        axes[1].set_title('Максиминный алгоритм кластеризации')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
