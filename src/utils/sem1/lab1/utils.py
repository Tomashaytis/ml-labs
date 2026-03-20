import numpy as np


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