import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Union, List


class ImagePlotter:
    def __init__(self, cmap: str = 'viridis', heatmap_cmap: str = 'Blues'):
        self._cmap = cmap
        self._heatmap_cmap = heatmap_cmap

    @property
    def cmap(self):
        return self._cmap

    @cmap.setter
    def cmap(self, value):
        self._cmap = value

    @property
    def heatmap_cmap(self):
        return self._heatmap_cmap

    @heatmap_cmap.setter
    def heatmap_cmap(self, value):
        self._heatmap_cmap = value

    @staticmethod
    def plot_one(data: Union[List, np.ndarray], title: str,
                 xlabel: str = '', ylabel: str = '', label: str = ''):
        """Отображает один график"""
        plt.figure(figsize=(8, 6))
        plt.plot(data, label=label if label else None)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if label:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    @staticmethod
    def plot_two(subtitle: str,
                 data1: Union[List, np.ndarray], title1: str,
                 data2: Union[List, np.ndarray], title2: str,
                 xlabel: str = '', ylabel: str = ''):
        """Отображает два графика рядом"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(subtitle)

        ax1.plot(data1)
        ax1.set_title(title1)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.grid(True, alpha=0.3)

        ax2.plot(data2)
        ax2.set_title(title2)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_three(subtitle: str,
                   data1: Union[List, np.ndarray], title1: str,
                   data2: Union[List, np.ndarray], title2: str,
                   data3: Union[List, np.ndarray], title3: str,
                   xlabel: str = '', ylabel: str = ''):
        """Отображает три графика в ряд"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(subtitle)

        ax1.plot(data1)
        ax1.set_title(title1)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.grid(True, alpha=0.3)

        ax2.plot(data2)
        ax2.set_title(title2)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        ax2.grid(True, alpha=0.3)

        ax3.plot(data3)
        ax3.set_title(title3)
        ax3.set_xlabel(xlabel)
        ax3.set_ylabel(ylabel)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_four(subtitle: str,
                  data1: Union[List, np.ndarray], title1: str,
                  data2: Union[List, np.ndarray], title2: str,
                  data3: Union[List, np.ndarray], title3: str,
                  data4: Union[List, np.ndarray], title4: str,
                  xlabel: str = '', ylabel: str = ''):
        """Отображает четыре графика в сетке 2x2"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
        fig.suptitle(subtitle)

        ax1.plot(data1)
        ax1.set_title(title1)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.grid(True, alpha=0.3)

        ax2.plot(data2)
        ax2.set_title(title2)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        ax2.grid(True, alpha=0.3)

        ax3.plot(data3)
        ax3.set_title(title3)
        ax3.set_xlabel(xlabel)
        ax3.set_ylabel(ylabel)
        ax3.grid(True, alpha=0.3)

        ax4.plot(data4)
        ax4.set_title(title4)
        ax4.set_xlabel(xlabel)
        ax4.set_ylabel(ylabel)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, cm: np.ndarray, class_names: list):
        """Отображает Confusion Matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap=self._heatmap_cmap,
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

    def imshow_one(self, img: np.ndarray, title: str):
        """Отображает одно изображение"""
        plt.figure(figsize=(8, 6))
        plt.imshow(img, cmap=self._cmap)
        plt.title(title)
        plt.axis('off')
        plt.show()

    def imshow_two(self, subtitle: str,
                   img1: np.ndarray, title1: str,
                   img2: np.ndarray, title2: str):
        """Отображает два изображения рядом"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(subtitle)

        ax1.imshow(img1, cmap=self._cmap)
        ax1.set_title(title1)
        ax1.axis('off')

        ax2.imshow(img2, cmap=self._cmap)
        ax2.set_title(title2)
        ax2.axis('off')

        plt.tight_layout()
        plt.show()

    def imshow_three(self, subtitle: str,
                     img1: np.ndarray, title1: str,
                     img2: np.ndarray, title2: str,
                     img3: np.ndarray, title3: str):
        """Отображает три изображения в ряд"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(subtitle)

        ax1.imshow(img1, cmap=self._cmap)
        ax1.set_title(title1)
        ax1.axis('off')

        ax2.imshow(img2, cmap=self._cmap)
        ax2.set_title(title2)
        ax2.axis('off')

        ax3.imshow(img3, cmap=self._cmap)
        ax3.set_title(title3)
        ax3.axis('off')

        plt.tight_layout()
        plt.show()

    def imshow_four(self, subtitle: str,
                    img1: np.ndarray, title1: str,
                    img2: np.ndarray, title2: str,
                    img3: np.ndarray, title3: str,
                    img4: np.ndarray, title4: str):
        """Отображает четыре изображения в ряд"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
        fig.suptitle(subtitle)

        ax1.imshow(img1, cmap=self._cmap)
        ax1.set_title(title1)
        ax1.axis('off')

        ax2.imshow(img2, cmap=self._cmap)
        ax2.set_title(title2)
        ax2.axis('off')

        ax3.imshow(img3, cmap=self._cmap)
        ax3.set_title(title3)
        ax3.axis('off')

        ax4.imshow(img4, cmap=self._cmap)
        ax4.set_title(title4)
        ax4.axis('off')

        plt.tight_layout()
        plt.show()

    def imshow_mnist(self, sample: np.ndarray, true_labels: np.array, title: str = "Digits of mnist dataset"):
        """Отображет цифры из датасета mnist"""
        if len(np.unique(true_labels)) < 10:
            raise ValueError('Not all digits are in the sample')

        digits = {}
        while len(digits) < 10:
            index = np.random.randint(len(sample))
            digit = true_labels[index]
            if digit not in digits:
                digits[digit] = sample[index]

        rows = 4
        cols = 3

        fig, axes = plt.subplots(rows, cols, figsize=(4, 5))
        fig.suptitle(title)

        for i in range(rows):
            for j in range(cols):
                if i < cols:
                    digit = i * cols + j + 1
                    axes[i, j].imshow(digits[digit], cmap=self._cmap)
                    axes[i, j].set_title(digit)
                    axes[i, j].axis('off')
                else:
                    axes[i, j].axis('off')

        axes[cols, 1].imshow(digits[0], cmap=self._cmap)
        axes[cols, 1].set_title(0)
        axes[cols, 1].axis('off')

        plt.tight_layout()
        plt.show()

    def imshow_cifer10(self, sample: np.ndarray, true_labels: np.array, class_names: list,
                       title: str = "Classes of cifar10 dataset"):
        """Отображает классы из датасета cifar10"""
        if len(np.unique(true_labels)) < 10:
            raise ValueError('Not all class labels are in the sample')

        class_images = {i: [] for i in range(10)}

        for class_label in range(10):
            indices = np.where(true_labels == class_label)[0]
            selected_indices = np.random.choice(indices, 4, replace=False)
            class_images[class_label] = list(sample[selected_indices])

        rows = 2
        cols = 5
        padding = 2

        fig, axes = plt.subplots(rows, cols, figsize=(10, 5))
        fig.suptitle(title)

        for i in range(rows):
            for j in range(cols):
                class_label = i * cols + j
                images = class_images[class_label]

                top_row = np.concatenate([images[0],
                                          np.ones((images[0].shape[0], padding, 3), dtype=np.uint8) * 255,
                                          images[1]], axis=1)
                bottom_row = np.concatenate([images[2],
                                             np.ones((images[2].shape[0], padding, 3), dtype=np.uint8) * 255,
                                             images[3]], axis=1)
                grid = np.concatenate([top_row,
                                       np.ones((padding, top_row.shape[1], 3), dtype=np.uint8) * 255,
                                       bottom_row], axis=0)

                axes[i, j].imshow(grid, cmap=self._cmap)
                axes[i, j].set_title(class_names[class_label])
                axes[i, j].axis('off')

        plt.tight_layout()
        plt.show()


