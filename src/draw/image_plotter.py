import numpy as np
import matplotlib.pyplot as plt


class ImagePlotter:
    def __init__(self, cmap: str = 'viridis'):
        self._cmap = cmap

    @property
    def cmap(self):
        return self._cmap

    @cmap.setter
    def cmap(self, value):
        self._cmap = value

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

    def imshow_mnist(self, sample: np.ndarray, true_labels: np.array):
        """Отображет все цифры из датасета mnist"""
        if len(np.unique(true_labels)) < 10:
            raise ValueError('Not all digits are in the sample')

        digits = {}
        while len(digits) < 10:
            index = np.random.randint(len(sample))
            digit = true_labels[index]
            if digit not in digits:
                digits[digit] = sample[index]

        fig, axes = plt.subplots(4, 3, figsize=(4, 5))
        fig.suptitle("Цифры датасета mnist")

        for i in range(4):
            for j in range(3):
                if i < 3:
                    digit = i * 3 + j + 1
                    axes[i, j].imshow(digits[digit], cmap=self._cmap)
                    axes[i, j].set_title(digit)
                    axes[i, j].axis('off')
                else:
                    axes[i, j].axis('off')

        axes[3, 1].imshow(digits[0], cmap=self._cmap)
        axes[3, 1].set_title(0)
        axes[3, 1].axis('off')

        plt.tight_layout()
        plt.show()


