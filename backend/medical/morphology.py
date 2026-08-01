import numpy as np
from scipy import ndimage


class Morphology:
    @staticmethod
    def close(mask, kernel=5):
        return ndimage.binary_closing(mask, structure=np.ones((kernel, kernel)))

    @staticmethod
    def open(mask, kernel=3):
        return ndimage.binary_opening(mask, structure=np.ones((kernel, kernel)))

    @staticmethod
    def fill(mask):
        return ndimage.binary_fill_holes(mask)
