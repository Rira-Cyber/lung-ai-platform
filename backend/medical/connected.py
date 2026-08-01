import numpy as np

from skimage.measure import label, regionprops


class ConnectedComponents:
    @staticmethod
    def largest(mask):
        labels = label(mask)

        regions = regionprops(labels)

        if len(regions) == 0:
            return np.zeros_like(mask)

        largest = max(regions, key=lambda r: r.area)

        return labels == largest.label

    @staticmethod
    def largest_n(mask, n=2):
        labels = label(mask)

        regions = sorted(regionprops(labels), key=lambda r: r.area, reverse=True)

        result = np.zeros_like(mask)

        for region in regions[:n]:
            result[labels == region.label] = True

        return result
