import numpy as np


class CTPreprocessor:
    def __init__(self):
        pass

    def to_hu(self, ct):
        volume = ct.volume.astype(np.float32)

        hu_volume = np.zeros_like(volume)

        for i, ds in enumerate(ct.datasets):
            slope = float(ds.RescaleSlope)
            intercept = float(ds.RescaleIntercept)

            hu_volume[i] = volume[i] * slope + intercept

        return hu_volume

    def clip_hu(self, hu_volume, min_hu=-1000, max_hu=400):
        return np.clip(hu_volume, min_hu, max_hu)

    def window(self, hu_volume, window_center=-600, window_width=1500):
        lower = window_center - window_width / 2
        upper = window_center + window_width / 2

        windowed = np.clip(hu_volume, lower, upper)

        return windowed
