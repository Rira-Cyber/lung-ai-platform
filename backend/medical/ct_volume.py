from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pydicom
class CTVolume:

    def __init__(self , dicom_folder):

        self.dicom_folder = Path(dicom_folder)

        self.datasets = None
        self.volume = None

        self.pixel_spacing = None
        self.slice_thickness = None
        self.load()

    def load(self):

        dicom_files = list(self.dicom_folder.glob("*.dcm"))

        if len(dicom_files) == 0:
            raise FileNotFoundError('No DICOM files found.')
        
        datasets = [
            pydicom.dcmread(
                f,
                stop_before_pixels=True
            )
            for f in dicom_files
        ]

        datasets.sort(
            key=lambda ds: float(ds.ImagePositionPatient[2])
        )

        slices = []

        for ds in datasets:
            full_ds = pydicom.dcmread(ds.filename)
            slices.append(full_ds.pixel_array)
        self.datasets = datasets
        self.volume = np.stack(slices , axis=0)
        self.pixel_spacing = datasets[0].PixelSpacing
        self.slice_thickness = float(
            datasets[0].SliceThickness
        )
    def __repr__(self):

        return (
            "CTVolume\n"
            "-------------------------\n"
            f"Slices          : {self.volume.shape[0]}\n"
            f"Rows            : {self.volume.shape[1]}\n"
            f"Columns         : {self.volume.shape[2]}\n"
            f"Pixel Spacing   : {self.pixel_spacing}\n"
            f"Slice Thickness : {self.slice_thickness} mm"
        )

    def shape(self):
        return self.volume.shape
    
    def axial(self , index):
        return self.volume[index]
    
    def coronal(self , index):
        return self.volume[: , index , :]
    
    def sagittal(self , index):
        return self.volume[: , : , index]
    def to_hu(self):

        slope = float(self.datasets[0].RescaleSlope)
        intercept = float(self.datasets[0].RescaleIntercept)

        hu = self.volume.astype(np.float32)

        hu = hu * slope + intercept

        return hu
    def window(self, level=-600, width=1500):

        hu = self.to_hu()

        low = level - width / 2
        high = level + width / 2

        hu = np.clip(hu, low, high)

        hu = (hu - low) / (high - low)

        return hu
    def show(
        self,
        index,
        hu=False
    ):

        if hu:
            img = self.to_hu()[index]
        else:
            img = self.volume[index]

        plt.figure(figsize=(6,6))

        plt.imshow(
            img,
            cmap="gray"
        )

        plt.axis("off")

        plt.show()