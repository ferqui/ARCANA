import os
import torch
import pickle
import numpy as np

from tonic.dataset import Dataset
from scipy import ndimage, signal
from typing import Callable, Optional
from sklearn import preprocessing
from pathlib import Path

from scipy import signal, ndimage
from scipy.interpolate import interp1d


class RoshamboDataset(Dataset):
    """`EMG from forearm datasets for roshambo hand gestures recognition <https://zenodo.org/record/3194792#.Y1qqO3ZBxEY>`_

    ::

        @dataset{donati_elisa_2019_3194792,
            author = {Donati, Elisa},
            title = {EMG from forearm datasets for hand gestures recognition},
            month = may,
            year = 2019,
            publisher = {Zenodo},
            doi = {10.5281/zenodo.3194792},
            url = {https://doi.org/10.5281/zenodo.3194792}
        }

    Parameters:
        save_to (string): Location to save files to on disk.
        train (bool): If True, uses training subset, otherwise testing subset.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
        transforms (callable, optional): A callable of transforms that is applied to both data and
                                         labels at the same time.
        upsample (int): Upsample factor used to resample the EMG signal (default: 5 ~ 1KHz)
    Returns:
        A dataset object that can be indexed or iterated over. One sample returns a tuple of (sensor readings, targets).
    """

    def __init__(
        self,
        save_to: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        upsample: Optional[int] = 5,
    ):
        save_to = os.path.expanduser(save_to)
        super().__init__(
            save_to,
            transform=transform,
            target_transform=target_transform,
        )

        base_url = "https://zenodo.org/record/3194792/files/Roshambo.zip?download=1"

        self.upsample = upsample
        self.transform = transform
        self.target_transform = target_transform

        self.url = base_url
        self.filename = "Roshambo.zip"
        self.file_md5 = None

        if not self._check_exists():
            self.download()

            files = Path(os.path.join(self.location_on_system, "Roshambo")).glob(
                "*_ann.npy"
            )

            fs = 200  # 200 Hz sampling rate
            window = int(0.2 * fs)  # Time window of 200 ms
            X = []
            y = []
            for f in files:
                filename = str(f)
                labels = np.load(filename)
                filename = filename[:-7] + "emg.npy"
                data = np.load(filename)

                k_old = 0
                k_new = 0
                for i in range(0, len(labels) - 1):
                    if labels[i] != labels[i + 1]:
                        k_new = i

                        emg_singlerep = data[k_old:k_new]
                        n_iter = int(len(emg_singlerep) / window)
                        for k in range(0, n_iter + 1):
                            start = k * window - 1
                            stop = (k + 1) * window - 1
                            if k == 0:
                                start = 0
                                stop = 40

                            emg = emg_singlerep[start:stop]
                            if start < 399 and len(emg) == window:
                                X.append(emg)
                                y.append(labels[i])
                        k_old = k_new + 1

            y = np.array(y)

            le = preprocessing.LabelEncoder()
            y = le.fit_transform(y)

            self.data = {"X": X, "y": y}
            with open(
                os.path.join(self.location_on_system, "roshambo.pkl"), "wb"
            ) as file:
                pickle.dump(self.data, file)
        else:
            with open(
                os.path.join(self.location_on_system, "roshambo.pkl"), "rb"
            ) as file:
                self.data = pickle.load(file)

    def _check_exists(self):
        return os.path.isfile(os.path.join(self.location_on_system, "roshambo.pkl"))

    def __len__(self):
        return len(self.data["y"])

    def __getitem__(self, idx):
        sensor = self.data["X"][idx]
        sensor = signal.resample(
            sensor, sensor.shape[0] * self.upsample
        )  # Convert from 200Hz to 1KHz
        sensor = torch.tensor(sensor, dtype=torch.float32)

        label = torch.tensor(self.data["y"][idx])
        if self.transform:
            sensor = self.transform(sensor)
        if self.target_transform:
            label = self.target_transform(label)
        return sensor, label
