import os
import torch
import pickle
import numpy as np

from tonic.dataset import Dataset
from scipy import ndimage, signal
from typing import Callable, Optional
from sklearn import preprocessing
from pathlib import Path


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
    Returns:
        A dataset object that can be indexed or iterated over. One sample returns a tuple of (sensor readings, targets).
    """

    def __init__(
        self,
        save_to: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(
            save_to,
            transform=transform,
            target_transform=target_transform,
        )

        base_url = "https://zenodo.org/record/3194792/files/Roshambo.zip?download=1"

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

            X = []
            y = []
            for f in files:
                filename = str(f)
                labels = np.load(filename)
                filename = filename[:-7] + "emg.npy"
                data = np.load(filename)

                indices = [0]
                previous_val = labels[0]
                for i, value in enumerate(labels):
                    if value != previous_val:
                        X.append(data[indices[-1] : i, :])
                        y.append(previous_val)
                        indices.append(i)
                    previous_val = value

            X = np.array(X, dtype=object)
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
        sensor = signal.resample(sensor, sensor.shape[0] * 5)
        sensor = torch.tensor(sensor)

        label = torch.tensor(self.data["y"][idx])
        if self.transform:
            sensor = self.transform(sensor)
        if self.target_transform:
            label = self.target_transform(label)
        return sensor, label
