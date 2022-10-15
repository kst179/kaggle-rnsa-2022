from audioop import minmax
from os import scandir
from pathlib import Path
from typing import List
import torch
import pandas as pd
import cv2
from utils.misc import minmax_normalize


class ImagesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: Path | str,
        labels_csv_path: Path | str,
        scan_uids: List[Path] | None = None,
        image_names: List[Path] | None = None,
        mean: float = 0.1309,
        std: float = 0.2285,
    ):
        self.root = Path(root)
        self.labels_csv_path = labels_csv_path
        self.scan_uids = scan_uids
        self.image_names = image_names
        self.mean = mean
        self.std = std

        self.labels = pd.read_csv(labels_csv_path).set_index("StudyInstanceUID")
        
        if self.scan_uids is None and self.image_names is None:
            self.scan_uids = list(set([
                path.stem[:-2] # remove _<vnum> suffix
                for path in self.root.glob("*_[1-7].png")
            ]))

        if self.image_names is None:
            pairs = set([
                path.stem 
                for path in self.root.glob("*_[1-7].png")
            ])
            self.image_names = [
                f"{scan_uid}_{vertebre}"
                for vertebre in range(1, 8)
                for scan_uid in self.scan_uids
                if f"{scan_uid}_{vertebre}" in pairs
            ]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, item):
        image_name = self.image_names[item]
        scan_uid, vertebre = image_name.split('_')
        vertebre = int(vertebre)
        label = self.labels.loc[scan_uid, f"C{vertebre}"]
        image = cv2.imread(str(self.root / f"{image_name}.png"), cv2.IMREAD_GRAYSCALE)
        image = minmax_normalize(image, inplace=False)

        image = (image - self.mean) / self.std

        image = torch.as_tensor(image[None, :, :], dtype=torch.float)

        return image, label, vertebre - 1