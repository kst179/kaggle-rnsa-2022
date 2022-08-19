import os
from pathlib import Path

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset

from typing import List


class SegmentationDataset(Dataset):
    def __init__(self, root: Path, image_names: List[str], train: bool, crop_size=128):
        self.root = root
        self.image_names = image_names

        self.images = [
            root / "images3d" / image_name
            for image_name in image_names
        ]

        self.segmentation_masks = [
            root / "segmentations" / image_name
            for image_name in image_names
        ]
        
        self.train = train
        self.crop_size = crop_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = torch.load(self.images[index])
        seg_mask = torch.load(self.segmentation_masks[index])
        
        if self.train:            
            x, y, z = np.random.randint(np.maximum(np.array(image.shape) - self.crop_size, 1))

            image = image[x:x+self.crop_size, y:y+self.crop_size, z:z+self.crop_size]
            seg_mask = seg_mask[x:x+self.crop_size, y:y+self.crop_size, z:z+self.crop_size]
            
            pad = []

            w, h, d = image.shape
            pw = max(0, self.crop_size - w)
            ph = max(0, self.crop_size - h)
            pd = max(0, self.crop_size - d)

            pad = (0, pd, 0, ph, 0, pw)

            image = torch.functional.F.pad(image, pad)
            seg_mask = torch.functional.F.pad(seg_mask, pad)

        return image.unsqueeze(0), seg_mask


if __name__ == "__main__":
    root = Path("./preprocessed_data")
    
    image_names = [
        path.name
        for path in root.glob("images3d/*")
    ]

    dataset = SegmentationDataset(root, image_names, train=True)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=os.cpu_count(), pin_memory=True)
    image, seg_mask = next(iter(dataloader))
    print(image.shape, seg_mask.shape)

    for batch in tqdm.tqdm(dataloader):
        pass