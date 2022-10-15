from configparser import Interpolation
from sys import orig_argv
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List
import numpy as np
from scipy.ndimage import gaussian_filter

from preprocessing.segmentation_weights import calculate_weights
from utils.affine3d import center_rotation
from utils.misc import pad_to
import gc

class IterativeSegDataset(Dataset):
    def __init__(
            self,
            root: Path,
            image_names: List[str],
            train: bool,
            size: int = 128,
            gamma: float = 8.0,
            sigma: float = 4.0,
        ):

        self.root = root
        self.image_names = image_names
        self.train = train
        self.size = size
        self.gamma = gamma
        self.sigma = sigma

        self.images = [
            root / "images3d" / image_name
            for image_name in image_names
        ]

        self.segmentation_masks = [
            root / "segmentations" / image_name
            for image_name in image_names
        ]
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = torch.load(self.images[index])
        seg_mask = torch.load(self.segmentation_masks[index])
        
        if self.train:
            # random rotations
            if np.random.rand() < 0.75:
                yaw = np.random.uniform(-10, 10)
                pitch = np.random.uniform(-35, 35)
                roll = np.random.uniform(-5, 5)

                # clear cuda cache before using it for 3d affine rotations
                gc.collect()
                torch.cuda.empty_cache()

                image = center_rotation(image, x=pitch, y=roll, z=yaw, 
                                        order="zyx", degrees=True, interpolation="linear", device="cuda")
                seg_mask = center_rotation(seg_mask, x=pitch, y=roll, z=yaw, 
                                           order="zyx", degrees=True, interpolation="nearest", device="cuda")

            presented_vertebrae = np.unique(seg_mask)[1:]
            
            if np.random.rand() < 0.20 or len(presented_vertebrae) == 0:
                # random chunk from image
                x, y, z = np.random.randint(np.maximum(np.array(image.shape) - self.size, 1))
                label = 0

            else:
                # select target vertabre
                label = np.random.choice(presented_vertebrae)
                vertebre_voxels = (seg_mask == label).nonzero()
                min_point, _ = vertebre_voxels.min(axis=0)
                max_point, _ = vertebre_voxels.max(axis=0)

                if np.random.rand() < 0.70:
                    # should contain entire vertabre

                    min_point, max_point = (
                        torch.maximum(max_point - self.size, torch.zeros_like(min_point)),
                        min_point,
                    )
                else:
                    # may partially overlap vertebre, but cover its center
                    center = vertebre_voxels.float().mean(axis=0)

                    min_point = torch.maximum(center - self.size, torch.zeros_like(min_point))
                    max_point = center
                
                max_point = torch.maximum(max_point, min_point)

                x, y, z = (np.random.randint(min_point[i], max_point[i] + 1) for i in range(3))

            orig_seg_mask = seg_mask

            image = image[x:x+self.size, y:y+self.size, z:z+self.size]
            seg_mask = seg_mask[x:x+self.size, y:y+self.size, z:z+self.size]

            if torch.any(seg_mask == 7):
                patch_labels = seg_mask.unique()
                patch_labels = patch_labels[patch_labels > 0]
                label = np.random.choice(patch_labels)

            image = pad_to(image, self.size)
            seg_mask = pad_to(seg_mask, self.size)

            if label == 0:
                instance_mask = seg_mask > 0
                binary_mask = torch.zeros_like(image)
                completness = 0
            else:
                instance_mask = ((0 < seg_mask) & (seg_mask < label)).float()
                binary_mask = (seg_mask == label).long()
                completness = int(binary_mask.sum() == (orig_seg_mask == label).sum())
                
            # 50% random left-right flip augmentation
            if np.random.rand() < 0.5:
                image = torch.flip(image, (0,))
                instance_mask = torch.flip(instance_mask, (0,))
                binary_mask = torch.flip(binary_mask, (0,))

            # select one of gaussian noise or blur
            if np.random.rand() < 0.50:
                sigma = np.random.uniform(0.0, 0.07)
                image += np.random.randn(*image.shape) * sigma
            else:
                sigma = np.random.uniform(0.0, 0.5)
                image.numpy()[...] = gaussian_filter(image.numpy(), sigma=sigma)

            weights = calculate_weights(binary_mask.numpy().astype(np.bool), self.gamma, self.sigma)
            weights = torch.tensor(weights, dtype=torch.float)

            return (
                image.unsqueeze(0).float(),
                instance_mask.unsqueeze(0).float(), 
                binary_mask.unsqueeze(0).float(),
                weights.unsqueeze(0),
                label,
                completness
            )

        return image.unsqueeze(0), seg_mask
