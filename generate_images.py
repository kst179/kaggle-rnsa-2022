from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tqdm.notebook as tqdm
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

from datasets.vertabrea_dataset import VertebreaDataset
from utils.misc import minmax_normalize

if __name__ == "__main__":
    dataset = VertebreaDataset(
        "data/train_images/",
        "data/generated_segmentations/",
        "data/segmentation_meta.json",
        "data/train.csv",
        image_res_pix_per_mm=(4, 4, 4),
        device="cuda",
    )

    output_path = Path("./data/train_images2d")
    for batch in tqdm.tqdm(dataset.dataloader, smoothing=0.01):
        _, _, name, _ = batch
        name = name[0]

        for i, v_batch in enumerate(dataset.get_vertebrea(batch)):
            image, mask, mask_, label = v_batch
            mask1 = gaussian_filter(mask.cpu().numpy(), sigma=3)
            mask2 = gaussian_filter(mask.cpu().numpy(), sigma=5)
            smooth_mask = mask1 * (1 - mask2)

            image2d_axial = (image.cpu().numpy() * smooth_mask).sum(axis=2) / \
                            smooth_mask.sum(axis=2).clip(min=1e-8)
            image2d_axial = image2d_axial.T[::-1]
            minmax_normalize(image2d_axial)

            cv2.imwrite(str(output_path / "axial" / f"{name}_{i+1}.png"), 
                        (image2d_axial * 2**16).astype(np.uint16))

            image2d_coronal = (image.cpu().numpy() * smooth_mask).sum(axis=1) / \
                              smooth_mask.sum(axis=1).clip(min=1e-8)
            image2d_coronal = image2d_coronal.T[::-1]
            minmax_normalize(image2d_coronal)

            cv2.imwrite(str(output_path / "coronal" / f"{name}_{i+1}.png"), 
                        (image2d_coronal * 2**16).astype(np.uint16))

            image2d_sagittal = (image.cpu().numpy() * smooth_mask).sum(axis=0) / \
                               smooth_mask.sum(axis=0).clip(min=1e-8)
            image2d_sagittal = image2d_sagittal.T[::-1]
            minmax_normalize(image2d_sagittal)

            cv2.imwrite(str(output_path / "sagittal" / f"{name}_{i+1}.png"), 
                        (image2d_sagittal * 2**16).astype(np.uint16))