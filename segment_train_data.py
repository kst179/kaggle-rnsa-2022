import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from datasets.nii_dataset import NiiDataset
from models.unet3d import Unet3DIterative
from postprocessing.segmentation import segmentation
from preprocessing.clahe3d import clahe3d
from utils.affine3d import resize
from utils.misc import minmax_normalize
from utils.visualization import visualize

input_dir    = Path("./data/train_images")
output_dir   = Path("./data/generated_segmentations")
weights_path = Path("./weights/iterative_segmentation_best_weights_8446iou.pth")
meta_path    = Path("./data/segmentation_meta.json")

clahe_params = dict(
    size=32,
    clip_limit=2,
    n_bins=512
)

seg_params = dict(
    size=96,
    min_voxels=2500,
    filter_mask=True,
    closing_radius=0,
)

def preprocess(image, affine):
    image = torch.as_tensor(image, dtype=torch.float, device="cuda")
    image = minmax_normalize(image)
    image = resize(image, affine.diagonal()[:3], device="cuda")
    image = clahe3d(image, **clahe_params, device="cuda")

    return image

def extract_segmentation_meta(segmentation_meta, name, mask):
    # name = seg_path.name.removesuffix(".nii")
    # mask = nib.load(seg_path).get_fdata().astype(np.int16)
    for v in range(1, 8):
        b_mask = mask == v
        voxels = np.stack(np.where(b_mask), axis=-1)

        if len(voxels) >= 3:
            pca = PCA(n_components=3).fit(voxels)
            components = pca.components_

            inversed_axis = components[range(3), range(3)] < 0
            components[inversed_axis] *= -1

        segmentation_meta.append({
            "uid": name,
            "vertebre": v,
            "presented": len(voxels) > 0,
            "num_voxels": len(voxels),
            "center": voxels.mean(axis=0).tolist() 
                if len(voxels) > 0 else None,
            "bbox": [*voxels.min(axis=0), *voxels.max(axis=0)] 
                if len(voxels) > 0 else None,
            "bbox_center": (voxels.min(axis=0) + voxels.max(axis=0)) / 2 
                if len(voxels > 0) else None,
            "pca_components": components.tolist()
                if len(voxels) >= 3 else None,
        })


if __name__ == "__main__":
    dataset = NiiDataset(input_dir, preprocess=preprocess, return_name=True)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda x, y: (x[0], y[0]), num_workers=os.cpu_count())
    model = Unet3DIterative().cuda()

    weights = torch.load(weights_path)
    model.load_state_dict(weights)

    if not meta_path.exists():
        segmentation_meta = []
    else:
        raise ValueError("AAA you are rewriting segmentation metadata file!")

    for image, name in tqdm.tqdm(dataset):
        mask = segmentation(model, image, **seg_params)
        mask = mask.astype(np.uint8)
        # visualize(image, mask)

        extract_segmentation_meta(segmentation_meta, name, mask)

        mask_nii = nib.Nifti1Image(mask, np.eye(4))
        nib.save(mask_nii, output_dir / f"{name}.nii.gz")

    segmentation_meta = pd.DataFrame(segmentation_meta)
    segmentation_meta.to_json(meta_path, orient="records")
