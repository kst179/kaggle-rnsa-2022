import multiprocessing as mp
import os

import nibabel as nib
import numpy as np
import pandas as pd
import tqdm

from utils.constants import root_dir
from utils.data_loaders import load_dcm_as_rsa_voxel_image


def minmax_normalize(image):
    min_val = image.min()
    max_val = image.max()
    return (image - min_val) / (max_val - min_val)


def dcm2nifti(image_uid, bbox):
    image, spacings, invert_z = load_dcm_as_rsa_voxel_image(image_uid)
    
    if bbox is not None:
        w, h, d = image.shape
        bbox["x"] = w - 1 - bbox["x"] - bbox["width"]
        bbox["y"] = h - 1 - bbox["y"] - bbox["height"]

        z = bbox["slice_number"]
        if invert_z:
            z = d - 1 - z
        
        bbox["z"] = z

    image = np.round(minmax_normalize(image) * (2**16 - 1)).astype(np.uint16)
    transform = np.eye(4)
    transform[range(3), range(3)] = spacings

    nii = nib.Nifti1Image(image, transform)
    nib.save(nii, root_dir / "train_images_nii" / f"{image_uid}.nii.gz")

    return bbox


if __name__ == "__main__":
    exclude = set(
        path.stem.removeprefix(".nii")
        for path in root_dir.glob("train_images_nii/*")
    )

    image_uids = list([
        path.name
        for path in root_dir.glob("train_images/*")
        if path.name not in exclude
    ])

    bbox_df = pd.read_csv(root_dir / "train_bounding_boxes.csv", index_col="StudyInstanceUID")
    unique_uids_with_bbox = np.unique(bbox_df)

    # image_uids = [image_uid for image_uid in image_uids if image_uid in bbox_df.index]

    new_bboxes = []

    with mp.Pool(processes=os.cpu_count()) as pool:
        handlers = []
        for image_uid in image_uids:
            if image_uid in bbox_df.index:
                bbox = bbox_df.loc[image_uid]
            else:
                bbox = None

            handler = pool.apply_async(dcm2nifti, (image_uid, bbox))
            handlers.append(handler)
        
        for handler in tqdm.tqdm(handlers):
            bbox = handler.get()
            if bbox is not None:
                bbox.to_csv(root_dir / "train_bounding_boxes_nii.csv", mode='a', header=False)
