import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import tqdm
from utils.preprocessing import clahe3d, minmax_normalize, rescale
from utils.constants import external_data_root, segmentation_preprocessed_dir


if __name__ == "__main__":
    image_paths = list(external_data_root.glob("*/*/rawdata/*/*.nii.gz"))#[19:20]
    masks_paths = list(external_data_root.glob("*/*/derivatives/*/*.nii.gz"))#[19:20]
    except_paths = set([
        path.name
        for path in (segmentation_preprocessed_dir / "images3d").glob("*")
    ])

    for image_path, mask_path in zip(tqdm.tqdm(image_paths), masks_paths):
        scan_uid = image_path.parent.name
        
        image_file = nib.load(image_path)
        mask_file = nib.load(mask_path)

        image_file = nib.as_closest_canonical(image_file)
        mask_file = nib.as_closest_canonical(mask_file)

        image = image_file.get_fdata()
        mask = mask_file.get_fdata()

        image = minmax_normalize(image)

        orig_shape = image.shape

        spacings = image_file.affine.diagonal()[:-1]

        mask[mask > 7] = 0
        if mask.max() == 0:
            continue

        z_coords = np.where(mask > 0)[2]

        pad = int(100 / spacings[2])
        z_min = max(z_coords.min() - pad, 0)
        z_max = min(z_coords.max() + pad, image.shape[2])

        image = image[:, :, z_min:z_max]
        mask = mask[:, :, z_min:z_max]

        image = rescale(image, spacings, 1)
        mask = rescale(mask, spacings, 1, "nearest")

        image = clahe3d(image, size=32, clip_limit=2, n_bins=512)

        image = torch.tensor(np.ascontiguousarray(image)).float()
        mask = torch.tensor(np.ascontiguousarray(mask)).long()

        torch.save(image, segmentation_preprocessed_dir / "images3d" / scan_uid)
        torch.save(mask, segmentation_preprocessed_dir / "segmentations" / scan_uid)

        meta = json.loads((segmentation_preprocessed_dir / "metadata.json").read_text())
        meta[scan_uid] = {
            "original_resolution": list(orig_shape),
            "new_resolution": list(image.shape),
            "original_spacings": list(spacings),
            "new_spacings": 1,
        }
        (segmentation_preprocessed_dir / "metadata.json").write_text(json.dumps(meta))

    print("DONE!")