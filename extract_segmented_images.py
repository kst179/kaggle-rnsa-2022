import json

import numpy as np
import torch
import tqdm
from utils.constants import root_dir, segmentation_preprocessed_dir
from utils.data_loaders import load_dcm_as_rsa_voxel_image, load_segmentation_mask
from utils.preprocessing import clahe3d, minmax_normalize, rescale


if __name__ == "__main__":
    segmented_scan_uids = [
        path.name.replace('.nii', '')
        for path in root_dir.glob("segmentations/*.nii")
    ]

    for scan_uid in tqdm.tqdm(segmented_scan_uids):
        image, spacings = load_dcm_as_rsa_voxel_image(scan_uid)
        mask = load_segmentation_mask(scan_uid)

        image = minmax_normalize(image)
        mask[mask > 7] = 0

        orig_shape = image.shape

        image = rescale(image, spacings, resolution=1, interpolation="trilinear")
        mask = rescale(mask, spacings, resolution=1, interpolation="nearest")

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
