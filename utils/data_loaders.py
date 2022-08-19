import nibabel as nib
import numpy as np
import pydicom as dcm
from utils.constants import root_dir


def load_dcm_scans(scan_uid):
    scan_files = root_dir.glob(f"train_images/{scan_uid}/*.dcm")
    scan_files = sorted(scan_files, key=lambda path: int(path.stem))
    
    scans = [
        dcm.dcmread(scan_file) 
        for scan_file in scan_files
    ]

    return scans


def load_segmentation_mask(scan_uid):
    segmentation_file = nib.load(root_dir / "segmentations" / f"{scan_uid}.nii")
    segmentation_file = nib.as_closest_canonical(segmentation_file)
    segmentation_array = np.array(segmentation_file.get_fdata())
    return segmentation_array


def scans_to_voxel_image(scans):
    height, width = scans[0].pixel_array.shape
    depth = len(scans)

    image = np.empty((height, width, depth))
    for i, scan in enumerate(scans):
        image[:, :, i] = scan.pixel_array.T[::-1, ::-1]

    _, _, z0 = scans[0].get("ImagePositionPatient")
    _, _, z1 = scans[1].get("ImagePositionPatient")

    z_spacing = z1 - z0
    y_spacing, x_spacing = map(float, scans[0].get("PixelSpacing"))

    if z_spacing < 0:
        z_spacing = -z_spacing
        image = image[:, :, ::-1]

    spacings = np.array([x_spacing, y_spacing, z_spacing])

    return image, spacings


def load_dcm_as_rsa_voxel_image(image_uid):
    scans = load_dcm_scans(image_uid)
    image, spacings = scans_to_voxel_image(scans)
    return image, spacings
