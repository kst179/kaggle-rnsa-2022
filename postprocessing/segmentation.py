import numpy as np
import torch
from numpy.typing import NDArray

from models.unet3d import Unet3DIterative
from utils.misc import pad_to

from skimage.morphology import binary_closing, ball


import cc3d

def morphological_closing(mask, radius=1):
    structure = ball(radius, bool)
    mask = binary_closing(mask, structure)

    return mask

def filter_segmentation_mask(mask: NDArray[np.int32]) -> NDArray[np.int32]:
    components, N = cc3d.connected_components(mask, connectivity=6, return_N=True)
    
    vertebrae = [(0, None) for _ in range(7)]

    for component_label in range(N):
        cc_mask = (components == component_label)
        num_voxels = cc_mask.sum()
        label = mask[cc_mask].max()

        if label == 0:
            continue

        if vertebrae[label-1][0] < num_voxels:
            vertebrae[label-1] = (num_voxels, cc_mask)

    filtered_mask = np.zeros_like(mask)
    for i, (_, bmask) in enumerate(vertebrae, start=1):
        filtered_mask[bmask] = i

    return filtered_mask


def filter_binary_mask(mask: NDArray[np.bool]) -> NDArray[np.bool]:
    components_mask = cc3d.connected_components(mask.astype(int) + 1, connectivity=6)

    zeros = []
    ones = []

    # print(components_mask.max())

    for _, component_mask in cc3d.each(components_mask, binary=True):
        label = mask[component_mask].max()
        # print(label)

        if label:
            ones.append((component_mask.sum(), component_mask))
        else:
            zeros.append((component_mask.sum(), component_mask))

    ones = sorted(ones, key=lambda x: -x[0])
    zeros = sorted(zeros, key=lambda x: -x[0])

    # print(next(zip(*ones)))
    # print(next(zip(*zeros)))

    for _, component_mask in ones[1:]:
        mask[component_mask] = 0
    
    for _, component_mask in zeros[1:]:
        mask[component_mask] = 1

    return mask


def pick_patch(image, instance_mask, anchor=None, center=None, size=96):
    w, h, d = image.shape[-3:]

    if anchor is None and center is not None:
        anchor = torch.round(center - size / 2).long()
    elif anchor is None:
        raise ValueError("Should pass one of center or anchor parameters")

    if not isinstance(anchor, torch.Tensor):
        anchor = torch.tensor(anchor, dtype=torch.long, device='cpu')

    anchor = anchor.clamp(
        min=torch.zeros_like(anchor),
        max=torch.tensor([w, h, d]) - size
    )

    # anchor = torch.minimum(anchor, max=torch.tensor([w, h, d]) - size)
    # anchor = torch.maximum(anchor, torch.zeros_like(anchor))

    x1, y1, z1 = anchor
    x2, y2, z2 = anchor + size

    image_patch = pad_to(image[..., x1:x2, y1:y2, z1:z2], size)
    instance_mask_patch = pad_to(instance_mask[..., x1:x2, y1:y2, z1:z2], size)

    return image_patch, instance_mask_patch, anchor

@torch.no_grad()
def segmentation(
    model: Unet3DIterative,
    image: NDArray[np.float64],
    size: int = 96,
    device: str = 'cuda',
    min_voxels_partial: int = 1000,
    min_voxels: int = 2500,
    max_iter: int = 10,
    delta_dist: float = 2.0,
    closing_radius: int = 0,
    filter_mask: bool = False,
) -> NDArray[np.int32]:
    w, h, d = image.shape

    # add batch&channel move to gpu if needed
    image = torch.as_tensor(image[None, None], dtype=torch.float32, device=device)
    instance_mask = torch.zeros_like(image, dtype=torch.bool, device=device)
    segmentation_mask = torch.zeros_like(image, dtype=torch.uint8, device=device)        

    model.to(device)
    model.eval()

    c1_found = False

    # z is inverted because we want to move scanning chunk from up to down
    z_iter = np.linspace(0, d-size, (d + size - 1) // size, dtype=int)[::-1]
    x_iter = np.linspace(0, w-size, (w + size - 1) // size, dtype=int)
    y_iter = np.linspace(0, h-size, (h + size - 1) // size, dtype=int)

    grid = map(np.ravel, np.meshgrid(z_iter, x_iter, y_iter, indexing="ij"))

    for z, x, y in zip(*grid):
        anchor = torch.tensor([x, y, z], dtype=torch.long)

        image_patch, instance_mask_patch, _ = pick_patch(
            image, instance_mask, anchor=anchor, size=size
        )

        output, _, _ = model(image_patch, instance_mask_patch)
        mask = output[0, 0] > 0
        
        vertebre_voxels = mask.nonzero()

        if len(vertebre_voxels) > min_voxels_partial:
            center = vertebre_voxels.float().mean(dim=0).cpu() + anchor

            image_patch, instance_mask_patch, anchor = pick_patch(
                image, instance_mask, center=center, size=size
            )
            output, _, _ = model(image_patch, instance_mask_patch)
            mask = output[0, 0] > 0

            vertebre_voxels = mask.nonzero()
            
            if len(vertebre_voxels) > min_voxels:
                center = vertebre_voxels.float().mean(dim=0).cpu() + anchor
                c1_found = True

                break

    if not c1_found:
        return segmentation_mask[0, 0].cpu().numpy()

    vertebra = 1
    prev_center = center

    while vertebra < 8:
        for iter in range(max_iter):
            image_patch, instance_mask_patch, anchor = pick_patch(
                image, instance_mask, center=center, size=size
            )

            output, _, _ = model(image_patch, instance_mask_patch)
            mask = output[0, 0] > 0

            if closing_radius or filter_mask:
                mask = mask.cpu().numpy()
                if closing_radius:
                    mask = morphological_closing(mask, closing_radius)
                if filter_mask:
                    mask = filter_binary_mask(mask)
                mask = torch.as_tensor(mask, device=device)
            
            prev_center = center

            vertebre_voxels = mask.nonzero() + anchor.to(device)
            if len(vertebre_voxels) == 0:
                return segmentation_mask[0, 0].cpu().numpy()

            center = vertebre_voxels.float().mean(axis=0).cpu()

            if (iter == max_iter - 1 or 
                (len(vertebre_voxels) > min_voxels and 
                 np.linalg.norm(center - prev_center) < delta_dist)):

                segmentation_mask[
                    ...,
                    vertebre_voxels[:, 0],
                    vertebre_voxels[:, 1],
                    vertebre_voxels[:, 2],
                ] = vertebra

                instance_mask[
                    ...,
                    vertebre_voxels[:, 0],
                    vertebre_voxels[:, 1],
                    vertebre_voxels[:, 2],
                ] = 1

                vertebra += 1
                break


    return segmentation_mask[0, 0].cpu().numpy()