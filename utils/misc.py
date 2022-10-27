import numpy as np
import torch
from torch import Tensor
from typing import Tuple
from numpy.typing import NDArray

def minmax_normalize(image: Tensor, inplace=True) -> Tensor:
    min_val = image.min()
    max_val = image.max()
    
    if inplace:
        if isinstance(image, Tensor):
            image.sub_(min_val)
            image.div_(max_val - min_val)
        else:
            image -= min_val
            image /= max_val - min_val

        return image

    return (image - min_val) / (max_val - min_val)

def pad_to_grid(
    image: Tensor, 
    grid_size: int,
    mode: str = "constant",
) -> Tensor:
    """
    Pads 3d image to the shape where each dim size is divisible by grid_size
    
    example
    >>> image = torch.rand(3, 4, 5)
    >>> pad_multiple(image, 4).shape  
    (4, 4, 8)

    Args:
        image (Tensor): image to be padded
        grid_size (int): the desirable divisor of padded image shape
        mode (str): same as torch.functional.F.pad mode argument

    Returns:
        Tensor: padded image
    """
    pad_w, pad_h, pad_d = [
        s - (s + grid_size - 1) // grid_size * grid_size
        for s in image.shape[-3:]
    ]
    image = torch.functional.F.pad(image, (0, pad_d, 0, pad_h, 0, pad_w), mode=mode)

    return image

def pad_to(image, size):
    shape = image.shape[-3:]
    w, h, d = (max(size - shape[i], 0) for i in range(3))
    image = torch.functional.F.pad(image, (0, d, 0, h, 0, w))
    return image

def rescale(image, spacings, resolution, interpolation="trilinear"):
    image = torch.tensor(np.ascontiguousarray(image))[None, None, ...]
    size = np.array(image.shape[-3:])
    size = np.floor(size * spacings / resolution).astype(int)
    size = tuple(size)

    image = torch.functional.F.interpolate(image, size=size, mode=interpolation)
    return image.numpy()[0, 0]

def voxels_pca(voxels: Tensor) -> Tensor:
    _, _, V = torch.pca_lowrank(voxels, q=3, center=True)
    return V.T

def split_to_patches(
    image: Tensor, 
    size: Tuple[int],
) -> Tensor:

    if isinstance(size, int):
        size = tuple(size for _ in range(3))

    w, h, d = image.shape
    _w, _h, _d = (dim // s for dim, s in zip([w, h, d], size))
    
    return (
        image
        .reshape(_w, size[0], _h, size[1], _d, size[2])
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(-1, *size)
    )

class GaussianFilter:
    def __init__(
        self, 
        size: int = 11,
        sigma: Tuple[float, float, float] = (3, 3, 3),
    ):
        self.size = size
        self.sigma = sigma

        sigma = torch.tensor(sigma, dtype=torch.float)

        kernel = torch.stack(
            torch.meshgrid(
                *[torch.arange(size) for _ in range(3)], 
                indexing="ij",
            ), dim=-1
        ) - (size-1) /  2

        kernel = torch.exp(- 0.5 * (kernel**2 / sigma**2).sum(dim=-1)) / \
                 ((2 * torch.pi)**(-3/2) * sigma.prod()**0.5)

        kernel = kernel / kernel.sum()

        self.kernel = kernel

    def __call__(self, image: Tensor):
        kernel = self.kernel.to(image.device)

        image = torch.functional.F.conv3d(
            image[None, None], kernel[None, None], padding=self.size//2)[0, 0]

        return image
