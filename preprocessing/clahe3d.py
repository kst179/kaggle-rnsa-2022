from operator import is_
from pydoc import cli
from time import time
from typing import Literal
import numpy as np
from numba import cuda, njit
from numpy.typing import NDArray
from torch import Tensor
import torch

from utils.misc import pad_to_grid


def clahe3d(
    image: NDArray | Tensor,
    size: int = 8, 
    clip_limit: float = 5.0, 
    n_bins: int = 256,
    device: Literal["auto", "cpu", "cuda"] = "auto",
):
    is_tensor = isinstance(image, Tensor)
    data_device = "cpu" if not is_tensor else image.device.type

    if device == "auto":
        device = data_device

    if device == "cpu" and is_tensor:
        image = image.cpu().numpy()

    if device == "cpu":
        output = __clahe3d_cpu(image, size, clip_limit, n_bins)
        if is_tensor:
            output = torch.as_tensor(output, device=data_device)

        return output

    return __clahe3d_gpu(image, size, clip_limit, n_bins)


@njit
def __clahe3d_cpu(image: np.ndarray, size: int = 8, clip_limit=5, n_bins=256):
    w, h, d = image.shape

    new_w = (w + size - 1) // size * size
    new_h = (h + size - 1) // size * size
    new_d = (d + size - 1) // size * size

    # reflect padding
    image = np.concatenate((
        image[size-1::-1],
        image,
        image[:-(new_w - w + size)-1:-1],
    ), axis=0)

    image = np.concatenate((
        image[:, size-1::-1],
        image,
        image[:, :-(new_h - h + size)-1:-1],
    ), axis=1)
    
    image = np.concatenate((
        image[:, :, size-1::-1],
        image,
        image[:, :, :-(new_d - d + size)-1:-1],
    ), axis=2)

    image_quantized = image * (n_bins - 1)

    mapings = np.empty((new_w // size + 2, new_h // size + 2, new_d // size + 2, n_bins), dtype=np.float64)

    for x in range(new_w // size + 2):
        for y in range(new_h // size + 2):
            for z in range(new_d // size + 2):
                block = image_quantized[
                    x * size:(x + 1) * size,
                    y * size:(y + 1) * size,
                    z * size:(z + 1) * size,
                ]

                hist_i, _ = np.histogram(block.flatten(), n_bins, [0, n_bins])
                hist = hist_i.astype(np.float64).clip(0, clip_limit)
                cdf = np.cumsum(hist)
                denom = cdf.max() - cdf.min()
                if denom == 0:
                    mapings[x, y, z] = 0
                else:
                    mapings[x, y, z, :] = (cdf - cdf.min()) / denom

    out_image = np.empty((w, h, d))

    for x in range(w):
        for y in range(h):
            for z in range(d):
                i = (x + size // 2) // size
                j = (y + size // 2) // size
                k = (z + size // 2) // size

                dx = x - i * size + size / 2 + 0.5
                dy = y - j * size + size / 2 + 0.5
                dz = z - k * size + size / 2 + 0.5

                ix = size - dx
                iy = size - dy
                iz = size - dz

                v = int(image_quantized[x+size, y+size, z+size])

                out_image[x, y, z] = (
                    mapings[  i,   j,   k, v] * ix * iy * iz +
                    mapings[i+1,   j,   k, v] * dx * iy * iz +
                    mapings[  i, j+1,   k, v] * ix * dy * iz +
                    mapings[  i,   j, k+1, v] * ix * iy * dz +
                    mapings[i+1, j+1,   k, v] * dx * dy * iz +
                    mapings[i+1,   j, k+1, v] * dx * iy * dz +
                    mapings[  i, j+1, k+1, v] * ix * dy * dz +
                    mapings[i+1, j+1, k+1, v] * dx * dy * dz
                ) / size**3

    return out_image


def __clahe3d_gpu(
    image: NDArray | Tensor,
    size: int = 8,
    clip_limit: float = 5.0,
    n_bins: int = 256,
):
    is_tensor = isinstance(image, Tensor)
    data_device = "cpu" if not is_tensor else image.device.type

    if data_device == "cpu":
        image = torch.as_tensor(image, device="cuda")
    
    d_image = cuda.as_cuda_array(image)

    cdfs_shape = [
        (image.shape[i] + size - 1) // size
        for i in range(3)
    ]
    cdfs_shape.append(n_bins)

    t_cdfs = torch.zeros(cdfs_shape, device="cuda")
    d_cdfs = cuda.as_cuda_array(t_cdfs)
    
    tpb = (8, 8, 8)
    bpg = tuple(
        (image.shape[i] + tpb[i] - 1) // tpb[i]
        for i in range(3)
    )

    __calculate_hists[bpg, tpb](d_image, d_cdfs, size)

    t_cdfs.clip_(max=clip_limit)
    t_cdfs.cumsum_(dim=-1)

    cdfs_min = t_cdfs[..., 0]
    cdfs_max = t_cdfs[..., -1]
    
    # clip to avoid division by zero
    denom = (cdfs_max - cdfs_min).clip(min=1.0)

    t_cdfs.sub_(cdfs_min.unsqueeze(-1))
    t_cdfs.div_(denom.unsqueeze(-1))

    tpb = (8, 8, 8)
    bpg = tuple(
        (image.shape[i] + tpb[i] - 1) // tpb[i]
        for i in range(3)
    )

    t_output_image = torch.empty(image.shape, device="cuda")
    d_output_image = cuda.as_cuda_array(t_output_image)

    __interpolate[bpg, tpb](d_image, d_output_image, d_cdfs, size, n_bins)

    if is_tensor:
        return t_output_image.to(data_device)
    
    return t_output_image.cpu().numpy()


@cuda.jit
def __calculate_hists(image, hists, size):
    x, y, z = cuda.grid(3)
    w, h, d = image.shape
    _, _, _, n_bins = hists.shape

    if x >= w or y >= h or z >= d:
        return

    i = x // size
    j = y // size
    k = z // size

    # if x >= w: x = 2 * w - x - 1
    # if y >= h: x = 2 * h - y - 1
    # if z >= d: x = 2 * d - z - 1

    v = int(round(image[x, y, z] * (n_bins - 1)))
    cuda.atomic.add(hists, (i, j, k, v), 1.0) 


@cuda.jit
def __interpolate(input, output, cdfs, size, n_bins):
    w, h, d = output.shape
    w_, h_, d_, _ = cdfs.shape
    x, y, z = cuda.grid(3)

    if x >= w or y >= h or z >= d:
        return

    i = (x - size // 2) // size
    j = (y - size // 2) // size
    k = (z - size // 2) // size

    _i = i + 1
    _j = j + 1
    _k = k + 1

    dx = x - i * size - size / 2 + 0.5
    dy = y - j * size - size / 2 + 0.5
    dz = z - k * size - size / 2 + 0.5

    ix = size - dx
    iy = size - dy
    iz = size - dz

    i = max(i, 0)
    j = max(j, 0)
    k = max(k, 0)

    _i = min(_i, w_ - 1)
    _j = min(_j, h_ - 1)
    _k = min(_k, d_ - 1)

    v = int(round(input[x, y, z] * n_bins))

    output[x, y, z] = (
        cdfs[ i,  j,  k, v] * ix * iy * iz +
        cdfs[_i,  j,  k, v] * dx * iy * iz +
        cdfs[ i, _j,  k, v] * ix * dy * iz +
        cdfs[ i,  j, _k, v] * ix * iy * dz +
        cdfs[_i, _j,  k, v] * dx * dy * iz +
        cdfs[_i,  j, _k, v] * dx * iy * dz +
        cdfs[ i, _j, _k, v] * ix * dy * dz +
        cdfs[_i, _j, _k, v] * dx * dy * dz
    ) / size**3
