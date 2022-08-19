from numba import njit
import numpy as np
import cv2
import torch

@njit
def clahe3d(image: np.ndarray, size: int = 8, clip_limit=5, n_bins=256):
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

                hist, _ = np.histogram(block.flatten(), n_bins, [0, n_bins])
                hist = hist.clip(0, clip_limit)
                cdf = np.cumsum(hist).astype(np.float64)
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


def denoising(image):
    image = (image * 255).astype(np.uint8)
    images = list(image.transpose(2, 0, 1))
    denoised_imgs = np.empty_like(image)

    for i in range(1, image.shape[2] - 1):
        denoised_imgs[:, :, i] = cv2.fastNlMeansDenoisingMulti(images, i, temporalWindowSize=3)

    return denoised_imgs.astype(np.float64) / 255


def minmax_normalize(image):
    min_val = image.min()
    max_val = image.max()
    return (image - min_val) / (max_val - min_val)


def rescale(image, spacings, resolution, interpolation="trilinear"):
    image = torch.tensor(np.ascontiguousarray(image))[None, None, ...]
    size = np.array(image.shape[-3:])
    size = np.floor(size * spacings / resolution).astype(int)
    size = tuple(size)

    image = torch.functional.F.interpolate(image, size=size, mode=interpolation)
    return image.numpy()[0, 0]

def pad16(image):
    w, h, d = image.shape[-3:]
    w_, h_, d_ = (np.array([w, h, d]) + 15) // 16 * 16
    image = torch.functional.F.pad(image, (0, d_ - d, 0, h_ - h, 0, w_ - w))

    return image