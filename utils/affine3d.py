from numba import njit, cuda
import math
import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from typing import Tuple


@njit
def __warp_affine_cpu(input, output, T, order=1):
    w, h, d = input.shape
    w_, h_, d_ = output.shape

    for i in range(w_):
        for j in range(h_):
            for k in range(d_):
                x = T[0, 0] * i + T[0, 1] * j + T[0, 2] * k + T[0, 3]
                y = T[1, 0] * i + T[1, 1] * j + T[1, 2] * k + T[1, 3]
                z = T[2, 0] * i + T[2, 1] * j + T[2, 2] * k + T[2, 3]
            
                if x < 0 or y < 0 or z < 0 or x + 1 >= w or y + 1 >= h or z + 1 >= d:
                    output[i, j, k] = 0
                    continue

                if order == 0:
                    output[i, j, k] = input[
                        int(round(x)),
                        int(round(y)),
                        int(round(z)),
                    ]
                else:
                    x_ = int(math.floor(x))
                    y_ = int(math.floor(y))
                    z_ = int(math.floor(z))

                    dx = x - x_
                    dy = y - y_
                    dz = z - z_

                    ix = 1 - dx
                    iy = 1 - dy
                    iz = 1 - dz

                    output[i, j, k] = (
                        input[x_  , y_  , z_  ] * ix * iy * iz +
                        input[x_+1, y_  , z_  ] * dx * iy * iz +
                        input[x_  , y_+1, z_  ] * ix * dy * iz +
                        input[x_  , y_  , z_+1] * ix * iy * dz +
                        input[x_+1, y_+1, z_  ] * dx * dy * iz +
                        input[x_+1, y_  , z_+1] * dx * iy * dz +
                        input[x_  , y_+1, z_+1] * ix * dy * dz +
                        input[x_+1, y_+1, z_+1] * dx * dy * dz
                    )


@cuda.jit
def __warp_affine_cuda(input, output, T, order=1):
    w, h, d = input.shape
    w_, h_, d_ = output.shape
    i, j, k = cuda.grid(3)
    
    if i >= w_ or j >= h_ or k >= d_:
        return

    x = T[0, 0] * i + T[0, 1] * j + T[0, 2] * k + T[0, 3]
    y = T[1, 0] * i + T[1, 1] * j + T[1, 2] * k + T[1, 3]
    z = T[2, 0] * i + T[2, 1] * j + T[2, 2] * k + T[2, 3]

    if x < 0 or y < 0 or z < 0 or x + 1 >= w or y + 1 >= h or z + 1 >= d:
        output[i, j, k] = 0
        return

    if order == 0:
        output[i, j, k] = input[
            int(round(x)),
            int(round(y)),
            int(round(z)),
        ]
        return

    x_ = int(math.floor(x))
    y_ = int(math.floor(y))
    z_ = int(math.floor(z))

    dx = x - x_
    dy = y - y_
    dz = z - z_

    ix = 1 - dx
    iy = 1 - dy
    iz = 1 - dz

    output[i, j, k] = (
        input[x_  , y_  , z_  ] * ix * iy * iz +
        input[x_+1, y_  , z_  ] * dx * iy * iz +
        input[x_  , y_+1, z_  ] * ix * dy * iz +
        input[x_  , y_  , z_+1] * ix * iy * dz +
        input[x_+1, y_+1, z_  ] * dx * dy * iz +
        input[x_+1, y_  , z_+1] * dx * iy * dz +
        input[x_  , y_+1, z_+1] * ix * dy * dz +
        input[x_+1, y_+1, z_+1] * dx * dy * dz
    )


def warp_affine3d(
    image: Tensor,
    inv_transform: NDArray,
    shape: Tuple[int] = None,
    interpolation: str = "linear", 
    device: str = "auto",
) -> Tensor:
    """
    Applies 3d affine transformation to image via 
    trilinear or nearest neighbors interpolation methods

    Args:
        image (NDArray | Tensor): image to be transformed
        inv_transform (NDArray): 
            Inverse affine transform to be applied.
            Should be 4x4 matrix representing mapping from warped image coordinates to original one:
                (x, y, z, 1) = inv_transform @ (x', y', z', 1),
            where x, y, z is a coordiates of the original image
            and x', y', z' is a coordinates of the warped one
        shape (Tuple[int] | torch.Size | None):
            Desirable size of the warped image.
            If None, the shape of original image is taken.
            Defaults to None.
        interpolation (Literal["linear", "nearest"]): 
            interpolation method. Defaults to "linear".
        device (Literal["auto", "cpu", "cuda"]):
            device used for calculations. If "auto",
            device is determined by input data location.
            Defaults to "auto".

    Returns:
        NDArray | Tensor:
            warped image, same array type and dtype as the input image, 
            the shape is equal to given one (if shape is not None, else 
            it is equal to the input image shape).
    """

    is_tensor = isinstance(image, Tensor)
    data_device = "cpu" if not is_tensor else image.device.type

    if device == "auto":
        device = data_device

    if device == "cpu" and is_tensor:
        image = image.cpu().numpy()

    order = 1 if interpolation == "linear" else 0

    if shape is None:
        shape = image.shape

    if not isinstance(shape, Tuple):
        shape = tuple(shape)
    
    if device == "cpu":
        warped_image = np.empty(shape, dtype=image.dtype)
        __warp_affine_cpu(image, warped_image, inv_transform, order)

        if is_tensor:
            return torch.as_tensor(warped_image, device=data_device)

        return warped_image

    nthreads = (8, 8, 8)
    nblocks = tuple(
        (shape[i] + nthreads[i] - 1) // nthreads[i]
        for i in range(3)
    )

    if data_device == "cpu":
        image = torch.as_tensor(image, device="cuda")
        
    d_image = cuda.as_cuda_array(image)
    
    inv_transform = torch.as_tensor(inv_transform, device="cuda")
    d_inv_transform = cuda.as_cuda_array(inv_transform)

    warped_image = torch.empty(shape, dtype=image.dtype, device="cuda")
    d_warped_image = cuda.as_cuda_array(warped_image)

    __warp_affine_cuda[nblocks, nthreads](d_image, d_warped_image, d_inv_transform, order)

    if is_tensor:
        return warped_image.to(data_device)

    return warped_image.cpu().numpy()


def translation_tfm(t):
    transform = np.eye(4)
    transform[:3, 3] = t
    return transform


def scale_tfm(s):
    transform = np.eye(4)
    transform[range(3), range(3)] = s
    return transform


def rotation_tfm(R):
    transform = np.eye(4)
    transform[:3, :3] = R
    return transform


def euler_rotation_tfm(x=0, y=0, z=0, order="xyz", degrees=True):
    if degrees:
        x, y, z = (
            a * np.pi / 180
            for a in (x, y, z)
        )

    R = np.eye(3)
    for s in order:
        if s == "x":
            c = np.cos(x)
            s = np.sin(x)

            R = [
                [1, 0, 0],
                [0, c, -s],
                [0, s, c],
            ] @ R

        elif s == "y":
            c = np.cos(y)
            s = np.sin(y)

            R = [
                [c, 0, s],
                [0, 1, 0],
                [-s, 0, c],
            ] @ R

        elif s == "z":
            c = np.cos(z)
            s = np.sin(z)

            R = [
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1],
            ] @ R
    
    return rotation_tfm(R)


def resize(
    image: Tensor,
    scale: Tuple[float, float, float] = None,
    size: Tuple[int, int, int] = None,
    interpolation: str = "linear",
    device: str = "auto",
):
    if scale is not None and size is not None:
        raise ValueError("Need to specify only one of `scale` or `size`")

    old_size = np.array(image.shape)

    if scale is not None:
        scale = np.array(scale)
        size = np.ceil(old_size * scale).astype(int)
        scale = size / old_size

    elif size is not None:
        size = np.array(size)
        scale = size / old_size

    else:
        raise ValueError("Need to specify at least one of `scale` or `size`")

    transform = scale_tfm(1 / scale)
    return warp_affine3d(image, transform, size, interpolation, device)


def center_rotation(
    image: Tensor,
    x: float = 0,
    y: float = 0,
    z: float = 0,
    order: str = "xyz",
    degrees: bool = True,
    interpolation: str = "linear",
    device: str = "auto",
):
    size = np.array(image.shape)

    transform = (
        translation_tfm(size / 2) @
        euler_rotation_tfm(x, y, z, order, degrees).T @
        translation_tfm(-size / 2)
    )

    return warp_affine3d(image, transform, size, interpolation, device)


# if __name__ == "__main__":
#     from matplotlib import pyplot as plt

#     s = 128
#     image = np.arange(s**2).reshape(s, s, 1).repeat(3, -1)

#     center_rotation = np.linalg.inv(
#         translation_tfm([s/2 + 0.5, s/2 + 0.5, 0]) @
#         euler_rotation_tfm(0, 0, 45) @
#         translation_tfm([-s/2 + 0.5, -s/2 + 0.5, 0])
#     )

#     center_resize = np.linalg.inv(
#         translation_tfm([s/2, s/2, 0]) @ 
#         scale_tfm([0.5, 0.5, 1]) @
#         translation_tfm([-s/2, -s/2, 0])
#     )

#     rotated = warp_affine3d(image, center_rotation, shape=image.shape)
#     rescaled = warp_affine3d(image, center_resize, shape=image.shape)
#     resized = resize(image, [0.5, 0.5, 1.0])

#     plt.subplot(221)
#     plt.imshow(image[..., 1])
#     plt.subplot(222)
#     plt.imshow(rotated[..., 1])
#     plt.subplot(223)
#     plt.imshow(rescaled[..., 1])
#     plt.subplot(224)
#     plt.imshow(resized[..., 1])
#     plt.show()