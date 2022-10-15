import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit
def find_borders(binary_mask: NDArray[np.bool]) -> NDArray[np.bool]:
    # 6-connectivity offsets
    offsets = [
        (0, 0, 1),
        (0, 1, 0),
        (1, 0, 0),
        
        (0, 0, -1),
        (0, -1, 0),
        (-1, 0, 0),
    ]

    w, h, d = binary_mask.shape

    mask = np.zeros_like(binary_mask)

    for x in range(w):
        for y in range(h):
            for z in range(d):
                if not binary_mask[x, y, z]:
                    continue

                if x == 0 or y == 0 or z == 0 or \
                   x == w-1 or y == h-1 or z == d-1:
                    mask[x, y, z] = 1
                    continue

                for dx, dy, dz in offsets:
                    if not binary_mask[x+dx, y+dy, z+dz]:
                        mask[x, y, z] = 1
                        break

    return mask


@njit
def distance_from_surface(binary_mask: NDArray[np.bool], max_dist=25) -> NDArray[np.int64]:
    # 6-connectivity offsets
    offsets = np.array([
        (0, 0, 1),
        (0, 1, 0),
        (1, 0, 0),
        
        (0, 0, -1),
        (0, -1, 0),
        (-1, 0, 0),
    ])
    
    w, h, d = binary_mask.shape
    inf = max_dist + 2
    
    borders_mask = find_borders(binary_mask)
    been_here = np.zeros_like(binary_mask)
    distances = np.ones_like(binary_mask, dtype=np.int64) * inf
    borders_points = np.stack(np.where(borders_mask), axis=-1)

    head = []
    tail = []

    for x, y, z in borders_points:
        distances[x, y, z] = 0
        been_here[x, y, z] = True
        tail.append([x, y, z])

    while head or tail:
        if not head:
            while tail:
                head.append(tail.pop()) 

        x, y, z = head.pop()

        for dx, dy, dz in offsets:
            i = x + dx
            j = y + dy
            k = z + dz

            if (i < 0 or j < 0 or k < 0 or
                i >= w or j >= h or k >= d or
                been_here[i, j, k]):

                continue

            been_here[i, j, k] = True
            distances[i, j, k] = distances[x, y, z] + 1
            tail.append([i, j, k])

    return distances


def calculate_weights(binary_mask: NDArray[np.bool], gamma=8, sigma=6, tol=1e-3) -> NDArray[np.float64]:
    max_dist = np.ceil((-np.log(tol / gamma) * sigma**2) ** 0.5)

    distances = distance_from_surface(binary_mask, max_dist=max_dist)
    return gamma * np.exp(-distances**2 / sigma**2) + 1.0
