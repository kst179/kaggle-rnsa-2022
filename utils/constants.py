import numpy as np
from pathlib import Path

root_dir = Path("./data")
segmentation_preprocessed_dir = Path("preprocessed_data")
external_data_root = Path("./external_data")


colors = np.array([
    [0, 0, 0],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [0, 255, 255],
    [255, 0, 255],
    [255, 255, 255],
])