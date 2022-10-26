from argparse import ArgumentParser
from collections import deque
import cv2
import numpy as np
from pathlib import Path
from utils.visualization import visualize
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from utils.affine3d import resize

class Images(Dataset):
    def __init__(self, root, labels_path, scan_uids):
        self.root = root
        self.labels = pd.read_csv(labels_path).set_index("StudyInstanceUID")
        self.scan_uids = scan_uids

    def __len__(self):
        return len(self.scan_uids)

    def __getitem__(self, item):
        image_uid = self.scan_uids[item]
        img_slices = []
        mask_slices = []
        
        for path in self.root.glob(f"images/{image_uid}/*.png"):
            image = cv2.imread(str(path), 2) / 65535
            mask = cv2.imread(str(path).replace("images", "masks"), 0)
            img_slices.append(image[::-1].T)
            mask_slices.append(mask[::-1].T)

        image = np.stack(img_slices, axis=-1)
        mask = np.stack(mask_slices, axis=-1)

        image = resize(image, scale=(1, 1, 2))
        mask = resize(mask, scale=(1, 1, 2), interpolation="nearest")

        labels = self.labels.loc[image_uid, [f"C{i}" for i in range(1, 8)]].to_numpy()
        
        return image, mask, labels

def id(x):
    return x[0]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("split", type=str)
    args = parser.parse_args()

    scan_uids = np.load(args.split, allow_pickle=True) 

    root = Path("data/train_slices")

    dataset = Images(root, "./data/train.csv", scan_uids)

    dataloader = DataLoader(dataset, 1, collate_fn=id, num_workers=2, prefetch_factor=1)
    dl_iter = iter(dataloader)
    memory = {0: next(dl_iter)}

    i = 0
    def get_other(diff):
        global i
        global dl_iter
            
        i = i + diff
        if i == len(scan_uids):
            cv2.destroyAllWindows()
            exit(0)

        if i not in memory:
            if i < min(memory.keys()):
                i = min(memory.keys())
            elif i == max(memory.keys()) + 1:
                memory[i] = next(dl_iter)

            if max(memory.keys()) - min(memory.keys()) > 2:
                memory.pop(min(memory.keys()))

        image, mask, labels = memory[i]

        scan_uid = scan_uids[i]
        return image, mask, scan_uid, labels
    
    image, mask, name, labels = get_other(0)

    mkup_dir = Path("./data/our_bboxes")
    mkup_dir.mkdir(exist_ok=True)

    visualize(image, mask, name=name, labels=labels, mkup_dir=mkup_dir, get_other=get_other)
