import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from pathlib import Path

from datasets.segmentation_dataset import SegmentationDataset
from models.unet3d import Unet3D

from tqdm import tqdm

root = Path("./preprocessed_data")
output_dir = Path("./segmentation_validation")

def get_ious(x, y):
    ious = []

    for label in range(1, 8):
        x_ = x == label
        y_ = y == label

        i = x_ & y_
        u = x_ | y_

        if u.sum() == 0:
            ious.append(1)
        else:
            ious.append(i.sum() / u.sum())

    return torch.tensor(ious)


def validation(model: nn.Module, dataloader: DataLoader):
    model.eval()
    model.cpu()

    ious = 0
    
    for i, (image, mask) in enumerate(tqdm(dataloader)):
        # image, mask = image.cuda(), mask.cuda()

        # pad spatial dims to be divisable by 16
        w, h, d = image.shape[-3:]
        w_, h_, d_ = (np.array([w, h, d]) + 15) // 16 * 16
        image = torch.functional.F.pad(image, (0, d_ - d, 0, h_ - h, 0, w_ - w))

        with torch.no_grad():
            output = model(image)[:, :, :w, :h, :d]

        out_mask = output.argmax(dim=1)

        ious += get_ious(out_mask, mask)

        name = dataloader.dataset.image_names[i]
        torch.save(image[0, 0].cpu(), output_dir / "images3d" / name)
        torch.save(out_mask[0].cpu(), output_dir / "segmentations" / name)

    ious /= len(dataloader.dataset)

    return ious

def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer):
    model.train()
    model.cuda()

    avg_loss = 0
    num_samples = 0

    pbar = tqdm(dataloader)
    for image, mask in pbar:
        image, mask = image.cuda(), mask.cuda()

        output = model(image)
        loss = torch.functional.F.cross_entropy(output, mask)

        avg_loss += loss.item()
        num_samples += image.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"{avg_loss / num_samples:.4}")
        pbar.refresh()

    avg_loss /= num_samples
    return avg_loss

def train():
    batch_size = 2
    num_epochs = 100
    learning_rate = 3e-4

    image_names = [
        path.name
        for path in root.glob("images3d/*")
    ]

    val_images = image_names[:20]
    train_images = image_names[20:]

    train_data = SegmentationDataset(root, train_images, train=True)
    val_data = SegmentationDataset(root, val_images, train=False)

    train_dataloader = DataLoader(
        train_data, batch_size, shuffle=True, num_workers=0, pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True,
    )

    model = Unet3D(batch_norm=True).cuda()

    # ious = validation(model, val_dataloader)
    # print(ious)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_score = 0

    for epoch in tqdm(range(num_epochs)):
        loss = train_epoch(model, train_dataloader, optimizer)
        print(flush=True)
        print(f"Loss: {loss}")

        ious = validation(model, val_dataloader)
        mIoU = ious.mean()

        print(flush=True)
        print(f"mIoU: {mIoU}")
        for i in range(7):
            print(f"\tC{i+1}: {ious[i]}")

        if mIoU > best_score:
            best_score = mIoU
            torch.save(model.state_dict(), "segmentation_best_weights.pth")

    print(f"BestScore: {best_score}")


if __name__ == "__main__":
    train()