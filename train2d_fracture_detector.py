import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm

from datasets import ImagesDataset
from train_classification import WeightedBCEWithLogits

from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def train():
    num_epoch = 100
    lr = 3e-4
    batch_size = 2
    step_each = 8
    val_batch_size = 8
    weights_path = "weights/effnet2d_classification_axial_last_checkpoint.pth"

    seg_meta = pd.read_json("./data/segmentation_meta.json")
    bad_images = seg_meta[seg_meta["num_voxels"] < 5000]["uid"].unique()
    good_images = list(set(seg_meta["uid"].unique()) - set(bad_images))

    train_images, val_images = train_test_split(good_images, test_size=0.2,
                                                shuffle=True, random_state=179)

    root = Path("data/train_images2d/axial")
    labels_path = Path("data/train.csv")
    train_dataset = ImagesDataset(root, labels_path, train_images)
    val_dataset = ImagesDataset(root, labels_path, val_images)

    loaders_params = dict(
        num_workers=1, # os.cpu_count(),
        pin_memory=True, 
        persistent_workers=True,
    )

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, **loaders_params)
    val_loader = DataLoader(val_dataset, val_batch_size, **loaders_params)

    model = torchvision.models.efficientnet_b5(pretrained=True, progress=True)
    model.features[0][0] = nn.Conv2d(1, 48, 3, 2, 1, bias=False)
    model.classifier[1] = nn.Linear(2048, 1 + 7)
    model.cuda()

    if weights_path:
        print(f"Start learning from {weights_path}")
        model.load_state_dict(torch.load(weights_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    best_loss = torch.inf

    for epoch in tqdm.trange(num_epoch):
        train_epoch(model, optimizer, scheduler, train_loader, epoch, step_each)
        loss = validation(model, val_loader, epoch)

        if loss < best_loss:
            best_loss = best_loss
            torch.save(model.state_dict(), "weights/effnet2d_classification_axial_best_checkpoint.pth")
        
        torch.save(model.state_dict(), "weights/effnet2d_classification_axial_last_checkpoint.pth")

        scheduler.step()


def train_epoch(
    model: Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    dataloader: DataLoader,
    epoch: int,
    step_each: int,
):
    _ = scheduler

    model.train()

    weighted_bce = WeightedBCEWithLogits()
    cross_entropy = nn.CrossEntropyLoss()

    fcl = 0
    vcl = 0
    num = 0

    pbar = tqdm.tqdm(dataloader)
    for i, batch in enumerate(pbar):
        images, fracture_labels, vertebre_labels = [
            t.to("cuda") for t in batch
        ]

        output = model(images)
        fracture_logits, vertebre_logits = output[:, 0], output[:, 1:]
        fracture_cls_loss = weighted_bce(fracture_logits, fracture_labels)
        vertebre_cls_loss = cross_entropy(vertebre_logits, vertebre_labels)
        
        loss = fracture_cls_loss + 0.2 * vertebre_cls_loss
        loss.backward()

        batch_size = images.shape[0]
        fcl += fracture_cls_loss.item() * batch_size
        vcl += vertebre_cls_loss.item() * batch_size
        num += batch_size

        if (i + 1) % step_each == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            pbar.set_description(f"Train || Epoch [{epoch: >3}] || "
                                 f"FCL: {fcl / num:.5f} | "
                                 f"VCL: {vcl / num:.5f} | "
                                 f"lr: {optimizer.param_groups[0]['lr']:.3}")
    pbar.close()


@torch.no_grad()
def validation(
    model: Module,
    val_loader: DataLoader,
    epoch: int
):
    model.eval()

    weighted_bce = WeightedBCEWithLogits()
    cross_entropy = nn.CrossEntropyLoss()

    fcl = 0
    vcl = 0
    num = 0
    
    pbar = tqdm.tqdm(val_loader)

    for batch in pbar:
        images, fracture_labels, vertebre_labels = [
            t.to("cuda") for t in batch
        ]

        output = model(images)
        fracture_logits, vertebre_logits = output[:, 0], output[:, 1:]
        fracture_cls_loss = weighted_bce(fracture_logits, fracture_labels)
        vertebre_cls_loss = cross_entropy(vertebre_logits, vertebre_labels)

        batch_size = images.shape[0]

        fcl += fracture_cls_loss.item() * batch_size
        vcl += vertebre_cls_loss.item() * batch_size
        num += batch_size

        pbar.set_description(f"Val   || Epoch [{epoch: >3}] || "
                                f"FCL: {fcl / num:.5f} | "
                                f"VCL: {vcl / num:.5f}")
    pbar.close()

    return fcl / num


if __name__ == "__main__":
    train()