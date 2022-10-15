from dataclasses import dataclass
import gc
from multiprocessing import reduction
from pathlib import Path
from sched import scheduler
from tabnanny import check
from turtle import forward, pos, position
from typing import OrderedDict
import numpy as np
import pandas as pd
import torch
from torch import nn
import tqdm

from datasets.vertabrea_dataset import VertebreaDataset
from models.efficientnet3d import EfficientNet3d
from sklearn.model_selection import train_test_split

# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

@dataclass
class CheckpointData:
    model_state_dict: OrderedDict
    optim_state_dict: OrderedDict
    sched_state_dict: OrderedDict
    next_epoch: int
    training_completed: bool    


class Checkpoint():
    def __init__(self, model, optimizer, scheduler, path="weights/classification_checkpoint.pth"):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.path = path

    def load(self):
        checkpoint: CheckpointData = torch.load(self.path)

        self.model.load_state_dict(checkpoint.model_state_dict)
        self.optimizer.load_state_dict(checkpoint.optim_state_dict)
        self.scheduler.load_state_dict(checkpoint.sched_state_dict)

        return checkpoint.next_epoch, checkpoint.training_completed
    
    def __call__(self, next_epoch, training_completed=False):
        checkpoint = CheckpointData(
            self.model.state_dict(),
            self.optimizer.state_dict(),
            self.scheduler.state_dict(),
            next_epoch,
            training_completed,
        )

        torch.save(checkpoint, self.path)


class WeightedBCEWithLogits(nn.Module):
    def __init__(self, positive_weight=2, negative_weight=1):
        super().__init__()
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight

    def forward(self, input, target):
        weights = torch.where(target == 1, self.positive_weight, self.negative_weight)
        loss = torch.functional.F.binary_cross_entropy_with_logits(input, target.float(), reduction="none")
        return (loss * weights).sum() / weights.sum()


def take_first(iterable, n=100):
    for item, _ in zip(iterable, range(n)):
        yield item


def train():
    num_epochs = 20
    batch_size = 1
    step_each = 8   # num batches in single optimization step
    checkpoint_path = Path("weights/classification_checkpoint.pth")
    load_from_checkpoint = True
    start_lr = 1e-2
    end_lr = 1e-5
    val_batch_size = 2
    first_epoch = 0
    skip_first_epoch_training = False

    train_imgs_dir = Path("./data/train_images")

    images = [
        path.name.removesuffix(".nii.gz")
        for path in train_imgs_dir.glob("*.nii.gz")
    ]

    train_images, val_images = train_test_split(
        images, test_size=0.2, shuffle=True, random_state=179)

    dataset_params = dict(
        images_dir="data/train_images/",
        segmentations_dir="data/generated_segmentations/",
        segmentation_meta_path="data/segmentation_meta.json",
        labels_file="data/train.csv",
        num_workers=4,
        prefetch_factor=1,
        image_size_mm=(96, 96, 48),
        image_res_pix_per_mm=(2, 2, 2),
    )

    train_data = VertebreaDataset(
        image_names=train_images,
        shuffle=True,
        **dataset_params,
    )

    val_data = VertebreaDataset(
        image_names=val_images,
        **dataset_params,
    )

    gamma = (end_lr / start_lr) ** (1 / (num_epochs - 1))

    model = EfficientNet3d().cuda()
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=start_lr, momentum=0.9, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

    checkpoint = Checkpoint(model, optimizer, scheduler, checkpoint_path)

    if load_from_checkpoint:
        first_epoch, skip_first_epoch_training = checkpoint.load()
        
        print(f"Started with checkpoint: {checkpoint_path}, epoch: {first_epoch}")
        if skip_first_epoch_training:
            print("First epoch training is skipped")

    loss_fn = WeightedBCEWithLogits()

    best_loss = torch.inf

    for epoch in tqdm.trange(first_epoch, num_epochs):
        if not skip_first_epoch_training:
            checkpoint(epoch)
            train_epoch(model, train_data, optimizer, loss_fn, batch_size, epoch, 
                        step_each=step_each, checkpoint=checkpoint, checkpoint_each=1000)
            checkpoint(epoch, training_completed=True)
        else:
            skip_first_epoch_training = False

        val_loss = validation(model, val_data, loss_fn, val_batch_size, epoch)
        print()
        print(f"Val loss: {val_loss:.5}")
        # break

        if best_loss > val_loss:
            torch.save(model.state_dict(), "weights/classification_best_weights.pth")
        scheduler.step()

    print(f"Best loss: {best_loss}")


def get_streaming_loader(
    dataset: VertebreaDataset, 
    batch_size: int
):
    pbar = tqdm.tqdm(dataset.dataloader, smoothing=0.01)

    def loader():
        nonlocal pbar

        batch = []

        for raw_data in pbar:
            gc.collect
            torch.cuda.empty_cache()

            for vertebre_data in dataset.get_vertebrea(raw_data):
                if vertebre_data is None:
                    continue

                batch.append(vertebre_data)

                if len(batch) == batch_size:
                    yield torch.utils.data.default_collate(batch)
                    batch = []

        if batch:
            yield torch.utils.data.default_collate(batch)

    return pbar, loader()


def train_epoch(
    model: nn.Module,
    dataset: VertebreaDataset,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    batch_size: int,
    epoch: int,
    step_each: int,
    checkpoint: Checkpoint,
    checkpoint_each: int,
):
    model.train()
    cumulative_loss = 0
    num_samples = 0

    pbar, loader = get_streaming_loader(dataset, batch_size)
    #  = tqdm.tqdm(dataset.dataloader, smoothing=0.01)

    for i, batch in enumerate(loader):
        vertebre_image, target_vertebre_mask, other_vertebrea_mask, label = batch
        label = label.unsqueeze(1)
            
        output = model(
            vertebre_image,
            target_vertebre_mask, 
            other_vertebrea_mask
        )
        loss = loss_fn(output, label)

        loss.backward()

        if i % step_each == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if (i + 1) % checkpoint_each == 0:
            checkpoint(epoch)

        cumulative_loss += loss.item()
        num_samples += label.shape[0]

        pbar.set_description(f"Epoch [{epoch: >2}] Train loss: {cumulative_loss / num_samples:.5}")

        gc.collect()
        torch.cuda.empty_cache()

    if next(iter(model.parameters())).grad is not None:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def validation(
    model: nn.Module,
    dataset: VertebreaDataset,
    loss_fn: nn.Module,
    batch_size: int,
    epoch: int,
):
    model.eval()
    outputs = []
    labels = []

    pbar, loader = get_streaming_loader(dataset, batch_size)
    pbar.set_description(f"Epoch [{epoch: >2}] Validation")

    for batch in loader:
        vertebre_image, target_vertebre_mask, other_vertebrea_mask, label = batch
        label = label.unsqueeze(1)

        output = model(
            vertebre_image,
            target_vertebre_mask, 
            other_vertebrea_mask
        )

        outputs.append(output.to("cpu", non_blocking=True))
        labels.append(label.to("cpu", non_blocking=True))

        gc.collect()
        torch.cuda.empty_cache()

    outputs = torch.cat(outputs)
    labels = torch.cat(labels)

    loss = loss_fn(outputs, labels)
    
    del batch, label, labels, outputs
    gc.collect()
    torch.cuda.empty_cache()

    return loss


if __name__ == "__main__":
    train()
