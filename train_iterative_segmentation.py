from glob import glob
from pathlib import Path
from sched import scheduler

import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.iterative_segmentation_dataset import IterativeSegDataset
from models.efficientnet3d import EfficientUNet3d
from models.unet3d import Unet3DIterative
from postprocessing.segmentation import segmentation
from train_segentation import get_ious

root = Path("./preprocessed_data")
output_dir = Path("./segmentation_validation")

import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def validation(model: nn.Module, dataset: torch.utils.data.Dataset, postprocessing_params):
    model.eval()

    ious = 0
    
    for i, (image, mask) in enumerate(tqdm(dataset, leave=False)):
        image = image[0].numpy()
        predicted_mask = segmentation(model, image, **postprocessing_params)
        ious += get_ious(predicted_mask, mask.numpy())

        name = dataset.image_names[i]
        torch.save(image, output_dir / "images3d" / name)
        torch.save(predicted_mask, output_dir / "segmentations" / name)

    ious /= len(dataset)

    return ious

def train_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    optimizer: torch.optim.Optimizer,
    epoch: int,
    total_steps: int,
    min_fp_coef: float = 0.1,
    warm_start: bool = False,
):
    model.train()
    # model.cuda()

    avg_seg_loss = 0
    avg_fn_loss = 0
    avg_cls_loss = 0
    avg_cmp_loss = 0
    
    num_samples = 0

    pbar = tqdm(dataloader)
    for step, batch in enumerate(pbar):
        global_step = epoch * len(dataloader) + step

        (
            image,
            instance_mask,
            binary_mask,
            weights,
            labels,
            completnesses
        ) = map(torch.Tensor.cuda, batch)

        output, label_logits, completness_logits = model(image, instance_mask)
        completness_logits = completness_logits.squeeze(-1)
        
        # gamma = 1.5
        # eps = 1e-4
        probas = torch.sigmoid(output)#.clip(eps, 1-eps)
        # print(torch.any(torch.isnan(probas)))
        # print(torch.any(torch.isnan(torch.log(probas))))
        # print(torch.any(torch.isnan(probas**gamma)))
        # print(torch.any(torch.isnan(torch.log(1 - probas))))
        # print(torch.any(torch.isnan((1 - probas)**gamma)))
        # print(torch.any(torch.isnan(weights)))
        # seg_loss = torch.functional.F.binary_cross_entropy_with_logits(output, binary_mask, weights)
        seg_loss = (weights * (binary_mask * (1 - probas) + 
                               (1 - binary_mask) * probas)).mean()
        # seg_loss = -(weights * ( binary_mask * (1 - probas)**gamma * torch.log(probas) + 
        #                          (1 - binary_mask) * probas**gamma * torch.log(1 - probas) )).mean()
        # print(seg_loss)
        # seg_fn_loss = (weights * binary_mask * (1 - probas)).mean()

        # if not warm_start:
        #     eta = (global_step - total_steps / 4) / (total_steps / 15)
        #     fp_coef = (1 - min_fp_coef) / (1 + np.exp(-eta)) + min_fp_coef
        # else:
        #     fp_coef = 1.0
        # exit()

        cls_loss = torch.functional.F.cross_entropy(label_logits, labels)
        cmp_loss = torch.functional.F.binary_cross_entropy_with_logits(completness_logits, completnesses.float())

        total_loss = 10 * seg_loss + cls_loss + cmp_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        avg_seg_loss += seg_loss.item() * image.shape[0]
        # avg_fn_loss += seg_fn_loss.item()
        avg_cls_loss += cls_loss.item() * image.shape[0]
        avg_cmp_loss += cmp_loss.item() * image.shape[0]

        num_samples += image.shape[0]

        pbar.set_description(
            f"Epoch: {epoch+1} || "
            f"SEG: {avg_seg_loss / num_samples:.4f} | "
            # f"FN: {avg_fn_loss / num_samples:.4f} | "
            f"CLS: {avg_cls_loss / num_samples:.4f} | "
            f"CMP: {avg_cmp_loss / num_samples:.4f} | "
            # f"lambda: {fp_coef:.2f} | "
            f"lr: {optimizer.param_groups[0]['lr']:.3}"
        )
        pbar.refresh()

def train():
    batch_size = 4
    num_epochs = 1000
    learning_rate = 1e-4
    crop_size = 96
    pretrained_weights = "weights/iterative_segmentation_last_weights.pth"
    postprocessing_params = dict(
        size=crop_size,
        min_voxels_partial=1000,
        min_voxels = 2500,
        filter_mask=True,
        closing_radius=0,
    )

    image_names = [
        path.name
        for path in Path("preprocessed_data/images3d").glob("*")
    ]

    # presented_vertebrea = \
    #     pd.read_csv("preprocessed_data/presented_vertebrea.csv", index_col=0)
    # image_names = presented_vertebrea.index[presented_vertebrea.T.all()].tolist()

    # image_names.extend([
    #     "sub-verse506",
    #     "sub-verse521",
    # ])

    val_images = image_names[:20]
    train_images = image_names[20:]

    train_data = IterativeSegDataset(root, train_images, train=True, size=crop_size)
    val_data = IterativeSegDataset(root, val_images, train=False)

    train_dataloader = DataLoader(
        train_data, batch_size, shuffle=True, num_workers=0, pin_memory=True,
    )

    model = Unet3DIterative().cuda()
    # model = EfficientUNet3d().cuda()
    if pretrained_weights:
        model.load_state_dict(torch.load(pretrained_weights))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    best_score = 0

    total_steps = num_epochs * len(train_dataloader)
    warm_start = pretrained_weights != ""

    if warm_start:
        print(f"starting with pretrained net with weights path: {pretrained_weights}")

    for epoch in tqdm(range(num_epochs)):
        train_epoch(model, train_dataloader, optimizer, epoch, total_steps, warm_start=warm_start)
        gc.collect()
        torch.cuda.empty_cache()

        if epoch > 5 or warm_start:
            ious = validation(model, val_data, postprocessing_params)
            mIoU = ious.mean()

            print(flush=True)
            print(f"mIoU: {mIoU:.4f} || {' | '.join(f'C{i+1}: {ious[i]:.2f}' for i in range(7))}")

            if mIoU > best_score:
                best_score = mIoU
                torch.save(model.state_dict(), "weights/iterative_segmentation_best_weights.pth")
            
            torch.save(model.state_dict(), "weights/iterative_segmentation_last_weights.pth")

        scheduler.step()

    print(f"BestScore: {best_score}")


if __name__ == "__main__":
    train()
