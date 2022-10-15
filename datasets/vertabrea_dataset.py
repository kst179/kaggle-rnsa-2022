import gc
import torch
from pathlib import Path
from typing import Tuple
import nibabel as nib
import pandas as pd
from typing import List
from preprocessing.clahe3d import clahe3d
from datasets.nii_dataset import NiiDataset 
from utils.affine3d import resize, rotation_tfm, translation_tfm, scale_tfm, warp_affine3d
from utils.misc import voxels_pca, split_to_patches
import numpy as np
import numba

from utils.misc import minmax_normalize

class VertebreaDataset:
    def __init__(
        self,
        images_dir: str | Path,
        segmentations_dir: str | Path,
        segmentation_meta_path: str | Path,
        labels_file: str | Path,
        image_names: List[str | Path] | None = None,
        num_workers: int = 2,
        prefetch_factor: int = 2,
        shuffle: bool = True,
        persistent_workers: bool = True,
        device: str | torch.device = "cuda",
        image_size_mm: Tuple[int] = (96, 96, 48),
        image_res_pix_per_mm: Tuple[int] = (1, 1, 1),
        min_voxels = 5000,
    ):
        self.images_dir = Path(images_dir)
        self.segmentations_dir = Path(segmentations_dir)

        segmentations_meta = pd.read_json(segmentation_meta_path)
        self.segmentations_meta = segmentations_meta.set_index(["uid", "vertebre"])
        self.labels = pd.read_csv(labels_file).set_index("StudyInstanceUID")

        self.image_size_mm = np.asarray(image_size_mm)
        self.image_res_pix_per_mm = np.asarray(image_res_pix_per_mm)

        self.device = device
        self.min_voxels = min_voxels

        self.data = NiiDataset(
            root=self.images_dir,
            seg_masks_root=self.segmentations_dir,
            image_names=image_names,
            return_affine=True,
            return_name=True,
        )
        self.dataloader = torch.utils.data.DataLoader(
            self.data,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            shuffle=shuffle,
        )

    def get_vertebrea(self, batch):
        image, mask, name, affine = batch

        scale = affine[0].numpy().diagonal()[:3]
        name = name[0]

        voxels = (mask[0] > 0).nonzero()
        if len(voxels) < self.min_voxels:
            return []

        min_bound = voxels.min(dim=0).values
        max_bound = voxels.max(dim=0).values
        bbox_center = (min_bound + max_bound) / 2
        bbox_size = max_bound - min_bound
        bbox_size = bbox_size * 1.1

        mask_offset = (bbox_center - bbox_size / 2).long()
        image_offset = ((bbox_center - bbox_size / 2) / scale).long()
        bbox_size = bbox_size.long()

        x1, y1, z1 = np.maximum(image_offset.numpy(), 0)
        x2, y2, z2 = np.minimum((image_offset + (bbox_size / scale).long()).numpy(), image.shape[1:])
        image = torch.as_tensor(image[0, x1:x2, y1:y2, z1:z2], dtype=torch.float, device=self.device)

        x1, y1, z1 = np.maximum(mask_offset.numpy(), 0)
        x2, y2, z2 = np.minimum((mask_offset + bbox_size).numpy(), mask.shape[1:])
        mask = torch.as_tensor(mask[0, x1:x2, y1:y2, z1:z2], dtype=torch.long, device=self.device)

        # image = image[0].cuda()
        # mask = mask[0].cuda()

        image = minmax_normalize(image)
        image = clahe3d(image, size=64, clip_limit=2, n_bins=512, device=self.device)

        # get all vetrebrea images and move them to cpu
        # vertebrea_batches = []
        for vertebre in range(1, 8):
            v_batch = self.select_vertebre(image, mask, scale, vertebre, name, image_offset, mask_offset)
            if v_batch is not None:
                yield v_batch

                # vertebrea_batches.append([
                #     tensor.cpu()
                #     for tensor in v_batch
                # ])

        # # remove big arrays from cuda
        # del image, mask
        # gc.collect()
        # torch.cuda.empty_cache()

        # load images to cuda one-by-one before training
        # for v_batch in vertebrea_batches:
        #     v_batch = [
        #         tensor.cuda()
        #         for tensor in v_batch
        #     ]

        #     yield v_batch

    def get_vertebre_transform(self, vertebre, image_uid):
        metadata = self.segmentations_meta.loc[(image_uid, vertebre)]
        center = np.array(metadata["center"])

        # (x, y, z) = (right, forward, up)
        right = np.array([1, 0, 0])
        up = np.array(metadata["pca_components"][2])
        forward = np.cross(up, right)
        right = np.cross(forward, up)

        return center, right, forward, up

    def select_vertebre(self, image, mask, scale, vertebre, image_uid, image_offset, mask_offset):
        if self.segmentations_meta.loc[(image_uid, vertebre), "num_voxels"] < self.min_voxels:
            return None

        binary_mask = (mask == vertebre)
        side_vertebrea_mask = (mask > 0) & (mask != vertebre)

        size = self.image_size_mm * self.image_res_pix_per_mm
        center, right, forward, up = self.get_vertebre_transform(vertebre, image_uid)

        image_transform = (
            translation_tfm(-image_offset) @
            scale_tfm(1 / scale) @
            translation_tfm(center) @
            rotation_tfm(np.stack((right, forward, up), axis=1)) @
            scale_tfm(1 / self.image_res_pix_per_mm) @
            translation_tfm(-size / 2)
        )

        mask_transform = (
            translation_tfm(-mask_offset) @
            translation_tfm(center) @
            rotation_tfm(np.stack((right, forward, up), axis=1)) @
            scale_tfm(1 / self.image_res_pix_per_mm) @
            translation_tfm(-size / 2)
        )

        binary_mask = warp_affine3d(binary_mask.float(), mask_transform, 
                                    size, interpolation="nearest")
        side_vertebrea_mask = warp_affine3d(side_vertebrea_mask.float(), mask_transform, 
                                            size, interpolation="nearest")

        vertebre_image = warp_affine3d(image, image_transform, size)

        label = torch.tensor(self.labels.loc[image_uid, f"C{vertebre}"],
                             dtype=torch.float, device=self.device)

        return vertebre_image, binary_mask, side_vertebrea_mask, label



class VertebreaDataset2Res(VertebreaDataset):
    def __init__(
        self,
        images_dir: str | Path,
        segmentations_dir: str | Path,
        segmentation_meta_path: str | Path,
        labels_file: str | Path,
        image_names: List[str | Path] | None = None,
        num_workers: int = 2,
        prefetch_factor: int = 2,
        shuffle: bool = True,
        persistent_workers: bool = True,
        device: str | torch.device = "cuda",
        image_size_mm: Tuple[int] = (96, 96, 48),
        image_res_pix_per_mm: Tuple[int] = (1, 1, 1),
        patch_size_mm: Tuple[int] = (8, 8, 8),
        high_res_scale: Tuple[int] = (4, 4, 4), # = 0.25 mm / pixel
        max_patches_num = 200,
        min_voxels = 5000,
    ):
        super().__init__(
            images_dir,
            segmentations_dir,
            segmentation_meta_path,
            labels_file,
            image_names,
            num_workers,
            prefetch_factor,
            shuffle,
            persistent_workers,
            device,
            image_size_mm,
            image_res_pix_per_mm,
            max_patches_num,
            min_voxels,
        )

        self.patch_size_mm = np.asarray(patch_size_mm)
        self.high_res_pix_per_mm = np.asarray(high_res_scale)

        self.max_patches_num = max_patches_num


    def select_vertebre(self, image, mask, scale, vertebre, image_uid):
        vertebre_image, vertebre_mask, other_vertebrea_mask, label = \
            super().select_vertebre(image, mask, scale, vertebre, image_uid)

        size = self.image_size_mm * self.image_res_pix_per_mm
        center, right, forward, up = self.get_vertebre_transform(vertebre, image_uid)

        high_res_transform = (
            scale_tfm(1 / scale) @
            translation_tfm(center) @
            rotation_tfm(np.stack((right, forward, up), axis=1)) @
            scale_tfm(1 / self.high_res_pix_per_mm) @
            translation_tfm(-size * self.high_res_pix_per_mm / 2)
        )

        high_res_image = \
            warp_affine3d(image, high_res_transform, size * self.high_res_pix_per_mm)
        patches = \
            split_to_patches(high_res_image, self.patch_size_mm * self.high_res_pix_per_mm)

        patches_mask = torch.functional.F.max_pool3d(
            binary_mask.unsqueeze(0),
            kernel_size=tuple(self.patch_size_mm)
        ).squeeze(0).bool()

        idxs = patches_mask.flatten().nonzero()
        if idxs > self.max_patches_num:
            choice = np.random.choice(idxs, self.max_patches_num, replace=False)
            patches_mask[...] = 0
            patches_mask.flatten()[idxs] = 1

        patches = patches[patches_mask.flatten()]

        return vertebre_image, vertebre_mask, \
            other_vertebrea_mask, patches, patches_mask, label
       
        # metadata = self.segmentations_meta.loc[(image_uid, vertebre)]
        # # voxels = binary_mask.nonzero().float()

        # if metadata["num_voxels"] < self.min_voxels:
        #     return None

        # center = np.array(metadata["center"])

        # binary_mask = (mask == vertebre)
        # side_vertebrea_mask = (mask > 0) & (mask != vertebre)

        # # (x, y, z) = (right, forward, up)
        # right = np.array([1, 0, 0])
        # up = np.array(metadata["components"][2])
        # forward = np.cross(up, right)
        # right = np.cross(forward, up)

        # size = self.image_size_mm

        # low_res_transform = (
        #     scale_tfm(1 / scale) @
        #     translation_tfm(center) @
        #     rotation_tfm(np.stack((right, forward, up), axis=1)) @
        #     scale_tfm(1 / self.image_res_pix_per_mm) @
        #     translation_tfm(-size / 2)
        # )

        # high_res_transform = (
        #     scale_tfm(1 / scale) @
        #     translation_tfm(center) @
        #     rotation_tfm(np.stack((right, forward, up), axis=1)) @
        #     scale_tfm(1 / self.high_res_pix_per_mm) @
        #     translation_tfm(-size * self.high_res_pix_per_mm / 2)
        # )

        # mask_transform = (
        #     translation_tfm(center) @
        #     rotation_tfm(np.stack((right, forward, up), axis=1)) @
        #     translation_tfm(-size / 2)
        # )

        # high_res_image = warp_affine3d(image, high_res_transform, size * self.high_res_pix_per_mm)
        # patches = split_to_patches(high_res_image, self.patch_size_mm * self.high_res_pix_per_mm)

        # binary_mask = warp_affine3d(binary_mask.float(), mask_transform, 
        #                             size, interpolation="nearest")
        # side_vertebrea_mask = warp_affine3d(side_vertebrea_mask.float(), mask_transform, 
        #                                     size, interpolation="nearest")

        # patches_mask = torch.functional.F.max_pool3d(
        #     binary_mask.unsqueeze(0),
        #     kernel_size=tuple(self.patch_size_mm)
        # ).squeeze(0).bool()

        # # remove random patches if there is too many
        # if patches_mask.sum() > self.max_patches_num:
        #     idxs = patches_mask.flatten().nonzero()
        #     choice = torch.randperm(len(idxs))[:self.max_patches_num]
        #     idxs = idxs[choice]
        #     patches_mask[...] = 0
        #     patches_mask.flatten()[idxs] = 1

        # patches = patches[patches_mask.flatten()]
        # del high_res_image

        # low_res_image = warp_affine3d(image, low_res_transform, size)

        # label = torch.tensor(self.labels.loc[image_uid, f"C{vertebre}"],
        #                      dtype=torch.float, device=self.device)

        # return low_res_image, binary_mask, side_vertebrea_mask, patches, patches_mask, label

# class DblResDataset(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         images_dir: str | Path,
#         segmentations_dir: str | Path,
#         segmentation_meta_path: str | Path,
#         labels_file: str | Path,
#         image_names: List[str | Path] = None,
#         exclude: List[str | Path] = None,
#         # preprocess: Callable = None,
#         image_size_mm: Tuple[int] = (96, 96, 48),
#         patch_size_mm: Tuple[int] = (8, 8, 4),
#         high_res_scale: Tuple[int] = (4, 4, 4), # = 0.25 mm / pixel
#         device: str | torch.device = "cuda",
#     ):
#         self.images_dir = Path(images_dir)
#         self.segmentations_dir = Path(segmentations_dir)

#         self.segmentations_meta = pd.read_json(segmentation_meta_path).set_index("uid")
#         self.labels = pd.read_csv(labels_file).set_index("StudyInstanceUID")

#         if image_names is None:
#             image_names = [
#                 path.stem.removesuffix(".nii")
#                 for path in self.images_dir.glob("*.nii.gz")
#             ]

#         if exclude is not None:
#             exclude = set(exclude)
#             image_names = [
#                 name
#                 for name in image_names
#                 if name not in exclude
#             ]

#         self.image_names = image_names

#         vertebrea = []

#         for image_name in self.image_names:
#             if image_name not in self.segmentations_meta.index:
#                 continue

#             image_meta = self.segmentations_meta.loc[image_name].set_index("vertebre")
#             for vertebre, v_meta in image_meta.iterrows():
#                 if v_meta["num_voxels"] < 5000:
#                     continue

#                 vertebrea.append((image_name, vertebre))

#         self.vertebrea = vertebrea

#         self.image_size_mm = np.asarray(image_size_mm)
#         self.patch_size_mm = np.asarray(patch_size_mm)
#         self.high_res_scale = np.asarray(high_res_scale)
#         self.device = device

#     def __len__(self):
#         return len(self.vertebrea)

#     def __getitem__(self, index: int):
#         image_name, vertebre = self.vertebrea[index]

#         image_path = self.images_dir / f"{image_name}.nii.gz"
#         nii_image = nib.load(image_path)

#         seg_mask_path = self.segmentations_dir / f"{image_name}.nii.gz"
#         nii_seg_mask = nib.load(seg_mask_path)

#         scale = nii_image.affine.diagonal()[:3]

#         image = nii_image.get_fdata()
        
#         image = torch.tensor(image, dtype=torch.float32, device=self.device)
#         image = minmax_normalize(image)
#         image = clahe3d(image, size=64, clip_limit=2, n_bins=512)
        
#         mask = np.asanyarray(nii_seg_mask.dataobj)
#         mask = torch.tensor(mask, device=self.device)

#         binary_mask = (mask == vertebre)
#         voxels = binary_mask.nonzero().float()

#         center = voxels.mean(dim=0).cpu().numpy()

#         # (x, y, z) = (right, forward, up)
#         right = np.array([1, 0, 0])
#         up = voxels_pca(voxels)[2].cpu().numpy()
#         forward = np.cross(up, right)
#         right = np.cross(forward, up)

#         size = self.image_size_mm

#         low_res_transform = (
#             scale_tfm(1 / scale) @
#             translation_tfm(center) @
#             rotation_tfm(np.stack((right, forward, up), axis=1)) @
#             translation_tfm(-size / 2)
#         )

#         high_res_transform = (
#             scale_tfm(1 / scale) @
#             translation_tfm(center) @
#             rotation_tfm(np.stack((right, forward, up), axis=1)) @
#             scale_tfm(1 / self.high_res_scale) @
#             translation_tfm(-size * self.high_res_scale / 2)
#         )

#         mask_transform = (
#             translation_tfm(center) @
#             rotation_tfm(np.stack((right, forward, up), axis=1)) @
#             translation_tfm(-size / 2)
#         )

#         d = "auto"
#         high_res_image = warp_affine3d(image, high_res_transform, size * self.high_res_scale, device=d)
#         patches = split_to_patches(high_res_image, self.patch_size_mm * self.high_res_scale)

#         binary_mask = warp_affine3d(binary_mask.float(), mask_transform, size, interpolation="nearest", device=d)

#         patches_mask = torch.functional.F.max_pool3d(
#             binary_mask.unsqueeze(0),
#             kernel_size=tuple(self.patch_size_mm)
#         ).squeeze(0).bool()

#         patches = patches[patches_mask.flatten()]        
#         del high_res_image

#         low_res_image = warp_affine3d(image, low_res_transform, size, device=d)

#         return low_res_image, binary_mask.long(), patches, patches_mask
