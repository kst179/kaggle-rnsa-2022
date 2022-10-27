import torch
from torch import nn
from typing import List, Tuple, Type, Literal
from collections import OrderedDict
import numpy as np


class Iota:
    def __init__(self, start=0):
        self.i = start

    def __call__(self):
        i, self.i = self.i, self.i+1
        return i
        

class SqueezeExcitation(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        squeeze_channels: int,
        activation: type[nn.Module] = nn.SiLU,
    ):
        super().__init__(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, squeeze_channels, kernel_size=1),
            activation(inplace=True),
            nn.Conv3d(squeeze_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        scale = super().forward(input) 
        return scale * input

class ConvNormActiv(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        batch_norm: bool = True,
        bn_momentum: float = 0.1,
        dropout: float | bool = 0,
        activation: Type[nn.Module] = nn.SiLU,
        padding_mode: Literal["constant", "reflect"] = "constant",
    ):
        layers = []

        layers.append( nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, 
                                 stride=stride, padding=kernel_size // 2, padding_mode=padding_mode) )
        if batch_norm: layers.append( nn.BatchNorm3d(out_channels, momentum=bn_momentum) )
        if dropout: layers.append( nn.Dropout(dropout, inplace=True) )
        layers.append( activation(inplace=True) )

        super().__init__(*layers)


class InvResidualBlock(nn.Sequential):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        expansion_factor:int = 6, 
        stride: int = 1, 
        batch_norm: bool = False, 
        bn_momentum: float = 0.1,
        dropout: float | bool = 0,
        activation: Type[nn.Module] = nn.SiLU,
        padding_mode: Literal["zeros", "reflect"] = "zeros",
        squeeze_excitation: bool = True,
        stochastic_depth_prob: float = 0,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion_factor = expansion_factor
        self.stride = stride
        self.batch_norm = batch_norm
        self.stochastic_depth_prob = stochastic_depth_prob

        hid_channels = in_channels * expansion_factor

        layers = []

        if hid_channels != in_channels:
            layers.append( nn.Conv3d(in_channels, hid_channels, kernel_size=1, bias=False) )
            if batch_norm: layers.append( nn.BatchNorm3d(hid_channels, momentum=bn_momentum) )
            if dropout: layers.append( nn.Dropout(dropout, inplace=True) )
            layers.append( activation(inplace=True) )

        layers.append( nn.Conv3d(hid_channels, hid_channels, kernel_size=3, stride=stride, 
                                 padding=1, groups=hid_channels, bias=False, padding_mode=padding_mode) )
        if batch_norm: layers.append( nn.BatchNorm3d(hid_channels, momentum=bn_momentum) )
        if dropout: layers.append( nn.Dropout(dropout, inplace=True) )
        layers.append( activation(inplace=True) )

        if squeeze_excitation:
            squeeze_channels = max(1, in_channels // 4)
            layers.append( SqueezeExcitation(hid_channels, squeeze_channels, activation) )
        
        layers.append( nn.Conv3d(hid_channels, out_channels, kernel_size=1, bias=False) )
        if batch_norm: layers.append( nn.BatchNorm3d(out_channels, momentum=bn_momentum) )
        if dropout: layers.append( nn.Dropout(dropout, inplace=True) )
        layers.append( activation(inplace=True) )

        super().__init__(*layers)

    def forward(self, input):
        output = super().forward(input)

        if self.stride == 1 and self.in_channels == self.out_channels:
            if self.stochastic_depth_prob and self.training:
                batch_size = input.shape[0]
                noise = torch.zeros((batch_size, *input.shape[1:]), device=input.device)
                noise.bernoulli_(1 - self.stochastic_depth_prob)
                noise.div_(1 - self.stochastic_depth_prob)

                return input + output * noise

            return input + output

        return output


class EfficientNet3d(nn.Sequential):
    def __init__(
        self, 
        in_channels: int = 1,
        num_classes: int = 1,
        batch_norm: bool = True,
        bn_momentum: float = 0.1,
        dropout: float | bool = 0,
        activation: Type[nn.Module] = nn.SiLU,
        padding_mode: Literal["zeros", "reflect"] = "zeros",
        stochastic_depth_prob: float = 0.2
    ):
        shared_params = dict(
            batch_norm=batch_norm,
            bn_momentum=bn_momentum,
            dropout=dropout,
            activation=activation,
            padding_mode=padding_mode,
        )

        layers = []
        
        # input size [1, 96, 96, 96]
        # layers.append( nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode) )
        # if batch_norm: layers.append( nn.BatchNorm3d(32, momentum=bn_momentum) )
        # if dropout: layers.append( nn.Dropout(dropout, inplace=True) )
        # layers.append( activation(inplace=True) )
        layers.append( ConvNormActiv(in_channels, 32, kernel_size=3, stride=2, **shared_params) )

        iota = Iota(start=1)
        num_layers = 17
        sd_prob = lambda: stochastic_depth_prob * iota() / num_layers

        _ = nn.Sequential
        layers.extend([
            # [32, 48, 48, 48]
            _(InvResidualBlock(32, 16, expansion_factor=1, stride=1, stochastic_depth_prob=sd_prob(), **shared_params)),
            # [16, 48, 48, 48]
            _(InvResidualBlock(16, 24, expansion_factor=6, stride=2, stochastic_depth_prob=sd_prob(), **shared_params),
              InvResidualBlock(24, 24, expansion_factor=6, stride=1, stochastic_depth_prob=sd_prob(), **shared_params)),
            # [24, 24, 24, 24]
            _(InvResidualBlock(24, 32, expansion_factor=6, stride=2, stochastic_depth_prob=sd_prob(), **shared_params),
             InvResidualBlock(32, 32, expansion_factor=6, stride=1, stochastic_depth_prob=sd_prob(), **shared_params),
             InvResidualBlock(32, 32, expansion_factor=6, stride=1, stochastic_depth_prob=sd_prob(), **shared_params)),
            # [32, 12, 12, 12]
            _(InvResidualBlock(32, 64, expansion_factor=6, stride=2, stochastic_depth_prob=sd_prob(), **shared_params),
             InvResidualBlock(64, 64, expansion_factor=6, stride=1, stochastic_depth_prob=sd_prob(), **shared_params),
             InvResidualBlock(64, 64, expansion_factor=6, stride=1, stochastic_depth_prob=sd_prob(), **shared_params),
             InvResidualBlock(64, 64, expansion_factor=6, stride=1, stochastic_depth_prob=sd_prob(), **shared_params)),
            # [64, 6, 6, 6]
            _(InvResidualBlock(64, 96, expansion_factor=6, stride=2, stochastic_depth_prob=sd_prob(), **shared_params),
              InvResidualBlock(96, 96, expansion_factor=6, stride=1, stochastic_depth_prob=sd_prob(), **shared_params),
              InvResidualBlock(96, 96, expansion_factor=6, stride=1, stochastic_depth_prob=sd_prob(), **shared_params)),
            # [96, 3, 3, 3]
            _(InvResidualBlock(96, 160, expansion_factor=6, stride=2, stochastic_depth_prob=sd_prob(), **shared_params),
              InvResidualBlock(160, 160, expansion_factor=6, stride=1, stochastic_depth_prob=sd_prob(), **shared_params),
              InvResidualBlock(160, 160, expansion_factor=6, stride=1, stochastic_depth_prob=sd_prob(), **shared_params)),
            # [160, 1, 1, 1]
            _(InvResidualBlock(160, 320, expansion_factor=6, stride=1, stochastic_depth_prob=sd_prob(), **shared_params)),
        ])

        # layers.append( nn.Conv3d(320, 1280, kernel_size=1) )
        # if batch_norm: layers.append( nn.BatchNorm3d(1280, momentum=bn_momentum) )
        # if dropout: layers.append( nn.Dropout(dropout, inplace=True) )
        #     activation(inplace=True),

        layers.extend([
            ConvNormActiv(320, 1280, kernel_size=1, stride=1, **shared_params),

            # [1280, 6, 6, 6]
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            # [1280,]
            nn.Linear(1280, num_classes),
        ])


        super().__init__(*layers)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return nn.Sequential(OrderedDict(list(self._modules.items())[item]))

        return super().__getitem__(item)


# class MobileDecoder(nn.Sequential):
#     def __init__(
#         self, 
#         num_classes: int = 1,
#         batch_norm: bool = False,
#         bn_momentum: float = 0.1,
#         dropout: float | bool = 0,
#         activation: Type[nn.Module] = nn.SiLU,
#         padding_mode: Literal["zeros", "reflect"] = "zeros",
#         stochastic_depth_prob: float = 0.2,
#     ):
#         shared_params = dict(
#             batch_norm=batch_norm,
#             bn_momentum=bn_momentum,
#             dropout=dropout,
#             activation=activation,
#             padding_mode=padding_mode,
#         )

#         iota = Iota(start=7)
#         num_layers = 17
#         sd_prob = lambda: stochastic_depth_prob * iota() / num_layers

#         layers = []

#         layers.extend([
#             # [32, 48, 48, 48]
#             InvResidualBlock(32, 64, expansion_factor=6, stride=2, stochastic_depth_prob=sd_prob(), **shared_params),
#             InvResidualBlock(64, 64, expansion_factor=6, stride=1, stochastic_depth_prob=sd_prob(), **shared_params),
#             InvResidualBlock(64, 64, expansion_factor=6, stride=1, stochastic_depth_prob=sd_prob(), **shared_params),
#             InvResidualBlock(64, 64, expansion_factor=6, stride=1, stochastic_depth_prob=sd_prob(), **shared_params),
#             # [64, 24, 24, 24]
#             InvResidualBlock(64, 96, expansion_factor=6, stride=2, stochastic_depth_prob=sd_prob(), **shared_params),
#             InvResidualBlock(96, 96, expansion_factor=6, stride=1, stochastic_depth_prob=sd_prob(), **shared_params),
#             InvResidualBlock(96, 96, expansion_factor=6, stride=1, stochastic_depth_prob=sd_prob(), **shared_params),
#             # [96, 12, 12, 12]
#             InvResidualBlock(96, 160, expansion_factor=6, stride=2, stochastic_depth_prob=sd_prob(), **shared_params),
#             InvResidualBlock(160, 160, expansion_factor=6, stride=1, stochastic_depth_prob=sd_prob(), **shared_params),
#             InvResidualBlock(160, 160, expansion_factor=6, stride=1, stochastic_depth_prob=sd_prob(), **shared_params),
#             # [160, 6, 6, 6]
#             InvResidualBlock(160, 320, expansion_factor=6, stride=1, stochastic_depth_prob=sd_prob(), **shared_params),

#             nn.Conv3d(320, 1280, kernel_size=1),
#         ])

#         if batch_norm: layers.append( nn.BatchNorm3d(1280, momentum=bn_momentum) )
#         if dropout: layers.append( nn.Dropout(dropout, inplace=True) )

#         layers.extend([
#             activation(inplace=True),
#             # [1280, 6, 6, 6]
#             nn.AdaptiveAvgPool3d(1),
#             nn.Flatten(),
#             # [1280,]
#             nn.Linear(1280, num_classes),
#         ])

#         super().__init__(*layers)


class EfficientNet3dClassifier(nn.Module):
    def __init__(
        self,
        batch_norm: bool = True,
        bn_momentum: float = 0.1,
        dropout: float = 0,
        activation: Type[nn.Module] = nn.SiLU,
    ):
        self.net = EfficientNet3d(
            in_channels=3, 
            num_classes=1, 
            batch_norm=batch_norm,
            bn_momentum=bn_momentum,
            dropout=dropout,
            activation=activation,
        )

    def forward(
        self,
        vertebre_image: torch.Tensor,
        target_vertabre_mask: torch.Tensor,
        other_vertabrea_mask: torch.Tensor,
    ):
        input = torch.cat((
            vertebre_image.unsqueeze(1),
            target_vertabre_mask.unsqueeze(1),
            other_vertabrea_mask.unsqueeze(1),
        ), dim=1)

        return self.net(input)


class EfficientUNet3d(nn.Module):
    def __init__(
        self,
        batch_norm: bool = True,
        bn_momentum: float = 0.1,
        dropout: float | bool = 0,
        activation: Type[nn.Module] = nn.SiLU,
        padding_mode: Literal["zeros", "reflect"] = "zeros",
    ):
        super().__init__()

        shared_params = dict(
            batch_norm=batch_norm,
            bn_momentum=bn_momentum,
            dropout=dropout,
            activation=activation,
            padding_mode=padding_mode,
        )

        # block num    ||  0 |  1 |  2 |  3 |  4 |  5 |   6 |   7 || 
        # layers num   ||    |  1 |  2 |  3 |  4 |  3 |   3 |   1 ||
        # input size   || 96 | 48 | 48 | 24 | 12 |  6 |   3 |   1 ||
        # out channels || 32 | 16 | 24 | 32 | 64 | 96 | 160 | 320 ||
        # unet block   ||       0      |  1 |  2 |     3    |
        backbone = EfficientNet3d(in_channels=2, num_classes=1 + 7 + 1)
        # print(backbone[0:3])

        self.dwn_blocks = nn.ModuleList([
            backbone[0:3],
            backbone[3:4],
            backbone[4:5],
            backbone[5:7],
        ])

        self.classifier = backbone[7:]

        self.intermediate_layer = ConvNormActiv(160, 160, kernel_size=1, stride=1, **shared_params)

        _ = nn.Sequential
        self.up_blocks = nn.ModuleList([
            _(InvResidualBlock(in_channels=160, out_channels=64, expansion_factor=6, **shared_params),
              nn.Upsample(scale_factor=3, mode="nearest")),

            _(InvResidualBlock(in_channels=64 * 2, out_channels=32, expansion_factor=6, **shared_params),
              nn.Upsample(scale_factor=2, mode="nearest")),

            _(InvResidualBlock(in_channels=32 * 2, out_channels=24, expansion_factor=6, **shared_params),
              nn.Upsample(scale_factor=2, mode="nearest")),

            _(InvResidualBlock(in_channels=24 * 2, out_channels=1, expansion_factor=6, **shared_params),
              nn.Upsample(scale_factor=4, mode="trilinear")),
        ])
    
    def forward(self, image, instance_mask):
        x1 = self.dwn_blocks[0](torch.cat((image, instance_mask), dim=1))
        x2 = self.dwn_blocks[1](x1)
        x3 = self.dwn_blocks[2](x2)
        x4 = self.dwn_blocks[3](x3)
        x4 = self.intermediate_layer(x4)
        y = self.classifier(x4)

        label_logits, completness_logit = y[:, :-1], y[:, -1:]

        x4 = self.up_blocks[0](x4)
        x3 = self.up_blocks[1](torch.cat((x3, x4), dim=1))
        x2 = self.up_blocks[2](torch.cat((x2, x3), dim=1))
        image = self.up_blocks[3](torch.cat((x1, x2), dim=1))

        return image, label_logits, completness_logit
            

# class MobileNet3d2Res(nn.Module):
#     def __init__(
#         self,
#         high_res_scale: Tuple[int] = (4, 4, 4),
#         batch_norm: bool = False,
#         dropout: float = 0,
#         activation: Type[nn.Module] = nn.SiLU,
#     ):
#         super().__init__()
        
#         shared_params = dict(
#             batch_norm=batch_norm,
#             dropout=dropout,
#             activation=activation,
#         )

#         self.high_res_scale = high_res_scale

#         self.low_res_encoder = EfficientNet3d(in_channels=3, bn_momentum=0.01, **shared_params)
#         self.high_res_encoder = EfficientNet3d(in_channels=1, bn_momentum=0.1, **shared_params)
#         self.decoder = MobileDecoder(num_classes=1, bn_momentum=0.01, **shared_params)

#     def forward(
#         self,
#         low_res_image: torch.Tensor,
#         target_vertabre_mask: torch.Tensor,
#         side_vertabrea_mask: torch.Tensor,
#         high_res_patches: torch.Tensor,
#         patches_mask: torch.Tensor,
#         return_contexts: bool = False,
#     ):
#         """
#         Calculating logits of vertebrea fraction probability given the vertabre voxel image
#         in two resolutions: low-res for global context extraction and several high-res patches
#         defined by segmentation mask of vertabrea.

#         Args:
#             low_res_image (Tensor[float32]): 
#                 size [B, 96, 96, 48], low-res (1 mm) image used for calculation 
#                 of global context of voxel image

#             target_vertebre_mask (Tensor[bool]):
#                 size [B, 96, 96, 48], binary mask indicating target vertebre

#             side_vertebrea_mask (Tensor[bool]):
#                 size [B, 96, 96, 48], binary mask indicating other vertabrea

#             high_res_patches (Tensor[float32]):
#                 size [P, 32, 32, 32], high-res (0.25 mm) image patches 
#                 for local context calculation. The exact number of patches (batch size) 
#                 is sum of patches number in different images in the batch
#                 P = sum(<num_patches_in_image i> for i = 1..B)

#             patches_masks (Tensor[bool]):
#                 size [B, 12, 12, 6], boolean masks indicating 
#                 non-zero patches in image
#         """
#         # add single channel to tensors if needed 
#         low_res_image = low_res_image.unsqueeze(1)
#         target_vertabre_mask = target_vertabre_mask.unsqueeze(1)
#         side_vertabrea_mask = side_vertabrea_mask.unsqueeze(1)
#         high_res_patches = high_res_patches.unsqueeze(1)
        
#         # [B, 3, (96, 96, 48)] -- encode --> [B, 32, (12, 12, 6)]
#         global_context = self.low_res_encoder(
#             torch.cat((
#                 low_res_image,
#                 target_vertabre_mask,
#                 side_vertabrea_mask,
#             ), dim=1)
#         )

#         # [B, 32, (12, 12, 6)] -- upscale --> [B, 32, (48, 48, 24)]
#         global_context = torch.functional.F.interpolate(
#             global_context, 
#             scale_factor=self.high_res_scale
#         )

#         # [P, 1, (32, 32, 32)] -- encode --> [P, 32, (4, 4, 4)]
#         patch_features = self.high_res_encoder(high_res_patches)

#         batch_size = patches_mask.shape[0]
#         channels = patch_features.shape[1]
#         grid_size = np.array(patches_mask.shape[1:])
#         patch_size = np.array(patch_features.shape[2:])

#         # [B, (12, 12, 6), 32, (4, 4, 4)] -> [B, 32, (48, 48, 24)]
#         #  \----mask----/  \-patch_fs--/
#         local_context = torch.zeros((batch_size, *grid_size, channels, *patch_size), 
#                                     device=patch_features.device)
#         local_context[patches_mask] = patch_features

#         #     (b, gx, gy, gz, c, px, py, pz) -> 
#         #  -> (b, c, gx*px, gy*py, gz*pz)
#         local_context = (
#             local_context
#             .permute(0, 4, 1, 5, 2, 6, 3, 7)
#             .reshape(batch_size, channels, *(grid_size * patch_size))
#         )

#         # [B, 32, (48, 48, 24)] -- decode --> [B, 1]
#         proba_logit = self.decoder(global_context + local_context)

#         if return_contexts:
#             return proba_logit, global_context, local_context

#         return proba_logit