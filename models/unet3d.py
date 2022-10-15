from audioop import reverse
from email.mime import base
from turtle import hideturtle

import torch
import torch.nn as nn
import tqdm
from einops import rearrange
from torchsummary import summary


class PlanarConv(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, batchnorm=False):
        super(PlanarConv, self).__init__()

        if hidden_channels is None:
            hidden_channels = out_channels // 3

        self.conv_axial = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv_coronal = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv_sagittal = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)

        self.batchnorm = batchnorm
        if batchnorm:
             self.bn_layer = nn.BatchNorm3d(hidden_channels * 3)

        self.linear = nn.Conv3d(hidden_channels * 3, out_channels, kernel_size=1)
        
    def forward(self, x):
        """
        Args:
            x (tensor): tensor of shape [batch, in_channels, SI, AP, LR]

        Returns:
            (tensor): shape [batch, out_channels, SI, AP, LR]
        """
        batch_size = x.shape[0]

        x1 = self.conv_axial(rearrange(x, "b c i j k -> (b k) c i j"))
        x2 = self.conv_coronal(rearrange(x, "b c i j k -> (b j) c i k"))
        x3 = self.conv_sagittal(rearrange(x, "b c i j k -> (b i) c k j"))

        x = torch.cat([
            rearrange(x1, "(b k) c i j -> b c i j k", b = batch_size),
            rearrange(x2, "(b j) c i k -> b c i j k", b = batch_size),
            rearrange(x3, "(b i) c j k -> b c i j k", b = batch_size),
        ], dim=1)

        x = torch.relu(x)
        
        if self.batchnorm:
            x = self.bn_layer(x)

        x = self.linear(x)

        return x

class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1, batch_norm=False):
        super(ConvLayer, self).__init__(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(out_channels) if batch_norm else nn.Identity(),
        )


class Unet3D(nn.Module):
    def __init__(self, input_dim=1, output_dim=8, n_layers=4, base_channels=8, batch_norm=True):
        super(Unet3D, self).__init__()

        self.n_layers = n_layers
        self.base_channels = base_channels

        self.downsamples = nn.ModuleList()

        for i in range(n_layers):
            in_channels = base_channels * 2**i
            out_channels = in_channels * 2
            
            if i == 0:
                in_channels = input_dim

            layer = ConvLayer(in_channels, out_channels // 2, batch_norm=batch_norm)
            self.downsamples.append(layer)

            layer = ConvLayer(out_channels // 2, out_channels, stride=2, batch_norm=batch_norm)
            self.downsamples.append(layer)

        hid_dim = base_channels * 2**n_layers
        self.intermedate_conv = ConvLayer(hid_dim, hid_dim, batch_norm=batch_norm)

        self.upsample = nn.ModuleList()

        for i in reversed(range(n_layers)):
            in_channels = base_channels * 2**(i + 1)
            out_channels = in_channels // 2
            
            layer = ConvLayer(in_channels, out_channels, batch_norm=batch_norm)
            self.upsample.append(layer)

            layer = ConvLayer(out_channels * 2, out_channels, batch_norm=batch_norm)
            self.upsample.append(layer)

        self.output_conv = nn.Conv3d(base_channels, output_dim, 1)

    def encode(self, x):
        saved_x = []

        for i in range(self.n_layers):
            conv = self.downsamples[2*i]
            down = self.downsamples[2*i + 1]

            x = conv(x)
            saved_x.append(x)
            x = down(x)

        return x, saved_x

    def decode(self, x, saved_x):
        x = self.intermedate_conv(x)

        for i in range(self.n_layers):
            up = self.upsample[2*i]
            conv = self.upsample[2*i + 1]

            x = up(x)
            x = torch.functional.F.interpolate(x, scale_factor=2)
            x = torch.cat([x, saved_x[-i-1]], dim=1)
            x = conv(x)

        x = self.output_conv(x)

        return x

    def forward(self, x):
        x, saved_x = self.encode(x)
        x = self.decode(x, saved_x)

        return x

class Unet3DIterative(Unet3D):
    def __init__(self, input_dim=2, output_dim=1, n_labels=8, n_layers=4, base_channels=8, batch_norm=True):
        super().__init__(input_dim, output_dim, n_layers, base_channels, batch_norm)

        self.label_decoder = nn.Sequential(
            ConvLayer(128, 64, stride=2),
            ConvLayer(64, 32, stride=2),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )

        self.label_classifier = nn.Linear(32, n_labels)
        self.completness_classifier = nn.Linear(32, 1)

    def forward(self, x, instance_mask):
        x = torch.cat((x, instance_mask), dim=1)

        x, saved_x = self.encode(x)

        y = self.label_decoder(x)
        label_logits = self.label_classifier(y)
        completness = self.completness_classifier(y)

        x = self.decode(x, saved_x)

        return x, label_logits, completness


if __name__ == "__main__":
    model = Unet3DIterative().cuda()
    print(model)
    # summary(model, (1, 128, 128, 128), batch_size=2, device='cuda')

    def inf():
        while True:
            yield 0

    for i in tqdm.tqdm(inf()):
        dummy_image = torch.randn(2, 1, 128, 128, 128).cuda()
        dummy_mask = torch.ones(2, 1, 128, 128, 128).cuda()
        output_mask, label_logits, completness = model(dummy_image, dummy_mask)
        (output_mask.sum() + label_logits.sum() + completness.sum()).backward()