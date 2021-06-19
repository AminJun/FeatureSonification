import numpy as np
import torch
import torch.nn as nn


class AnyCoderBlock(nn.Sequential):
    def __init__(self, cls, in_ch: int, out_ch: int, kernel: int, stride: int, last: bool = False):
        if last:
            layers = [cls(in_ch, out_ch, kernel, stride), nn.Tanh()]
        else:
            layers = [cls(in_ch, out_ch, kernel, stride), nn.BatchNorm1d(out_ch), nn.Tanh()]
        super().__init__(*layers)

    def forward(self, input: torch.tensor) -> torch.tensor:
        #print(input.shape)
        return super(AnyCoderBlock, self).forward(input)


class ConvEncoder(nn.Sequential):
    def __init__(self, sizes: [int], kernel: [int], stride: [int]):
        layers = [AnyCoderBlock(nn.Conv1d, in_s, out_s, fil_s, str_s) for in_s, out_s, fil_s, str_s in
                  zip(sizes[:-1], sizes[1:], kernel, stride)]
        super().__init__(*layers)


class ConvDecoder(nn.Sequential):
    def __init__(self, sizes: [int] = None, kernel: [int] = None, stride: [int] = None):
        layers = [AnyCoderBlock(nn.ConvTranspose1d, in_s, out_s, fil_s, str_s) for in_s, out_s, fil_s, str_s in
                  zip(sizes[:-2], sizes[1:-1], kernel[:-1], stride[:-1])]
        layers += [AnyCoderBlock(nn.ConvTranspose1d, sizes[-2], sizes[-1], kernel[-1], stride[-1], last=True)]
        super().__init__(*layers)


class FCBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, last: bool = False):
        layers = [nn.Linear(in_ch, out_ch)] if last else [nn.Linear(in_ch, out_ch), nn.BatchNorm1d(out_ch), nn.Tanh()]
        super().__init__(*layers)


class FCEncoder(nn.Module):
    def __init__(self, sizes: [int]):
        super().__init__()
        self.layers = nn.Sequential(*[FCBlock(in_c, out_c) for in_c, out_c in zip(sizes[:-2], sizes[1:-1])])
        self.mu = FCBlock(sizes[-2], sizes[-1], last=True)
        self.log_var = FCBlock(sizes[-2], sizes[-1], last=True)

    def forward(self, x: torch.tensor) -> (torch.tensor, torch.tensor, torch.tensor):
        x = self.layers(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        z = mu + log_var.exp().sqrt() * torch.randn_like(log_var)
        return z, mu, log_var


class FCDecoder(nn.Sequential):
    def __init__(self, sizes: [int]):
        layers = [FCBlock(in_c, out_c) for in_c, out_c in zip(sizes[:-1], sizes[1:])]
        super().__init__(*layers)


class Encoder(nn.Module):
    def __init__(self, sample_input: torch.Tensor, conv_size: [int], kernel: [int], stride: [int], fc_size: [int]):
        super().__init__()
        self.conv = ConvEncoder(conv_size, kernel, stride)
        self.special_size = self.conv(sample_input).size()[1:]
        self.fc_size = [int(np.prod(self.special_size))] + fc_size
        self.fc = FCEncoder(self.fc_size)

    def forward(self, x: torch.tensor) -> (torch.tensor, torch.tensor, torch.tensor):
        x = self.conv(x).view(x.size(0), -1)
        return self.fc(x)

    def get_sizes(self) -> ([int], torch.Size):
        return self.fc_size, self.special_size


class Decoder(nn.Module):
    def __init__(self, special_size: torch.Size, conv_size: [int], kernel: [int], stride: [int], fc_size: [int]):
        super().__init__()
        self.fc = FCDecoder(fc_size)
        self.conv = ConvDecoder(conv_size, kernel, stride)
        self.special_size = special_size

    def forward(self, z: torch.tensor) -> torch.tensor:
        x = self.fc(z)
        x = x.view(x.size(0), *list(self.special_size))
        return self.conv(x)


class VAE(nn.Module):
    def __init__(self, sample_input: torch.tensor, conv_size: [int] = None, kernel: [int] = None, stride: [int] = None,
                 fc_size: [int] = None):
        super(VAE, self).__init__()
        #conv_size = conv_size or [128, 64, 32, 16, 8]
        conv_size = conv_size or [1, 2, 4, 8, 16, 32] 
        #      True size         [2:16000, 4:4000, 8:1000, 16:256, 32:64, 32:16]
        kernel = kernel or [5, 8, 8, 8, 8]
        stride = stride or [1, 4, 4, 4, 4]
        fc_size = fc_size or [512, 256, 32]
        self.encoder = Encoder(sample_input, conv_size, kernel, stride, fc_size)
        fc_size, special_size = self.encoder.get_sizes()
        print(fc_size)
        self.decoder = Decoder(special_size, conv_size[::-1], kernel[::-1], stride[::-1], fc_size[::-1])

    def forward(self, x: torch.tensor) -> (torch.tensor, torch.tensor, torch.tensor, torch.tensor):
        x = x.squeeze(1) if x.dim() == 4 else x
        z, mu, log_var = self.encoder(x)
        x = self.decoder(z)
        return x, z, mu, log_var
