import os

import torch
from torch import nn as nn
from torchvision.models import resnet50, resnet18

from models.jasper.jasper_augs import JasperAugs
from models.z_net import ZNet
from models.jasper import JasperEncoderDecoder, GreedyCTCDecoder
from models.vae import VAE


def _parallel_cuda(func):
    def to_parallel() -> nn.Module:
        model = func()
        model = nn.DataParallel(model.cuda())
        return model

    return to_parallel


@_parallel_cuda
def resnet50_untrained() -> nn.Module:
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 50)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return model


@_parallel_cuda
def resnet18_untrained() -> nn.Module:
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(512, 50)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return model


def resnet18_audio(base_path: str = 'checkpoints', dataset: str = 'esc50', architecture: str = 'resnet18',
                   checkpoint_no: int = 0) -> nn.Module:
    model = resnet18_untrained()
    path = os.path.join(base_path, dataset, '{}_{}.pt'.format(architecture, checkpoint_no))
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    return model.module


def resnet50_audio(base_path: str = 'checkpoints', dataset: str = 'esc50') -> nn.Module:
    model = resnet50_untrained()
    path = os.path.join(base_path, dataset, 'resnet50.pt')
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    return model.module


@_parallel_cuda
def jasper() -> nn.Module:
    model = JasperEncoderDecoder()
    checkpoint = torch.load(os.path.join('checkpoints', 'jasper', 'base.pt'), map_location="cpu")['state_dict']
    model.load_state_dict(checkpoint, strict=False)
    return model


def jasper_augs() -> nn.Module:
    model = JasperAugs()
    checkpoint = torch.load(os.path.join('checkpoints', 'jasper', 'jasper_aug.pt'), map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    return model


@_parallel_cuda
def greedy_decoder() -> nn.Module:
    return GreedyCTCDecoder()


def z_net() -> nn.Module:
    model = nn.DataParallel(ZNet())
    checkpoint = torch.load(os.path.join('checkpoints', 'z_net', 'base.pt'))
    model.load_state_dict(checkpoint)
    return model.module


def vae() -> nn.Module:
    model = nn.DataParallel(VAE(torch.zeros((1, 1, 16000))))
    checkpoint = torch.load(os.path.join('checkpoints', 'vae', 'base3.pt'))
    model.load_state_dict(checkpoint)
    return model.module


def encoder() -> nn.Module:
    vae_model = vae()
    return vae.encoder


def decoder() -> nn.Module:
    vae_model = vae()
    return vae.decoder
