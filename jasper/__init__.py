import torch.nn as nn
import torch
import os
from .model import JasperEncoderDecoder as _Jasper
from .jasper_augs import JasperAugs as _Aug

__all__ = ['jasper', 'to_mel']


def jasper() -> nn.Module:
    network = _Jasper()
    checkpoint = torch.load(os.path.join('checkpoints', 'jasper', 'base.pt'), map_location="cpu")['state_dict']
    network.load_state_dict(checkpoint, strict=False)
    return network


def to_mel() -> nn.Module:
    network = _Aug()
    checkpoint = torch.load(os.path.join('checkpoints', 'jasper', 'jasper_aug.pt'), map_location="cpu")
    network.load_state_dict(checkpoint, strict=False)
    return network
