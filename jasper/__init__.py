import torch.nn as nn
import torch
import os
from .model import JasperEncoderDecoder as _Jasper
from .jasper_augs import JasperAugs as _Aug

__all__ = ['Jasper']


def jasper() -> nn.Module:
    network = _Jasper()
    checkpoint = torch.load(os.path.join('checkpoints', 'jasper.pt'), map_location="cpu")['state_dict']
    network.load_state_dict(checkpoint, strict=False)
    return network


def to_mel() -> nn.Module:
    network = _Aug()
    checkpoint = torch.load(os.path.join('checkpoints', 'to_mel.pt'), map_location="cpu")
    network.load_state_dict(checkpoint, strict=False)
    return network


class Jasper(nn.Module):
    def __init__(self, classifier: nn.Module = None, augmentations: nn.Module = None):
        super().__init__()
        if classifier is None:
            classifier = jasper()
        if augmentations is None:
            augmentations = to_mel()
        self.augmentations = augmentations
        self.classifier = classifier.module if isinstance(classifier, nn.DataParallel) else classifier

    def forward(self, *inputs: [torch.Tensor]) -> [torch.Tensor]:
        augmented = self.augmentations(*inputs)
        return self.classifier(augmented)
