import torch
import torch.nn as nn

from datasets import SpeechCommands
from datasets.audio.speech_commands import SPEECH_COMMANDS_AUGs


class ZBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4, stride=2):
        super().__init__()
        self.layers = nn.Sequential(nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride),  nn.Sigmoid(),nn.BatchNorm1d(out_channels),)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # print(x.shape)
        return self.layers(x)


class ZNet(nn.Module):
    def __init__(self):
        super().__init__()
        # structure = [32, 1024, 32]
        # structure = [64, 1]
        structure = [1, 1]
        structure = structure[:-1] + structure[::-1]
        layers = [ZBlock(in_c, out_c) for in_c, out_c in zip([1] + structure, structure + [1])]
        self.layers = nn.Sequential(nn.BatchNorm1d(num_features=1), *layers)

    def forward(self, z: torch.tensor) -> torch.tensor:
        # z = z.unsqueeze(3).repeat(1, 1, 1, 16).view(z.size()[:-1] + (-1,))
        return self.layers(z) #  * 2. - 1.


class ZNetLoss(nn.Module):
    def __init__(self, check_dims: bool = True, compare_features: bool = False):
        super().__init__()
        self.check_dims = check_dims
        self.augs = SPEECH_COMMANDS_AUGs
        self.feature = compare_features

    def forward(self, output: torch.tensor, true_x: torch.tensor) -> torch.tensor:
        assert not self.check_dims or self.check(output, true_x)

        if self.feature:
            true_x = self.augs(true_x)
            output = self.augs(output)

        dims = true_x.dim()
        for d in range(true_x.dim()):
            output = output.narrow(d, 0, true_x.size(d))
        return (output - true_x).view(true_x.size(0), -1).norm(dim=1).mean()

    def check(self, output: torch.tensor, true_x: torch.tensor) -> bool:
        print('checking dims: ===')
        assert true_x.dim() == output.dim()
        for d in range(true_x.dim()):
            print(true_x.size(d), output.size(d))
            assert true_x.size(d) <= output.size(d) <= 2 * true_x.size(d)
        self.check_dims = False
        print('Dims were correct ===')
        return True
