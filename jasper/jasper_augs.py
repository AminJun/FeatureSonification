import torch
import torch.nn as nn
import librosa
from torch import tensor


class MyNormalize(nn.Module):
    def __init__(self, momentum: float = 0.1, size: int = 64):
        super().__init__()
        self.momentum = momentum
        self.running_mean = nn.Parameter(torch.zeros(1, size, 1))
        self.running_std = nn.Parameter(torch.zeros(1, size, 1))

    def forward(self, x: tensor, seq_len: tensor) -> torch.tensor:
        if self.training:
            mean, std = self.per_batch(x, seq_len)
            self.running_mean.data = (1. - self.momentum) * self.running_mean.data + self.momentum * mean.data
            self.running_std.data = (1. - self.momentum) * self.running_std.data + self.momentum * std.data
        else:
            mean, std = self.running_mean, self.running_std
        return (x - mean) / (std + 1e-5)

    @staticmethod
    def per_batch(x: tensor, seq_len: tensor) -> (tensor, tensor):
        index = (torch.arange(x.size(-1)).cuda() < seq_len)
        cat_x = x.transpose(0, 1)[:, index]
        x_mean, x_std = cat_x.mean(dim=1), cat_x.std(dim=1)
        return x_mean.view(1, -1, 1).clone().detach(), x_std.view(1, -1, 1).clone().detach()

    @staticmethod
    def per_instance(x: tensor, seq_len: tensor) -> (tensor, tensor):
        x_mean = x.new_zeros((seq_len.shape[0], x.shape[1]))
        x_std = x.new_zeros((seq_len.shape[0], x.shape[1]))
        for i in range(x.shape[0]):
            x_mean[i, :] = x[i, :, :seq_len[i]].mean(dim=1)
            x_std[i, :] = x[i, :, :seq_len[i]].std(dim=1)
        return x_mean.unsqueeze(2), x_std.unsqueeze(2)


class JasperAugs(nn.Module):
    def __init__(self):
        super(JasperAugs, self).__init__()

        self.win_length = 320
        self.hop_length = 160
        self.n_fft = 512

        fb = torch.tensor(librosa.filters.mel(16000, self.n_fft, n_mels=64), dtype=torch.float).unsqueeze(0)
        self.register_buffer("fb", fb)
        self.register_buffer("window", torch.hann_window(self.win_length, periodic=False))
        self.normalize_batch = MyNormalize()

    def get_seq_len(self, seq_len):
        return torch.ceil(seq_len.to(dtype=torch.float) / self.hop_length).to(dtype=torch.int)

    def get_spectrogram(self, x):
        return torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                          window=self.window)

    def forward(self, x: torch.Tensor, seq_len: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        seq_len = self.get_seq_len(seq_len)
        x = x[:, 0]

        x = x + 1e-5 * torch.randn_like(x)
        x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - 0.97 * x[:, :-1]), dim=1)
        x = self.get_spectrogram(x)
        x = x.pow(2).sum(-1)
        x = torch.matmul(self.fb.to(x.dtype), x)
        x = torch.log(x + 1e-20)
        x = self.normalize_batch(x, seq_len)

        max_len = x.size(-1)
        mask = torch.arange(max_len, dtype=seq_len.dtype).to(x.device).expand(x.size(0), max_len) >= seq_len
        x = x.masked_fill(mask.unsqueeze(1), 0)
        x = nn.functional.pad(x, pad=[0, (-x.size(-1) % 16)])
        return x, seq_len.view(-1)


class MelJasperAugs(nn.Module):
    def __init__(self, aug: JasperAugs):
        super().__init__()
        self.aug = aug

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = torch.tensor([x.size(-1)], device=x.device).view(x.size(0), 1)
        return self.aug.forward(x, seq_len)[0]
