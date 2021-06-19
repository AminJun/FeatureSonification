import torch
import torch.nn as nn
import torchaudio

from datasets.audio.vae_speech import vae_speech


class VAELoss(nn.Module):
    def __init__(self, lam_kld=1.):
        super().__init__()
        self.mse = nn.MSELoss(reduction='mean')
        lam_kld = 1
        self.lam_kld = lam_kld
        self.aug2 = vae_speech.augmentations.cuda()
        self.aug = torchaudio.transforms.Spectrogram(n_fft=500, hop_length=250 + 1).cuda()

    def forward(self, output: torch.tensor, y: torch.tensor, ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        rec_x, z, mu, log_var = output
        # print(y.shape, rec_x.shape)
        rec_x = rec_x[:y.size(0), :y.size(1), :y.size(2)]
        y = y[:rec_x.size(0), :rec_x.size(1), :rec_x.size(2)]
        aug_x = self.aug(rec_x)
        aug_y = self.aug(y)

        mse = self.mse(aug_x, aug_y)  # + self.mse(rec_x, y)

        kld = (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1) * -0.5
        kld = kld.mean(dim=0)
        return self.lam_kld * kld + mse, mse, kld
