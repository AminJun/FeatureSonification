import torch
import torch.nn as nn


class ChangeSampleRate(nn.Module):
    def __init__(self, input_rate: int = 0, output_rate: int = 0, output_size: int = 0):
        super().__init__()
        self.output_rate = output_rate
        self.input_rate = input_rate
        self.output_size = output_size

    def forward(self, wav: torch.tensor) -> torch.tensor:
        # Only accepts 1-channel waveform input
        wav = wav.view(wav.size(0), -1) if wav.dim() > 1 else wav.view(1, -1)
        new_length = wav.size(-1) * self.output_rate // self.input_rate
        new_length = new_length if self.output_size == 0 else self.output_size
        indices = (torch.arange(new_length).to(wav.device) * (wav.size(-1) / new_length))
        round_down = wav[:, indices.long()]
        round_up = wav[:, (indices.long() + 1).clamp(max=wav.size(-1) - 1)]
        output = round_down * (1. - indices.fmod(1.)).unsqueeze(0) + round_up * indices.fmod(1.).unsqueeze(0)
        return output
