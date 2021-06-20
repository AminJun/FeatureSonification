from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data.dataset import Dataset

import os


def load_timit(file: str):
    data_path = os.path.splitext(file)[0]
    with open(data_path + '.TXT', 'r') as txt_file:
        _, __, transcript = next(iter(txt_file)).strip().split(" ", 2)
    with open(data_path + '.WRD', 'r') as word_file:
        words = [l.strip().split(' ') for l in word_file]
        words = [(int(hd), int(tl), w) for (hd, tl, w) in words]
    with open(data_path + '.PHN', 'r') as phn_file:
        phonemes = [l.strip().split(' ') for l in phn_file]
        phonemes = [(int(hd), int(tl), w) for (hd, tl, w) in phonemes]
    wav, sr = torchaudio.load(data_path + '.WAV')
    return data_path, wav, transcript, words, phonemes


class Timit(Dataset):
    def __init__(self, root: str):
        self.root = root
        self.walker = list(str(p.stem) for p in Path(self.root).glob('*/*/*/*/*' + '.WAV'))

    def __getitem__(self, item) -> (str, torch.tensor, str, list, list):
        return load_timit(self.walker[item])

    def __len__(self) -> int:
        return len(self.walker)


class AbstractTimit(Dataset):
    def complete(self, wav: torch.tensor) -> torch.tensor:
        # if wav.size(-1) > self.time:
        #    raise Exception
        return wav.repeat(1, int(np.ceil(self.time / wav.size(-1))))[:, :self.time].clone().detach()

    def get_words(self, data: list, index: int) -> list:
        wav = data[1].cuda()
        return [(self.complete(wav[:, st:en]), w, index, st, en) for (st, en, w) in data[self._item_id] if
                st != en and self.subset in data[0]]

    def __init__(self, parent: Timit, time: int = 16000, subset: str = 'SI', debug: bool = False, item_id: int = 3):
        self._item_id = item_id
        assert (time + 1) % (320 * 8) == 0, f'time={time}+1 must be multiples of 320*8 bc of Jasper'
        self.data = []
        self.time = time
        self.subset = subset

        with torch.no_grad():
            for i, data in enumerate(parent):
                self.data += self.get_words(data, i)
                if debug and i > 3000:
                    break

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class TimitWord(AbstractTimit):
    def __init__(self, parent: Timit, time: int = 7*8*320 - 1, subset: str = 'SI', debug: bool = False):
        super().__init__(parent, time, subset, debug, item_id=3)


class TimitPhoneme(AbstractTimit):
    def __init__(self, parent: Timit, time: int = 7*8*320 - 1, subset: str = 'SI', debug: bool = False):
        super().__init__(parent, time, subset, debug, item_id=4)
