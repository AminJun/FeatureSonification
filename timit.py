import numpy as np
import torch
import torchaudio
from torch.utils.data.dataset import Dataset


class AbstractTimit(Dataset):
    def __init__(self, parent: list, time: int = 7 * 8 * 320 - 1, subset: str = 'SI', key: str = 'word'):
        # huggingface items key to word/phoneme dict per data.
        self._key = f'{key}_detail'
        """
          The length of output audio clip must follow the follwing formula, 
          Otherwise the Jasper's default augmentation will pad it with zeros and 
          hence there will be a huge bias towards zero, which eventually it will 
          degrade the quality of our sonification.  
        """
        assert (time + 1) % (320 * 8) == 0, f'time={time}+1 must be multiples of 320*8 bc of Jasper'
        self.data = []
        self.time = time
        self.subset = subset

        with torch.no_grad():
            for i, data in enumerate(parent):
                self.data += self.get_words(data, i)

    def complete(self, wav: torch.tensor) -> torch.tensor:
        if wav.size(-1) > self.time:
            raise Exception
        return wav.repeat(1, int(np.ceil(self.time / wav.size(-1))))[:, :self.time].clone().detach()

    def get_words(self, data: list, index: int) -> list:
        if data['sentence_type'] != self.subset:
            return []
        wav, sr = torchaudio.load(data['file'])
        wav = wav.view(-1)
        real_data = data[self._key]
        real_data = zip(real_data['start'], real_data['stop'], real_data['utterance'])
        return [(self.complete(wav[st:en]), w, index, st, en) for (st, en, w) in real_data if st != en]

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class TimitWord(AbstractTimit):
    def __init__(self, parent: Dataset, time: int = 7 * 8 * 320 - 1, subset: str = 'SI'):
        super().__init__(parent, time, subset, key='word')


class TimitPhoneme(AbstractTimit):
    def __init__(self, parent: Dataset, time: int = 7 * 8 * 320 - 1, subset: str = 'SI'):
        super().__init__(parent, time, subset, key='phonetic')
