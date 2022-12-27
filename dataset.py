from typing import List, Dict

import numpy as np
import torch.utils.data
from torch import Tensor


class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, text: List[np.ndarray], video: np.ndarray,
                 audio: np.ndarray, speaker: np.ndarray,
                 context: np.ndarray, label: np.ndarray) -> None:
        self.vision = video
        self.text = text
        self.audio = audio
        self.speaker = speaker
        self.context = context
        self.label = label

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return {
            'text': torch.Tensor(self.text[index]),
            'vision': torch.Tensor(self.vision[index]),
            'audio': torch.Tensor(self.audio[index]),
            'speaker': torch.Tensor(self.speaker[index]),
            'context': torch.Tensor(self.context[index]),
            'labels': torch.Tensor(self.label[index]).type(torch.LongTensor)
        }
