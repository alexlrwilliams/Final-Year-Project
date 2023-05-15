from typing import List, Dict

import numpy as np
import torch.utils.data
from torch import Tensor

from config import CONFIG


class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, text: List[np.ndarray], video: np.ndarray,
                 audio: np.ndarray, speaker: np.ndarray,
                 text_c: np.ndarray, audio_c: np.ndarray, video_c: np.ndarray, label: np.ndarray) -> None:
        self.vision = video if CONFIG.USE_VISUAL else None
        self.text = text if CONFIG.USE_TEXT else None
        self.audio = audio if CONFIG.USE_AUDIO else None
        self.speaker = speaker if CONFIG.USE_SPEAKER else None
        self.text_c = text_c if CONFIG.USE_CONTEXT and CONFIG.USE_TEXT else None
        self.audio_c = audio_c if CONFIG.USE_CONTEXT and CONFIG.USE_AUDIO else None
        self.video_c = video_c if CONFIG.USE_CONTEXT and CONFIG.USE_VISUAL else None
        self.label = label

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        data = {}
        if CONFIG.USE_TEXT:
            data = data | { 'text': torch.Tensor(self.text[index]) }
        if CONFIG.USE_VISUAL:
            data = data | { 'vision': torch.Tensor(self.vision[index]) }
        if CONFIG.USE_AUDIO:
            data = data | { 'audio': torch.Tensor(self.audio[index]) }
        if CONFIG.USE_SPEAKER:
            data = data | { 'speaker': torch.Tensor(self.speaker[index]) }
        if CONFIG.USE_CONTEXT:
            if CONFIG.USE_TEXT:
                data = data | { 'text_c': torch.Tensor(self.text_c[index]) }
            if CONFIG.USE_AUDIO:
                data = data | { 'audio_c': torch.Tensor(self.audio_c[index]) }
            if CONFIG.USE_VISUAL:
                data = data | {'video_c': torch.Tensor(self.video_c[index])}
        return data | { 'labels': torch.Tensor(self.label[index]).type(torch.LongTensor) }
