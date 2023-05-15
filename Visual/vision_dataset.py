import os
from typing import Callable, Dict

import PIL.Image
import pandas as pd
from torch.utils.data import Dataset


class VisionDataset(Dataset):
    """Implementation adapted from original MUStARD paper https://github.com/soujanyaporia/MUStARD/blob/master/visual/dataset.py"""

    def __init__(self, is_utterance: bool, transform: Callable = None, videos_data_path: str = "../data/extended_dataset.csv") -> None:

        self.FRAMES_DIR_PATH = "../data/frames/"
        if is_utterance:
            self.FRAMES_DIR_PATH += 'utterance'
        else:
            self.FRAMES_DIR_PATH += 'context'
        self.transform = transform

        df = pd.read_csv(videos_data_path, encoding="ISO-8859-1")
        self.video_ids = list(df['SCENE'].unique())

        for video_id in self.video_ids:
            video_folder_path = self._video_folder_path(video_id)
            if not os.path.exists(video_folder_path):
                raise FileNotFoundError(f"Directory {video_folder_path} not found, which was referenced in"
                                        f" {videos_data_path}")

        self.frame_count_by_video_id = {video_id: len(os.listdir(self._video_folder_path(video_id)))
                                        for video_id in self.video_ids}

    def _video_folder_path(self, video_id: str) -> str:
        path = os.path.join(self.FRAMES_DIR_PATH, video_id)
        if 'utterance' in self.FRAMES_DIR_PATH:
            path += '_u'
        else:
            path += '_c'
        return path

    def __getitem__(self, index) -> Dict[str, object]:
        video_id = self.video_ids[index]

        frames = []

        video_folder_path = self._video_folder_path(video_id)
        for i, frame_file_name in enumerate(os.listdir(video_folder_path)):
            frame = PIL.Image.open(os.path.join(video_folder_path, frame_file_name))
            if self.transform:
                frame = self.transform(frame)

            frames.append(frame)

        return {"id": video_id, "frames": frames}

    def __len__(self) -> int:
        return len(self.video_ids)