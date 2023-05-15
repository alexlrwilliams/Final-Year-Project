import json
from typing import List, Dict, Any

import h5py
import numpy as np
import torch
from h5py import File
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from config import CONFIG
from Textual.read_bart_embeddings import get_context_bart_encoding, get_target_bart_embedding
from dataset import MultiModalDataset
from utils import to_one_hot, get_data
import pandas as pd


class MultiModalDataLoader:

    def __init__(self) -> None:
        self.dataset = pd.read_csv(CONFIG.DATA_PATH, encoding = "ISO-8859-1")

        self.utterances = self.dataset[self.dataset['Sarcasm'].notna()]

        self.data_input, self.data_output = [], []

        # text
        text_bart_embeddings = get_target_bart_embedding() if CONFIG.USE_TEXT else None
        context_bart_embeddings = get_context_bart_encoding(self.dataset) if CONFIG.USE_CONTEXT else None

        # video
        video_features = torch.load(CONFIG.VISUAL_EMBEDDINGS) if CONFIG.USE_VISUAL else None
        video_context_features = torch.load(CONFIG.VISUAL_CONTEXT_EMBEDDINGS) if CONFIG.USE_VISUAL else None

        # audio
        audio_features = torch.load(CONFIG.AUDIO_EMBEDDINGS) if CONFIG.USE_AUDIO else None
        audio_context_features = torch.load(CONFIG.AUDIO_CONTEXT_EMBEDDINGS) if CONFIG.USE_CONTEXT else None

        self.parse_data(text_bart_embeddings, video_features, audio_features, context_bart_embeddings, audio_context_features, video_context_features)

        skf = StratifiedKFold(n_splits=CONFIG.SPLITS, shuffle=True)
        self.split_indices = [(train_index, test_index) for train_index, test_index in
                              skf.split(self.data_input, self.data_output)]

    def parse_data(self, text: List[np.ndarray], video: list[np.ndarray], audio: list[np.ndarray], text_c: List[list], audio_c: list[np.ndarray], video_c: list[np.ndarray]) -> None:
        """
        parse pretrained data into array of modalities for input to data loader

        :param audio: audio features
        :param video: video features
        :param text: utterance bert embeddings
        :param text_c: context bert embeddings
        :param audio_c: context audio embeddings
        :param video_c: context video embeddings
        """
        for index, row in enumerate(self.utterances.iterrows()):
            self.data_input.append(
                (text[index] if CONFIG.USE_TEXT else None,  # 0 TEXT_ID
                 video[row[1]['SCENE']] if CONFIG.USE_VISUAL else None,  # 1 VIDEO_ID
                 audio[row[1]['SCENE']] if CONFIG.USE_AUDIO else None,  # 2 AUDIO ID
                 row[1]['SHOW'],  # 3 SHOW_ID
                 row[1]["SPEAKER"] if CONFIG.USE_SPEAKER else None,  # 4 SPEAKER ID
                 text_c[index] if CONFIG.USE_CONTEXT and CONFIG.USE_TEXT else None,  # 5 CONTEXT ID
                 audio_c[row[1]['SCENE']] if CONFIG.USE_CONTEXT and CONFIG.USE_AUDIO else None, # 6 AUDIO CONTEXT ID
                 video_c[row[1]['SCENE']] if CONFIG.USE_CONTEXT and CONFIG.USE_VISUAL else None # 7 VIDEO CONTEXT ID
                 ))
            self.data_output.append(int(row[1]["Sarcasm"]))

    def get_data_loader(self, train_ind_si: List[int], author_ind: Dict[str, int]) -> DataLoader:
        """
        Get Data Loader for a specific set of indices

        :param train_ind_si: Array of indices for train data
        :param author_ind: Dictionary of all speakers and their corresponding indices

        :return: Dataloader for specific set of indices in data input from dataset
        """
        train_input = [self.data_input[ind] for ind in train_ind_si]

        train_out = np.array([self.data_output[ind] for ind in train_ind_si])
        train_out = np.expand_dims(train_out, axis=1)

        # Text Feature
        train_text_feature = get_data(CONFIG.TEXT_ID, train_input) if CONFIG.USE_TEXT else None

        # video Feature
        train_video_feature = None
        if CONFIG.USE_VISUAL:
            train_video_feature = get_data(CONFIG.VIDEO_ID, train_input)

        # audio Feature
        train_audio_feature = None
        if CONFIG.USE_AUDIO:
            train_audio_feature = get_data(CONFIG.AUDIO_ID, train_input)

        # speaker feature
        speaker_feature = None
        if CONFIG.USE_SPEAKER:
            speakers = get_data(CONFIG.SPEAKER_ID, train_input)
            speakers = [author_ind.get(person.strip(), author_ind["PERSON"]) for person in speakers]
            speaker_feature = to_one_hot(speakers, len(author_ind))

        # context feature
        text_context_feature = None
        audio_context_feature = None
        video_context_feature = None
        if CONFIG.USE_CONTEXT:
            if CONFIG.USE_TEXT:
                text_context_feature = get_data(CONFIG.CONTEXT_ID, train_input)
            if CONFIG.USE_AUDIO:
                audio_context_feature = get_data(CONFIG.AUDIO_CONTEXT_ID, train_input)
            if CONFIG.USE_VISUAL:
                video_context_feature = get_data(CONFIG.VIDEO_CONTEXT_ID, train_input)
        train_dataset = MultiModalDataset(train_text_feature,
                                          train_video_feature,
                                          train_audio_feature,
                                          speaker_feature,
                                          text_context_feature,
                                          audio_context_feature,
                                          video_context_feature,
                                          train_out)

        return DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True)
