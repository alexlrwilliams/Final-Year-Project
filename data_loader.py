import json
from typing import List, Dict, Any

import h5py
import numpy as np
from h5py import File
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

import config
from Textual.read_bart_embeddings import get_context_bart_encoding, get_target_bart_embedding
from dataset import MultiModalDataset
from utils import pickle_loader, to_one_hot, get_data
import pandas as pd


class MultiModalDataLoader:

    def __init__(self) -> None:
        self.dataset = pd.read_csv(config.DATA_PATH, encoding = "ISO-8859-1")

        self.utterances = self.dataset[self.dataset['Sarcasm'].notna()]

        self.data_input, self.data_output = [], []

        # text
        text_bert_embeddings = get_target_bart_embedding() if config.USE_TEXT else None

        # context bert embeddings
        context_bert_embeddings = get_context_bart_encoding(self.dataset) if config.USE_CONTEXT else None

        # video
        video_features_file = h5py.File('data/features/utterances_final/resnet_pool5.hdf5') if config.USE_VISUAL else None

        # audio
        audio_features = pickle_loader(config.AUDIO_PICKLE) if config.USE_AUDIO else None

        self.parse_data(text_bert_embeddings, video_features_file, audio_features, context_bert_embeddings)

        if config.USE_VISUAL:
            video_features_file.close()

        skf = StratifiedKFold(n_splits=config.SPLITS, shuffle=True)
        self.split_indices = [(train_index, test_index) for train_index, test_index in
                              skf.split(self.data_input, self.data_output)]

    def parse_data(self, text: List[np.ndarray], video: File, audio: Any, context: List[list]) -> None:
        """
        parse pretrained data into array of modalities for input to data loader

        :param context: context bert embeddings
        :param audio: audio features
        :param video: video features
        :param text: utterance bert embeddings
        """
        for index, row in enumerate(self.utterances.iterrows()):
            self.data_input.append(
                (text[index] if config.USE_TEXT else None,  # 0 TEXT_ID
                 video[index] if config.USE_VISUAL else None,  # 1 VIDEO_ID
                 audio[index] if config.USE_AUDIO else None,  # 2 AUDIO ID
                 row[1]['SHOW'],  # 3 SHOW_ID
                 row[1]["SPEAKER"] if config.USE_SPEAKER else None,  # 4 SPEAKER ID
                 context[index] if config.USE_CONTEXT else None  # 5 CONTEXT ID
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
        train_text_feature = get_data(config.TEXT_ID, train_input) if config.USE_TEXT else None

        # video Feature
        if config.USE_VISUAL:
            train_video_feature = get_data(config.VIDEO_ID, train_input)
            train_video_feature_mean = np.array([np.mean(feature_vector, axis=0) for feature_vector in train_video_feature])
        else:
            train_video_feature_mean = None

        # audio Feature
        if config.USE_AUDIO:
            audio = get_data(config.AUDIO_ID, train_input)
            train_audio_feature = np.array([np.mean(feature_vector, axis=1) for feature_vector in audio])
        else:
            train_audio_feature = None

        # speaker feature
        if config.USE_SPEAKER:
            speakers = get_data(config.SPEAKER_ID, train_input)
            speakers = [author_ind.get(person.strip(), author_ind["PERSON"]) for person in speakers]
            speaker_feature = to_one_hot(speakers, len(author_ind))
        else:
            speaker_feature = None

        # context feature
        context_feature = get_data(config.CONTEXT_ID, train_input) if config.USE_CONTEXT else None

        train_dataset = MultiModalDataset(train_text_feature,
                                          train_video_feature_mean,
                                          train_audio_feature,
                                          speaker_feature,
                                          context_feature, train_out)

        return DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
