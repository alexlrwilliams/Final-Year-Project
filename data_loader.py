import json
from typing import List, Dict, Any

import h5py
import numpy as np
from h5py import File
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

import config
from Textual.TextEncoding import get_bert_encoding, get_context_bert_encoding
from dataset import MultiModalDataset
from utils import pickle_loader, to_one_hot, get_data


class MultiModalDataLoader:

    def __init__(self) -> None:
        self.dataset_json = json.load(open(config.DATA_PATH_JSON, encoding="utf-8"))

        self.data_input, self.data_output = [], []

        # text
        text_bert_embeddings = get_bert_encoding()

        # context bert embeddings
        context_bert_embeddings = get_context_bert_encoding(self.dataset_json)

        # video
        video_features_file = h5py.File('data/features/utterances_final/resnet_pool5.hdf5')

        # audio
        audio_features = pickle_loader(config.AUDIO_PICKLE)

        self.parse_data(text_bert_embeddings, video_features_file, audio_features, context_bert_embeddings)

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
        for idx, ID in enumerate(self.dataset_json.keys()):
            self.data_input.append(
                (text[idx],  # 0 TEXT_ID
                 video[ID][()],  # 1 VIDEO_ID
                 audio[ID],  # 2 AUDIO ID
                 self.dataset_json[ID]["show"],  # 3 SHOW_ID
                 self.dataset_json[ID]["speaker"],  # 4 SPEAKER ID
                 context[idx]  # 5 CONTEXT ID
                 ))
            self.data_output.append(int(self.dataset_json[ID]["sarcasm"]))

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
        train_text_feature = get_data(config.TEXT_ID, train_input)

        # video Feature
        train_video_feature = get_data(config.VIDEO_ID, train_input)
        train_video_feature_mean = np.array([np.mean(feature_vector, axis=0) for feature_vector in train_video_feature])

        # audio Feature
        audio = get_data(config.AUDIO_ID, train_input)
        # (552, 283)

        train_audio_feature = np.array([np.mean(feature_vector, axis=1) for feature_vector in audio])

        # speaker feature
        speakers = get_data(config.SPEAKER_ID, train_input)
        speakers = [author_ind.get(person.strip(), author_ind["PERSON"]) for person in speakers]
        speaker_feature = to_one_hot(speakers, len(author_ind))

        # context feature
        context_utterances = get_data(config.CONTEXT_ID, train_input)
        mean_features = []
        for utterance in context_utterances:
            mean_features.append(np.mean(utterance, axis=0))

        context_feature = np.array(mean_features)

        train_dataset = MultiModalDataset(train_text_feature, train_video_feature_mean,
                                          train_audio_feature, speaker_feature, context_feature, train_out)

        return DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
