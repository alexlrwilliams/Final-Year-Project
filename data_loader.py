import os
import pickle

import h5py
import numpy as np
from sklearn.model_selection import StratifiedKFold

from utils import pickle_loader, toOneHot
import json
import config
from dataset import MultiModalDataset
from torch.utils.data import DataLoader
from Textual.TextEncoding import get_bert_encoding, get_context_bert_encoding

class MultiModalDataLoader:

    def __init__(self):
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

        self.parseData(text_bert_embeddings, video_features_file, audio_features, context_bert_embeddings)

        video_features_file.close()

        splits = 5
        skf = StratifiedKFold(n_splits=splits, shuffle=True)
        split_indices = [(train_index, test_index) for train_index, test_index in skf.split(self.data_input, self.data_output)]

        if not os.path.exists(config.INDICES_FILE):
            pickle.dump(split_indices, open(config.INDICES_FILE, 'wb'), protocol=2)

        self.split_indices = pickle_loader(config.INDICES_FILE)
        print('after pickle_loader: ')
        print(split_indices[0][0].shape, split_indices[0][1].shape)
        print(len(split_indices))

    def parseData(self, text, video, audio, context):
        for idx, ID in enumerate(self.dataset_json.keys()):
            # print(idx, 'processing ... ', ID) 0 proc`essing ...  1_60
            self.data_input.append(
                (text[idx],  # 0 TEXT_ID
                 video[ID][()],  # 1 VIDEO_ID
                 audio[ID],  # 2 AUDIO ID
                 self.dataset_json[ID]["show"],  # 3 SHOW_ID
                 self.dataset_json[ID]["speaker"],  # 4 SPEAKER ID
                 context[idx]  # 5 CONTEXT ID
                 ))
            self.data_output.append(int(self.dataset_json[ID]["sarcasm"]))

    def get_data_loader(self, train_ind_SI, author_ind):

        train_input = [self.data_input[ind] for ind in train_ind_SI]

        train_out = np.array([self.data_output[ind] for ind in train_ind_SI])
        train_out = np.expand_dims(train_out, axis=1)

        def getData(ID=None):
            return [instance[ID] for instance in train_input]

        # Text Feature
        train_text_feature = getData(config.TEXT_ID)

        # video Feature
        train_video_feature = getData(config.VIDEO_ID)
        train_video_feature_mean = np.array([np.mean(feature_vector, axis=0) for feature_vector in train_video_feature])

        # audio Feature
        audio = getData(config.AUDIO_ID)
        # (552, 283)

        train_audio_feature = np.array([np.mean(feature_vector, axis=1) for feature_vector in audio])

        # speaker feature
        speakers = getData(config.SPEAKER_ID)
        speakers = [author_ind.get(person.strip(), author_ind["PERSON"]) for person in speakers]
        speaker_feature = toOneHot(speakers, len(author_ind))

        # context feature
        context_utterances = getData(config.CONTEXT_ID)
        mean_features = []
        for utterance in context_utterances:
            mean_features.append(np.mean(utterance, axis=0))

        context_feature = np.array(mean_features)

        train_dataset = MultiModalDataset(train_text_feature, train_video_feature_mean,
                                          train_audio_feature, speaker_feature, context_feature, train_out)

        return DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
