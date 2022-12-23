import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from classifier import Classifier


class Fusion(nn.Module):
    def __init__(self, modal_1, modal_2, modal_3):
        super(Fusion, self).__init__()
        self.text_weight = nn.Linear(modal_1, 1)
        self.audio_weight = nn.Linear(modal_2, 1)
        self.visual_weight = nn.Linear(modal_3, 1)

    def forward(self, modal_1, modal_2, modal_3):
        dict = {0: modal_1, 1: modal_2, 2: modal_3}

        modal_1_w = self.text_weight(torch.tanh(modal_1))
        modal_2_w = self.audio_weight(torch.tanh(modal_2))
        modal_3_w = self.visual_weight(torch.tanh(modal_3))

        weights = torch.cat([modal_1_w, modal_2_w, modal_3_w], -1)
        normalised_weights = F.softmax(weights, -1)

        output = [
            np.multiply(
                dict[i].detach().cpu(),
                torch.index_select(normalised_weights, 1, torch.tensor([i])).detach().cpu()
            ) for i in range(3)
        ]
        return torch.cat(output, dim=-1)

class SubNet(nn.Module):
    def __init__(self, in_size, hidden_size, dropout):
        super(SubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))
        return y_3


# Define the multi-modal fusion network
class WeightedMultiModalFusionNetwork(Classifier):
    def __init__(self, speaker_num):
        super(WeightedMultiModalFusionNetwork, self).__init__()

        self.video_subnet = SubNet(config.VIDEO_DIM, config.VIDEO_HIDDEN, config.VIDEO_DROPOUT)
        self.audio_subnet = SubNet(config.AUDIO_DIM, config.AUDIO_HIDDEN, config.AUDIO_DROPOUT)
        self.text_subnet = SubNet(config.TEXT_DIM, config.TEXT_HIDDEN, config.TEXT_DROPOUT)
        self.context_subnet = SubNet(config.TEXT_DIM, config.CONTEXT_HIDDEN, config.CONTEXT_DROPOUT)
        self.speaker_subnet = SubNet(speaker_num, config.SPEAKER_HIDDEN, config.SPEAKER_DROPOUT)

        self.fusion_1 = Fusion(config.TEXT_HIDDEN, config.AUDIO_HIDDEN, config.VIDEO_HIDDEN)

        self.post_fusion_dropout = nn.Dropout(p=config.POST_FUSION_DROPOUT)

        self.post_fusion_layer_1 = nn.Linear(config.TEXT_HIDDEN + config.VIDEO_HIDDEN + config.AUDIO_HIDDEN,
                                             config.POST_FUSION_DIM)
        self.post_fusion_layer_2 = nn.Linear(config.POST_FUSION_DIM, config.POST_FUSION_DIM)

        self.post_fusion_layer_3 = nn.Linear(config.POST_FUSION_DIM + config.SPEAKER_HIDDEN + config.CONTEXT_HIDDEN,
                                             config.POST_FUSION_DIM)
        self.fc = nn.Linear(config.POST_FUSION_DIM, 2)

    def forward(self, text_x, video_x, audio_x, speaker_x, context_x):
        video_h = self.video_subnet(video_x)
        audio_h = self.audio_subnet(audio_x)
        text_h = self.text_subnet(text_x)
        speaker_h = self.speaker_subnet(speaker_x)
        context_h = self.context_subnet(context_x)

        fusion_h = self.fusion_1(text_h, audio_h, video_h)

        x = self.post_fusion_dropout(fusion_h)

        x = F.relu(self.post_fusion_layer_1(x), inplace=True)
        x = F.relu(self.post_fusion_layer_2(x), inplace=True)

        late_fusion = torch.cat([x, speaker_h, context_h], dim=-1)

        x = self.post_fusion_dropout(late_fusion)

        x = F.relu(self.post_fusion_layer_3(x), inplace=True)

        return self.fc(x)
