import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from classifier import Classifier


class Fusion(nn.Module):
    def __init__(self, in_features, out_features):
        super(Fusion, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, input):
        return self.linear(input)


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

        self.text_weight = nn.Parameter(torch.tensor(config.TEXT_INIT_WEIGHT))
        self.audio_weight = nn.Parameter(torch.tensor(config.VIDEO_INIT_WEIGHT))
        self.visual_weight = nn.Parameter(torch.tensor(config.AUDIO_INIT_WEIGHT))
        self.speaker_weight = nn.Parameter(torch.tensor(config.SPEAKER_INIT_WEIGHT))
        self.context_weight = nn.Parameter(torch.tensor(config.CONTEXT_INIT_WEIGHT))

        self.post_fusion_dropout = nn.Dropout(p=config.POST_FUSION_DROPOUT)

        self.post_fusion_layer_1 = Fusion(config.TEXT_HIDDEN + config.VIDEO_HIDDEN + config.AUDIO_HIDDEN,
                                          config.POST_FUSION_DIM)
        self.post_fusion_layer_2 = Fusion(config.POST_FUSION_DIM, config.POST_FUSION_DIM)
        self.post_fusion_layer_3 = Fusion(config.POST_FUSION_DIM + config.SPEAKER_HIDDEN + config.CONTEXT_HIDDEN,
                                          config.POST_FUSION_DIM)
        self.fc = nn.Linear(config.POST_FUSION_DIM, 2)

    def forward(self, text_x, video_x, audio_x, speaker_x, context_x):
        video_h = self.video_subnet(video_x)
        audio_h = self.audio_subnet(audio_x)
        text_h = self.text_subnet(text_x)
        speaker_h = self.speaker_subnet(speaker_x)
        context_h = self.context_subnet(context_x)

        text_w = self.text_weight * text_h
        audio_w = self.audio_weight * audio_h
        visual_w = self.visual_weight * video_h
        speaker_w = self.speaker_weight * speaker_h
        context_w = self.context_weight * context_h

        fusion_h = torch.cat([visual_w, text_w, audio_w], dim=-1)

        x = self.post_fusion_dropout(fusion_h)

        x = F.relu(self.post_fusion_layer_1(x), inplace=True)
        x = F.relu(self.post_fusion_layer_2(x), inplace=True)

        late_fusion = torch.cat([x, speaker_w, context_w], dim=-1)

        x = self.post_fusion_dropout(late_fusion)

        x = F.relu(self.post_fusion_layer_3(x), inplace=True)

        return self.fc(x)
