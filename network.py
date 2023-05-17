import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from classifier import Classifier
from config import CONFIG
from fusion import SingleModalityFusion, DoubleModalityFusion, TripleModalityFusion

class SubNet(nn.Module):
    """
        Produce a pytorch neural network module used as a subnetwork for each modality
    """

    def __init__(self, in_size: int, hidden_size: int, dropout: float) -> None:
        """
            Initializes internal SubNet state
        """
        super(SubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        """
            :param x: A tensor representing the input modality
            :return: A tensor containing the output of the three layers
        """
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))
        return y_3


class WeightedMultiModalFusionNetwork(Classifier):
    """
        Produce a pytorch neural network module - final top-level multi-modal fusion network
    """

    def __init__(self, speaker_num) -> None:
        """
            Initializes internal WeightedMultiModalFusionNetwork state
        """
        super(WeightedMultiModalFusionNetwork, self).__init__()

        self.video_subnet = SubNet(CONFIG.VIDEO_DIM, CONFIG.VIDEO_HIDDEN, CONFIG.VIDEO_DROPOUT) if CONFIG.USE_VISUAL else None
        self.audio_subnet = SubNet(CONFIG.AUDIO_DIM, CONFIG.AUDIO_HIDDEN, CONFIG.AUDIO_DROPOUT) if CONFIG.USE_AUDIO else None
        self.text_subnet = SubNet(CONFIG.TEXT_DIM, CONFIG.TEXT_HIDDEN, CONFIG.TEXT_DROPOUT) if CONFIG.USE_TEXT else None
        print(CONFIG.USE_SPEAKER)
        self.speaker_subnet = SubNet(speaker_num, CONFIG.SPEAKER_HIDDEN, CONFIG.SPEAKER_DROPOUT) if CONFIG.USE_SPEAKER else None

        self.t_fusion = SingleModalityFusion(CONFIG.TEXT_HIDDEN)
        self.a_fusion = SingleModalityFusion(CONFIG.AUDIO_HIDDEN)
        self.v_fusion = SingleModalityFusion(CONFIG.VIDEO_HIDDEN)
        self.ta_fusion = DoubleModalityFusion(CONFIG.TEXT_HIDDEN, CONFIG.AUDIO_HIDDEN)
        self.va_fusion = DoubleModalityFusion(CONFIG.VIDEO_HIDDEN, CONFIG.AUDIO_HIDDEN)
        self.tv_fusion = DoubleModalityFusion(CONFIG.TEXT_HIDDEN, CONFIG.VIDEO_HIDDEN)
        self.tva_fusion = TripleModalityFusion(CONFIG.TEXT_HIDDEN, CONFIG.VIDEO_HIDDEN, CONFIG.AUDIO_HIDDEN)

        self.post_fusion_layer_dropout = nn.Dropout(CONFIG.POST_FUSION_DROPOUT)

        self.post_fusion_layer_1 = nn.Sequential(
            nn.Linear(CONFIG.POST_FUSION_DIM_1, CONFIG.POST_FUSION_DIM_2),
            nn.BatchNorm1d(CONFIG.POST_FUSION_DIM_2),
            nn.ReLU(),
            nn.Dropout(CONFIG.POST_FUSION_DROPOUT))

        self.post_fusion_layer_2 = nn.Sequential(
            nn.Linear(CONFIG.POST_FUSION_DIM_2, CONFIG.POST_FUSION_DIM_3),
            nn.BatchNorm1d(CONFIG.POST_FUSION_DIM_3),
            nn.ReLU(),
            nn.Dropout(CONFIG.POST_FUSION_DROPOUT))

        self.fc = nn.Linear(CONFIG.POST_FUSION_DIM_3, 2)

    def forward(self, text_x: Tensor, video_x: Tensor, audio_x: Tensor, speaker_x: Tensor, text_c: Tensor, audio_c: Tensor, video_c: Tensor) -> Tensor:
        """
            Read the bert text embeddings

            :param context_x: A tensor representing the context feature
            :param speaker_x: A tensor representing the speaker feature
            :param audio_x: A tensor representing the audio modality
            :param video_x: A tensor representing the video modality
            :param text_x: A tensor representing the textual modality
            :param text_c: A tensor representing the textual context modality
            :param audio_c: A tensor representing the audio context modality
            :param video_c: A tensor representing the visual context modality

            :return: A tensor containing the output of the model
        """
        video_h = self.video_subnet(video_x) if CONFIG.USE_VISUAL else None
        audio_h = self.audio_subnet(audio_x) if CONFIG.USE_AUDIO else None
        text_h = self.text_subnet(text_x) if CONFIG.USE_TEXT else None
        speaker_h = self.speaker_subnet(speaker_x) if CONFIG.USE_SPEAKER else None
        text_c_h = self.text_subnet(text_c) if CONFIG.USE_CONTEXT and CONFIG.USE_TEXT else None
        audio_c_h = self.audio_subnet(audio_c) if CONFIG.USE_CONTEXT and CONFIG.USE_AUDIO else None
        video_c_h = self.video_subnet(video_c) if CONFIG.USE_CONTEXT and CONFIG.USE_VISUAL else None

        if CONFIG.USE_VISUAL and CONFIG.USE_AUDIO and CONFIG.USE_TEXT:
            fusion_h = self.tva_fusion(text_h, text_c_h, video_h, video_c_h, audio_h, audio_c_h, speaker_h)
        elif CONFIG.USE_VISUAL and CONFIG.USE_AUDIO:
            fusion_h = self.va_fusion(video_h, video_c_h, audio_h, audio_c_h, speaker_h)
        elif CONFIG.USE_TEXT and CONFIG.USE_AUDIO:
            fusion_h = self.ta_fusion(text_h, text_c_h, audio_h, audio_c_h, speaker_h)
        elif CONFIG.USE_TEXT and CONFIG.USE_VISUAL:
            fusion_h = self.tv_fusion(text_h, text_c_h, video_h, video_c_h, speaker_h)
        elif CONFIG.USE_TEXT:
            fusion_h = self.t_fusion(text_h, text_c_h, speaker_h)
        elif CONFIG.USE_AUDIO:
            fusion_h = self.a_fusion(audio_h, audio_c_h, speaker_h)
        elif CONFIG.USE_VISUAL:
            fusion_h = self.v_fusion(video_h, video_c_h, speaker_h)
        else:
            raise RuntimeError("No Modality found.")
        x = self.post_fusion_layer_1(fusion_h)
        x = self.post_fusion_layer_dropout(x)

        x = self.post_fusion_layer_2(x)
        x = self.post_fusion_layer_dropout(x)

        return self.fc(x)
