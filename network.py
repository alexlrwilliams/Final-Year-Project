import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import config
from classifier import Classifier


class SingleModalityFusion(nn.Module):
    """
        Produce a pytorch neural network module used for multi-modal fusion
        Attributes:
            n_speaker - Number of speakers
            input_embedding_a - size of imbedded feature vector
            output - size of output
    """

    def __init__(self, n_speaker: int, input_embedding_a: int, output: int) -> None:
        """
            Initializes internal ModalityFusion state
        """
        super(SingleModalityFusion, self).__init__()
        self.n_speaker = n_speaker
        self.input_embedding = input_embedding_a
        self.shared_embedding = 1024
        self.projection_embedding = 512
        self.dropout = 0.5

        self.context_share = nn.Linear(self.input_embedding, self.shared_embedding)
        self.utterance_share = nn.Linear(self.input_embedding, self.shared_embedding)

        self.norm_context = nn.BatchNorm1d(self.shared_embedding)
        self.norm_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.collaborative_gate_1 = nn.Linear(2*self.shared_embedding, self.projection_embedding)
        self.collaborative_gate_2 = nn.Linear(self.projection_embedding, self.shared_embedding)

        self.final_layer = nn.Sequential(
            nn.Linear(self.n_speaker+self.shared_embedding, output),
            nn.BatchNorm1d(output),
            nn.ReLU(),
            nn.Dropout(self.dropout))

    def attention(self, featureA, featureB):
        input = torch.cat((featureA, featureB), dim=1)
        return nn.functional.softmax(self.collaborative_gate_1(input), dim=1)

    def attention_aggregator(self, feA, feB):
        """ This method caluates the attention for feA with respect to others"""
        input = self.attention(feA, feB)
        # here we call for pairwise attention
        return nn.functional.softmax(self.collaborative_gate_2(input), dim=1)

    def forward(self, uA: Tensor, cA: Tensor, speaker_embedding: Tensor) -> Tensor:
        """
            Read the bert text embeddings

            :param uA: A tensor representing the utterance modality
            :param cA: A tensor representing the context modality
            :param speaker_embedding: A tensor representing the speaker embedding

            :return: A tensor containing the concatenated / fused output of the three input modalities
        """

        shared_context = self.norm_context(
            nn.functional.relu(self.context_share(cA)))

        shared_utterance = self.norm_utterance(
            nn.functional.relu(self.utterance_share(uA)))

        updated_shared = shared_utterance * self.attention_aggregator(
            shared_utterance, shared_context)

        return self.final_layer(torch.cat((updated_shared, speaker_embedding), dim=1))


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
            Read the bert text embeddings

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

        self.video_subnet = SubNet(config.VIDEO_DIM, config.VIDEO_HIDDEN, config.VIDEO_DROPOUT) if config.USE_VISUAL else None
        self.audio_subnet = SubNet(config.AUDIO_DIM, config.AUDIO_HIDDEN, config.AUDIO_DROPOUT) if config.USE_AUDIO else None
        self.text_subnet = SubNet(config.TEXT_DIM, config.TEXT_HIDDEN, config.TEXT_DROPOUT) if config.USE_TEXT else None
        self.context_subnet = SubNet(config.TEXT_DIM, config.CONTEXT_HIDDEN, config.CONTEXT_DROPOUT) if config.USE_CONTEXT else None
        self.speaker_subnet = SubNet(speaker_num, config.SPEAKER_HIDDEN, config.SPEAKER_DROPOUT) if config.USE_SPEAKER else None

        self.fusion = SingleModalityFusion(config.SPEAKER_HIDDEN, config.TEXT_HIDDEN, config.POST_FUSION_DIM_1)

        self.post_fusion_layer_dropout = nn.Dropout(config.POST_FUSION_DROPOUT)

        self.post_fusion_layer_1 = nn.Sequential(
            nn.Linear(config.POST_FUSION_DIM_1, config.POST_FUSION_DIM_2),
            nn.BatchNorm1d(config.POST_FUSION_DIM_2),
            nn.ReLU())

        self.post_fusion_layer_2 = nn.Sequential(
            nn.Linear(config.POST_FUSION_DIM_2, config.POST_FUSION_DIM_3),
            nn.BatchNorm1d(config.POST_FUSION_DIM_3),
            nn.ReLU())

        self.fc = nn.Linear(config.POST_FUSION_DIM_3, 2)

    def forward(self, text_x: Tensor, video_x: Tensor, audio_x: Tensor, speaker_x: Tensor, context_x: Tensor) -> Tensor:
        """
            Read the bert text embeddings

            :param context_x: A tensor representing the context feature
            :param speaker_x: A tensor representing the speaker feature
            :param audio_x: A tensor representing the audio modality
            :param video_x: A tensor representing the video modality
            :param text_x: A tensor representing the textual modality

            :return: A tensor containing the output of the model
        """
        video_h = self.video_subnet(video_x) if config.USE_VISUAL else None
        audio_h = self.audio_subnet(audio_x) if config.USE_AUDIO else None
        text_h = self.text_subnet(text_x) if config.USE_TEXT else None
        speaker_h = self.speaker_subnet(speaker_x) if config.USE_SPEAKER else None
        context_h = self.context_subnet(context_x) if config.USE_CONTEXT else None

        fusion_h = self.fusion(text_h, context_h, speaker_h)

        x = self.post_fusion_layer_1(fusion_h)
        x = self.post_fusion_layer_dropout(x)

        x = self.post_fusion_layer_2(x)
        x = self.post_fusion_layer_dropout(x)

        return self.fc(x)
