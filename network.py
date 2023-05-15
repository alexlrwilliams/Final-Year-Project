import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config import CONFIG
from classifier import Classifier

class ModalityFusion(nn.Module):
    def __init__(self, n_speaker: int, input_embedding_a: int, output: int) -> None:
        super(ModalityFusion, self).__init__()

        self.n_speaker = n_speaker
        self.input_embedding_A = input_embedding_a
        self.shared_embedding = 1024
        self.projection_embedding = 512
        self.dropout = 0.5

        self.A_context_share = nn.Linear(self.input_embedding_A, self.shared_embedding)
        self.A_utterance_share = nn.Linear(self.input_embedding_A, self.shared_embedding)

        self.norm_A_context = nn.BatchNorm1d(self.shared_embedding)
        self.norm_A_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.collaborative_gate_1 = nn.Linear(2 * self.shared_embedding, self.projection_embedding)
        self.collaborative_gate_2 = nn.Linear(self.projection_embedding, self.shared_embedding)

        self.final_layer = nn.Sequential(
            nn.Linear(self.n_speaker + self.shared_embedding, output),
            nn.BatchNorm1d(output),
            nn.ReLU(),
            nn.Dropout(self.dropout))

    def attention(self, featureA, featureB):
        """ This method takes two features and caluate the attention """
        input = torch.cat((featureA, featureB), dim=1)
        return nn.functional.softmax(self.collaborative_gate_1(input), dim=1)

class SingleModalityFusion(ModalityFusion):
    """
        Produce a pytorch neural network module used for multi-modal fusion
        Attributes:
            n_speaker - Number of speakers
            input_embedding_a - size of imbedded feature vector
            output - size of output
    """

    def __init__(self, n_speaker: int, input_embedding_A: int, output: int) -> None:
        """
            Initializes internal ModalityFusion state
        """
        super(SingleModalityFusion, self).__init__(n_speaker, input_embedding_A, output)

        self.context_share = nn.Linear(self.input_embedding_A, self.shared_embedding)
        self.input_share = nn.Linear(self.input_embedding_A, self.shared_embedding)

        self.norm_context = nn.BatchNorm1d(self.shared_embedding)
        self.norm_input = nn.BatchNorm1d(self.shared_embedding)

    def attention_aggregator(self, feA, feB):
        """ This method caluates the attention for feA with respect to others"""
        input = self.attention(feA, feB)
        # here we call for pairwise attention
        return nn.functional.softmax(self.collaborative_gate_2(input), dim=1)

    def forward(self, input_modality: Tensor, context: Tensor, speaker_embedding: Tensor) -> Tensor:
        """
            Read the bert text embeddings

            :param input_modality: A tensor representing the input modality
            :param context: A tensor representing the context modality
            :param speaker_embedding: A tensor representing the speaker embedding

            :return: A tensor containing the concatenated / fused output of the three input modalities
        """

        shared_context = self.norm_context(
            nn.functional.relu(self.context_share(context)))

        shared_input = self.norm_input(
            nn.functional.relu(self.input_share(input_modality)))

        updated_shared = shared_input * self.attention_aggregator(
            shared_input, shared_context)

        return self.final_layer(torch.cat((updated_shared, speaker_embedding), dim=1))

class DoubleModalityFusion(ModalityFusion):
    def __init__(self, n_speaker: int, input_embedding_a: int, input_embedding_b: int, output: int) -> None:
        super(DoubleModalityFusion, self).__init__(n_speaker, input_embedding_a, output)

        self.input_embedding_B = input_embedding_b

        self.B_context_share = nn.Linear(self.input_embedding_B, self.shared_embedding)
        self.B_utterance_share = nn.Linear(self.input_embedding_B, self.shared_embedding)

        self.norm_B_context = nn.BatchNorm1d(self.shared_embedding)
        self.norm_B_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.final_layer = nn.Sequential(
            nn.Linear(self.n_speaker + 2 * self.shared_embedding, output),
            nn.BatchNorm1d(output),
            nn.ReLU(),
            nn.Dropout(self.dropout))

    def attention_aggregator(self, feA, feB, feC, feD):
        """ This method caluates the attention for feA with respect to others"""
        input = self.attention(feA, feB) + self.attention(feA,
                                                          feC) + self.attention(feA, feD)
        # here we call for pairwise attention
        return nn.functional.softmax(self.collaborative_gate_2(input), dim=1)

    def forward(self, uA, cA, uB, cB, speaker_embedding):
        """making Feature Projection in order to make all feature of same dimension"""

        shared_A_context = self.norm_A_context(
            nn.functional.relu(self.A_context_share(cA)))
        shared_A_utterance = self.norm_A_utterance(
            nn.functional.relu(self.A_utterance_share(uA)))

        shared_B_context = self.norm_B_context(
            nn.functional.relu(self.B_context_share(cB)))
        shared_B_utterance = self.norm_B_utterance(
            nn.functional.relu(self.B_utterance_share(uB)))

        updated_shared_A = shared_A_utterance * self.attention_aggregator(
            shared_A_utterance, shared_A_context, shared_B_context, shared_B_utterance)

        updated_shared_B = shared_B_utterance * self.attention_aggregator(
            shared_B_utterance, shared_B_context, shared_A_context, shared_A_utterance)

        input = torch.cat((updated_shared_A, updated_shared_B), dim=1)

        input = torch.cat((input, speaker_embedding), dim=1)

        return self.final_layer(input)

class TripleModalityFusion(DoubleModalityFusion):
    def __init__(self, n_speaker: int, input_embedding_a: int, input_embedding_b: int, input_embedding_C: int, output: int):
        super(TripleModalityFusion, self).__init__(n_speaker, input_embedding_a, input_embedding_b, output)

        self.input_embedding_C = input_embedding_C

        self.C_context_share = nn.Linear(self.input_embedding_C, self.shared_embedding)
        self.C_utterance_share = nn.Linear(self.input_embedding_C, self.shared_embedding)

        self.norm_C_context = nn.BatchNorm1d(self.shared_embedding)
        self.norm_C_utterance = nn.BatchNorm1d(self.shared_embedding)

        self.final_layer = nn.Sequential(
            nn.Linear(self.n_speaker + 3 * self.shared_embedding, output),
            nn.BatchNorm1d(output),
            nn.ReLU(),
            nn.Dropout(self.dropout))

    def attention_aggregator(self, feA, feB, feC, feD, feE, feF):
        """ This method caluates the attention for feA with respect to others"""
        input = self.attention(feA, feB) + self.attention(feA, feC) + self.attention(
            feA, feD) + self.attention(feA, feE) + self.attention(feA, feF)
        # here we call for pairwise attention
        return nn.functional.softmax(self.collaborative_gate_2(input), dim=1)

    def forward(self, uA, cA, uB, cB, uC, cC, speaker_embedding):
        """making Feature Projection in order to make all feature of same dimension"""

        shared_A_context = self.norm_A_context(
            nn.functional.relu(self.A_context_share(cA)))
        shared_A_utterance = self.norm_A_utterance(
            nn.functional.relu(self.A_utterance_share(uA)))

        shared_B_context = self.norm_B_context(
            nn.functional.relu(self.B_context_share(cB)))
        shared_B_utterance = self.norm_B_utterance(
            nn.functional.relu(self.B_utterance_share(uB)))

        shared_C_context = self.norm_C_context(
            nn.functional.relu(self.C_context_share(cC)))
        shared_C_utterance = self.norm_C_utterance(
            nn.functional.relu(self.C_utterance_share(uC)))


        updated_shared_A = shared_A_utterance * self.attention_aggregator(
            shared_A_utterance, shared_A_context, shared_C_context, shared_C_utterance, shared_B_context, shared_B_utterance)
        updated_shared_C = shared_C_utterance * self.attention_aggregator(
            shared_C_utterance, shared_C_context, shared_A_context, shared_A_utterance, shared_B_context, shared_B_utterance)
        updated_shared_B = shared_B_utterance * self.attention_aggregator(
            shared_B_utterance, shared_B_context, shared_A_context, shared_A_utterance, shared_C_context, shared_C_utterance)

        input = torch.cat((updated_shared_A, updated_shared_C), dim=1)
        input = torch.cat((input, updated_shared_B), dim=1)
        input = torch.cat((input, speaker_embedding), dim=1)

        return self.final_layer(input)

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

        self.video_subnet = SubNet(CONFIG.VIDEO_DIM, CONFIG.VIDEO_HIDDEN, CONFIG.VIDEO_DROPOUT) if CONFIG.USE_VISUAL else None
        self.audio_subnet = SubNet(CONFIG.AUDIO_DIM, CONFIG.AUDIO_HIDDEN, CONFIG.AUDIO_DROPOUT) if CONFIG.USE_AUDIO else None
        self.text_subnet = SubNet(CONFIG.TEXT_DIM, CONFIG.TEXT_HIDDEN, CONFIG.TEXT_DROPOUT) if CONFIG.USE_TEXT else None
        self.speaker_subnet = SubNet(speaker_num, CONFIG.SPEAKER_HIDDEN, CONFIG.SPEAKER_DROPOUT) if CONFIG.USE_SPEAKER else None

        self.t_fusion = SingleModalityFusion(CONFIG.SPEAKER_HIDDEN, CONFIG.TEXT_HIDDEN, CONFIG.POST_FUSION_DIM_1)
        self.a_fusion = SingleModalityFusion(CONFIG.SPEAKER_HIDDEN, CONFIG.AUDIO_HIDDEN, CONFIG.POST_FUSION_DIM_1)
        self.v_fusion = SingleModalityFusion(CONFIG.SPEAKER_HIDDEN, CONFIG.VIDEO_HIDDEN, CONFIG.POST_FUSION_DIM_1)
        self.ta_fusion = DoubleModalityFusion(CONFIG.SPEAKER_HIDDEN, CONFIG.TEXT_HIDDEN, CONFIG.AUDIO_HIDDEN, CONFIG.POST_FUSION_DIM_1)
        self.va_fusion = DoubleModalityFusion(CONFIG.SPEAKER_HIDDEN, CONFIG.VIDEO_HIDDEN, CONFIG.AUDIO_HIDDEN, CONFIG.POST_FUSION_DIM_1)
        self.tv_fusion = DoubleModalityFusion(CONFIG.SPEAKER_HIDDEN, CONFIG.TEXT_HIDDEN, CONFIG.VIDEO_HIDDEN, CONFIG.POST_FUSION_DIM_1)
        self.tva_fusion = TripleModalityFusion(CONFIG.SPEAKER_HIDDEN, CONFIG.TEXT_HIDDEN, CONFIG.VIDEO_HIDDEN, CONFIG.AUDIO_HIDDEN, CONFIG.POST_FUSION_DIM_1)

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

        x = self.post_fusion_layer_1(fusion_h)
        x = self.post_fusion_layer_dropout(x)

        x = self.post_fusion_layer_2(x)
        x = self.post_fusion_layer_dropout(x)

        return self.fc(x)
