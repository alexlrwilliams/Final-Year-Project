import torch
import torch.nn as nn
from torch import Tensor

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