# CONFIG.py
import torch

class Config:
    runs = 1

    USE_CONTEXT = True
    USE_SPEAKER = True

    USE_TEXT = False
    USE_AUDIO = False
    USE_VISUAL = False

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SPLITS = 10

    TEXT_ID = 0
    VIDEO_ID = 1
    AUDIO_ID = 2
    SHOW_ID = 3
    SPEAKER_ID = 4
    CONTEXT_ID = 5
    AUDIO_CONTEXT_ID = 6

    TEXT_DIM = 1024
    VIDEO_DIM = 2048
    AUDIO_DIM = 1024

    CONTEXT_HIDDEN = 32
    TEXT_HIDDEN = 32
    VIDEO_HIDDEN = 128
    AUDIO_HIDDEN = 16
    SPEAKER_HIDDEN = 4

    VIDEO_DROPOUT = 0.2
    AUDIO_DROPOUT = 0.2
    TEXT_DROPOUT = 0.2
    POST_FUSION_DROPOUT = 0.4
    SPEAKER_DROPOUT = 0.2
    CONTEXT_DROPOUT = 0.2

    POST_FUSION_DIM_1 = 512
    POST_FUSION_DIM_2 = 256
    POST_FUSION_DIM_3 = 128

    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 0.001

    EPOCHS = 200
    EARLY_STOPPING = 50
    BATCH_SIZE = 32

    DATA_PATH = "data/extended_dataset.csv"
    BART_TARGET_EMBEDDINGS = "data/bart-embeddings.pt"
    BART_CONTEXT_EMBEDDINGS = "data/bart-context-embeddings.pt"
    AUDIO_EMBEDDINGS = "data/audio-features.pt"
    AUDIO_CONTEXT_EMBEDDINGS = "data/audio-context-features.pt"
    MODEL_NAME = 'weighted_fusion'
    MODEL_PATH = "saved/" + MODEL_NAME + ".pth"
    RESULT_FILE = "output/{}.json"

class TextOnly(Config):
    USE_TEXT = True

class AudioOnly(Config):
    USE_AUDIO = True

class TextAndAudio(Config):
    USE_TEXT = True
    USE_AUDIO = True

CONFIG = TextAndAudio()