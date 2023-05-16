# CONFIG.py
import torch

class Config:
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
    VIDEO_CONTEXT_ID = 7

    TEXT_DIM = 1024
    VIDEO_DIM = 1000
    AUDIO_DIM = 1024

    TEXT_HIDDEN = 128
    VIDEO_HIDDEN = 128
    AUDIO_HIDDEN = 128
    SPEAKER_HIDDEN = 8

    VIDEO_DROPOUT = 0.2
    AUDIO_DROPOUT = 0.2
    TEXT_DROPOUT = 0.2
    POST_FUSION_DROPOUT = 0.4
    SPEAKER_DROPOUT = 0.2
    CONTEXT_DROPOUT = 0.2

    SHARED_EMBEDDING = 1024
    PROJECTION_EMBEDDING = 512

    POST_FUSION_DIM_2 = 256
    POST_FUSION_DIM_3 = 128

    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 0.001

    EPOCHS = 200
    EARLY_STOPPING = 20
    BATCH_SIZE = 32

    DATA_PATH = "data/extended_dataset.csv"
    BART_TARGET_EMBEDDINGS = "data/features/text/bart-embeddings.pt"
    BART_CONTEXT_EMBEDDINGS = "data/features/text/bart-context-embeddings.pt"
    AUDIO_EMBEDDINGS = "data/features/audio/audio-features.pt"
    AUDIO_CONTEXT_EMBEDDINGS = "data/features/audio/audio-context-features.pt"
    VISUAL_EMBEDDINGS = "data/features/visual/visual-features.pt"
    VISUAL_CONTEXT_EMBEDDINGS = "data/features/visual/visual-context-features.pt"
    MODEL_NAME = 'weighted_fusion'
    MODEL_PATH = "saved/" + MODEL_NAME + ".pth"
    RESULT_FILE = "output/{}.json"

class SingleModality(Config):
    POST_FUSION_DIM_1 = Config.SHARED_EMBEDDING + Config.SPEAKER_HIDDEN

class DoubleModality(Config):
    POST_FUSION_DIM_1 = Config.SHARED_EMBEDDING * 2 + Config.SPEAKER_HIDDEN

class VideoOnly(SingleModality):
    USE_VISUAL = True

class TextOnly(SingleModality):
    USE_TEXT = True

class AudioOnly(SingleModality):
    USE_AUDIO = True

class TextAndAudio(DoubleModality):
    USE_TEXT = True
    USE_AUDIO = True

class AudioAndVideo(DoubleModality):
    USE_AUDIO = True
    USE_VISUAL = True

class TextAndVideo(DoubleModality):
    USE_TEXT = True
    USE_VISUAL = True

class VideoAndAudioAndText(Config):
    POST_FUSION_DIM_1 = Config.SHARED_EMBEDDING * 3 + Config.SPEAKER_HIDDEN
    USE_AUDIO = True
    USE_VISUAL = True
    USE_TEXT = True

CONFIG = VideoAndAudioAndText()