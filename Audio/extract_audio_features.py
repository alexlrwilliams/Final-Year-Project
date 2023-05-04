import numpy as np
from numpy import ndarray
from transformers import AutoProcessor, HubertModel
import librosa
import torch
import os
from config import CONFIG


def get_librosa_features(path: str) -> np.ndarray:
    y, sr = librosa.load(path)

    S_full, phase = librosa.magphase(librosa.stft(y, hop_length=512))
    S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, metric="cosine")
    S_filter = np.minimum(S_full, S_filter)

    margin_v = 4
    mask_v = librosa.util.softmask(S_full - S_filter,
                                   margin_v * S_filter,
                                   power=2)

    S_foreground = mask_v * S_full

    return librosa.istft(S_foreground * phase)

def load_librosa_embeddings(audio_files: list[str], path: str) -> list[ndarray]:
    if os.path.exists(path):
        return torch.load(path)
    else:
        processed_audio = [get_librosa_features(path) for path in audio_files]
        torch.save(processed_audio, path)
        print(f"Processed {len(processed_audio)} audio files in librosa.")
        return processed_audio


def get_filename(path: str) -> str:
    base_filename = os.path.basename(path)

    if '_u' in base_filename or '_c' in base_filename:
        filename, _ = base_filename.rsplit('_', 1)
    else:
        filename, _ = os.path.splitext(base_filename)
    return filename

def get_embeddings(batch_size: int, audio_dir: str, embedding_name: str):
    if not os.path.exists('../data/audio' + embedding_name + '-features.pt'):
        audio_files = [os.path.join(audio_dir, file_name) for file_name in os.listdir(audio_dir)]
        print(f"Found {len(audio_files)} audio files.")

        if not os.path.exists('../data/hubert' + embedding_name + '-embeddings.pt'):
            processed_audio = load_librosa_embeddings(audio_files, '../data/librosa' + embedding_name + '-embeddings.pt')

            num_batches = (len(processed_audio) + batch_size - 1) // batch_size
            batches = [processed_audio[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
            del processed_audio

            print(f"Processing {num_batches} batches of size {batch_size}.")

            embeddings = []
            for batch in batches:
                with torch.no_grad():
                    input_values = processor(batch, padding=True, return_tensors="pt").input_values.to(CONFIG.DEVICE)
                    hidden_states = model(input_values).last_hidden_state
                    mean_last_4_layers = torch.mean(hidden_states[:, -4:, :], dim=1)
                    del batch
                    del input_values
                    del hidden_states
                    embeddings.append(mean_last_4_layers)

            embeddings = torch.cat(embeddings, dim=0)
            print(f"Processed {len(embeddings)} embeddings.")
            torch.save(embeddings, '../data/hubert' + embedding_name + '-embeddings.pt')
        else:
            embeddings = torch.load('../data/hubert' + embedding_name + '-embeddings.pt')
            print(f"Processed {len(embeddings)} embeddings.")

        ids = [get_filename(path) for path in audio_files]
        output = {ids[idx]: embeddings[idx] for idx in range(len(embeddings))}
        print(f"Output {len(output)} audio features.")
        torch.save(output, '../data/audio' + embedding_name + '-features.pt')

def get_context_embeddings():
    get_embeddings(1, "../data/audios/context", '-context')

def get_utterance_embeddings():
    get_embeddings(4, "../data/audios/utterance", '')

if __name__ == '__main__':
    processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
    model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft").to(CONFIG.DEVICE)

    get_context_embeddings()
    get_utterance_embeddings()
