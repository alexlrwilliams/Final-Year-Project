import jsonlines
import numpy as np

import config
from config import BERT_TARGET_EMBEDDINGS, BERT_CONTEXT_EMBEDDINGS


def read_embeddings(target_embedding_file):
    embeddings = []

    with jsonlines.open(target_embedding_file) as reader:
        for obj in reader:
            features = obj['features'][config.CLS_TOKEN_INDEX]
            bert_embedding_target = []
            for layer in [0, 1, 2, 3]:
                bert_embedding_target.append(np.array(features["layers"][layer]["values"]))

            bert_embedding_target = np.mean(bert_embedding_target, axis=0)
            embeddings.append(np.copy(bert_embedding_target))

    return embeddings


def get_bert_encoding():
    return read_embeddings(BERT_TARGET_EMBEDDINGS)


def get_context_bert_encoding(dataset):
    length = []
    for idx, ID in enumerate(dataset.keys()):
        length.append(len(dataset[ID]["context"]))

    context_utterance_embeddings = read_embeddings(BERT_CONTEXT_EMBEDDINGS)

    # Checking whether total context features == total context sentences
    assert (len(context_utterance_embeddings) == sum(length))

    # Rearrange context features for each target utterance
    cumulative_length = [length[0]]
    cumulative_value = length[0]
    for val in length[1:]:
        cumulative_value += val
        cumulative_length.append(cumulative_value)
    assert (len(length) == len(cumulative_length))
    end_index = cumulative_length
    start_index = [0] + cumulative_length[:-1]
    final_context_bert_features = []
    for start, end in zip(start_index, end_index):
        local_features = []
        for idx in range(start, end):
            local_features.append(context_utterance_embeddings[idx])
        final_context_bert_features.append(local_features)

    return final_context_bert_features