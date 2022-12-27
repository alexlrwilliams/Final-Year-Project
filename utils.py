import pickle
import sys
from typing import List, Optional, Dict, Tuple, Any

import numpy as np
import torch
from torch.nn import Module

import config


def pickle_loader(filename: str) -> Any:
    """
        :param filename: The ID of the data e.g. SPEAKER_ID representing the index of the data in the item

        :return: deserialised pickle data
    """
    if sys.version_info[0] < 3:
        return pickle.load(open(filename, 'rb'))
    else:
        return pickle.load(open(filename, 'rb'), encoding="latin1")


def get_data(type_id: int, train_input: List[Tuple[object]]) -> List[Any]:
    """
        :param type_id: The ID of the data e.g. SPEAKER_ID representing the index of the data in the item
        :param train_input: The train input data that contains text, video, audio, show, speaker, context

        :return: Array for specified data for ID for each instance in train_input
    """
    return [instance[type_id] for instance in train_input]


def get_author_ind(train_ind_si: List[int], data_input: List[Tuple[object]]) -> Dict[str, int]:
    """
    Generate one hot encoding of data. Using argmax on output will be equivalent to the input.

    :param train_ind_si: Array of indices for train data
    :param data_input: The train input data that contains text, video, audio, show, speaker, context

    :return: Dictionary of all speakers and their corresponding indices
    """

    train_input = [data_input[ind] for ind in train_ind_si]

    authors = get_data(config.SPEAKER_ID, train_input)
    author_list = set()

    # Anonymise all unnamed people to same author index
    author_list.add("PERSON")

    for author in authors:
        author = author.strip()
        # Anonymise all unnamed people to same author index
        if "PERSON" not in author:
            author_list.add(author)

    author_ind = {author: ind for ind, author in enumerate(author_list)}
    return author_ind


def to_one_hot(data: List[int], size: Optional[int] = None) -> np.ndarray:
    """
    Generate one hot encoding of data. Using argmax on output will be equivalent to the input.

    :param size: Number of classes to one hot encode
    :param data: An integer array whose elements are the speakers

    :return: Numpy array full of arrays of one hot encoded equivalents to input array.
    """
    targets = np.array([data]).reshape(-1)
    one_hot_data = np.eye(size)[targets]
    assert (np.array_equal(data, np.argmax(one_hot_data, axis=1)))
    return one_hot_data


def accuracy(output: np.numarray, labels: np.numarray) -> float:
    """
    Calculate the accuracy of the model

    :param output: An integer array whose elements are the predictions from the model
    :param labels: An integer array whose elements are the correct labels of the data

    :return: floating point number which represents the percentage accuracy
    """
    return (output.argmax(1) == torch.squeeze(labels.long())).sum().item()


class SaveBestModel:
    """
        Produce an object that stores the best epoch and accuracy for a certain model
    """

    def __init__(self):
        self.acc = float('-inf')
        self.epoch = 0
        self.model = None

    def save_if_best_model(self, current_acc: float, epoch: int, model: Module, path: str) -> None:
        """
            Save the model to a file if the accuracy is better than the best accuracy so far.

            :param current_acc: An integer of the accuracy for the current epoch
            :param epoch: An integer of the current epoch number
            :param model: The state of the model during current epoch
            :param path: An String representing the path of the file to save to
        """
        if current_acc > self.acc:
            self.acc, self.epoch, self.model = current_acc, epoch, model
            print(f"\nSaving best model for epoch: {epoch + 1}\n")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
            }, path)
