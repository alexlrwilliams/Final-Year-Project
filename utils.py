import pickle
import sys

import numpy as np
import torch

import config


def pickle_loader(filename):
    if sys.version_info[0] < 3:
        return pickle.load(open(filename, 'rb'))
    else:
        return pickle.load(open(filename, 'rb'), encoding="latin1")


def get_data(ID, train_input):
    return [instance[ID] for instance in train_input]


def get_author_ind(train_ind_SI, data_input):
    train_input = [data_input[ind] for ind in train_ind_SI]

    authors = get_data(config.SPEAKER_ID, train_input)
    author_list = set()
    author_list.add("PERSON")

    for author in authors:
        author = author.strip()
        if "PERSON" not in author:  # PERSON3 PERSON1 all --> PERSON haha
            author_list.add(author)

    author_ind = {author: ind for ind, author in enumerate(author_list)}
    return author_ind


def toOneHot(data, size=None):
    '''
    Returns one hot label version of data
    '''
    oneHotData = np.zeros((len(data), size))
    oneHotData[range(len(data)), data] = 1

    assert (np.array_equal(data, np.argmax(oneHotData, axis=1)))
    return oneHotData


def accuracy(output, labels):
    return (output.argmax(1) == torch.squeeze(labels.long())).sum().item()


class SaveBestModel:
    def __init__(self):
        self.acc = float('-inf')
        self.epoch = 0
        self.model = None

    def save_if_best_model(self, current_acc, epoch, model, path):
        if current_acc > self.acc:
            self.acc, self.epoch, self.model = current_acc, epoch, model
            print(f"\nSaving best model for epoch: {epoch + 1}\n")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
            }, path)
