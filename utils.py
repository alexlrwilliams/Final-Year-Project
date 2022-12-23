import pickle
import sys

import numpy as np
import torch
import tqdm

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


def evaluate(model, data_loader):
    model.eval()
    y_pred, y_true = [], []
    eval_loss = 0.0
    eval_acc = 0.0
    with torch.no_grad():
        with tqdm.tqdm(data_loader) as td:
            for batch_data in td:
                val_loss, val_acc, outputs, labels = model.step(batch_data)

                eval_loss += val_loss.item()
                eval_acc += val_acc

                y_pred.append(outputs.argmax(1).cpu())
                y_true.append(labels.squeeze().long().cpu())

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    eval_loss = eval_loss / len(pred)
    eval_acc = eval_acc / len(pred)

    return {
        'loss': eval_loss,
        'acc': eval_acc,
        'pred': pred,
        'true': true
    }


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
