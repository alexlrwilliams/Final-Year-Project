import numpy as np
import torch
import pickle
import sys
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


def val_evaluate(model, val_loader):
    for data in val_loader:
        outputs = [model.valStep(data)]
        # print(outputs)
    return model.validation_epoch_end(outputs)


def trainEval(model, train_loader):
    for data in train_loader:
        outputs = [model.trainEvalStep(data)]
        # print(outputs)
    return model.train_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader=None, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    epoch_validation_losses = []
    epoch_validation_accuracies = []
    print("start training")
    for epoch in tqdm.trange(epochs):

        # Training Phase
        for data in train_loader:
            loss = model.trainStep(data)
            # print(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Validation phase
        val_result = val_evaluate(model, val_loader)
        train_result = trainEval(model, train_loader)
        model.epoch_end(epoch, epochs, val_result, train_result)

        epoch_validation_losses.append(val_result["val_loss"])
        epoch_validation_accuracies.append(val_result["val_acc"])

        if (epoch + 1) % 200 == 0:
            for g in optimizer.param_groups:
                lr *= 0.1
                g['lr'] = lr

    print("finished training")
    return epoch_validation_losses, epoch_validation_accuracies
