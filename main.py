import os
import json

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from torchsummary import summary
from torch import optim
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import MultiModalDataLoader
from network import WeightedMultiModalFusionNetwork
from utils import get_author_ind
import config


def five_fold(cur_time):
    results = []
    dataloader_gen = MultiModalDataLoader()

    for fold, (train_index, test_index) in enumerate(dataloader_gen.split_indices):
        print(fold, '-' * 50)
        print(fold, train_index.shape, test_index.shape)

        print()

        train_ind_SI = train_index
        val_ind_SI = test_index
        test_ind_SI = test_index

        print(train_ind_SI.shape, val_ind_SI.shape, test_ind_SI.shape)

        author_ind = get_author_ind(train_ind_SI, dataloader_gen.data_input)
        speakers_num = len(author_ind)

        train_loader = dataloader_gen.get_data_loader(train_ind_SI, author_ind)
        val_loader = dataloader_gen.get_data_loader(val_ind_SI, author_ind)
        test_loader = dataloader_gen.get_data_loader(test_ind_SI, author_ind)

        # Initialize the model and move to the device
        model = WeightedMultiModalFusionNetwork(speakers_num)
        model = model.to(device)

        summary(model, [(config.TEXT_DIM,), (config.VIDEO_DIM,), (config.AUDIO_DIM,), (speakers_num,), (config.TEXT_DIM,)])

        best_result = do_train(model, train_loader, val_loader)

        print()
        print(f'load:{config.MODEL_PATH}')

        model.load_state_dict(torch.load(config.MODEL_PATH))
        model.to(device)

        # do test
        val_acc, y_pred, y_true = do_test(model, test_loader, mode="TEST")
        print('Test: ', val_acc)

        result_string = classification_report(y_true, y_pred, digits=3)
        print('confusion_matrix(y_true, y_pred)')
        print(confusion_matrix(y_true, y_pred))
        print(result_string)

        result_dict = classification_report(y_true, y_pred, digits=3, output_dict=True)
        result_dict['best_result'] = best_result
        results.append(result_dict)

    model_name = config.MODEL_NAME + str(cur_time)
    if not os.path.exists(os.path.dirname(config.RESULT_FILE)):
        os.makedirs(os.path.dirname(config.RESULT_FILE))
    with open(config.RESULT_FILE.format(model_name), 'w') as file:
        json.dump(results, file)
    print('dump results  into ', config.RESULT_FILE.format(model_name))

def printResult(model_name, fold):
    results = json.load(open(config.RESULT_FILE.format(model_name+str(fold)), "rb"))
    weighted_precision, weighted_recall = [], []
    weighted_fscores = []
    print("#" * 20)
    for fold, result in enumerate(results):
        weighted_fscores.append(result["weighted avg"]["f1-score"])
        weighted_precision.append(result["weighted avg"]["precision"])
        weighted_recall.append(result["weighted avg"]["recall"])
        print("Fold {}:".format(fold + 1))
        print("Weighted Precision: {}  Weighted Recall: {}  Weighted F score: {}".format(
            result["weighted avg"]["precision"],
            result["weighted avg"]["recall"],
            result["weighted avg"]["f1-score"]))
        print(f'best_epoch:{result["best_result"]["best_epoch"]}   best_acc:{result["best_result"]["best_acc"]}')
    print("#" * 20)
    print("Avg :")
    print("Weighted Precision: {:.3f}  Weighted Recall: {:.3f}  Weighted F score: {:.3f}".format(
        np.mean(weighted_precision),
        np.mean(weighted_recall),
        np.mean(weighted_fscores)))

    return {
        'precision': np.mean(weighted_precision),
        'recall': np.mean(weighted_recall),
        'f1': np.mean(weighted_fscores)
    }


def do_test(model, data_loader, mode="VAL"):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    y_pred, y_true = [], []
    eval_loss = 0.0
    eval_acc = 0.0
    with torch.no_grad():
        with tqdm(data_loader) as td:
            for batch_data in td:
                val_loss, val_acc, outputs, labels = model.valStep(batch_data, device)

                eval_loss += val_loss.item()
                eval_acc += val_acc

                y_pred.append(outputs.argmax(1).cpu())
                y_true.append(labels.squeeze().long().cpu())

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    eval_loss = eval_loss / len(pred)
    eval_acc = eval_acc / len(pred)
    print("%s-(%s) >> loss: %.4f acc: %.4f" % (mode, 'lf_dnn', eval_loss, eval_acc))

    return eval_acc, pred, true


def do_train(model, train_loader, val_loader):
    best_acc = 0
    epochs, best_epoch = 0, 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    while True:
        epochs += 1
        y_pred, y_true = [], []
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        with tqdm(train_loader) as td:
            for batch_data in td:
                optimizer.zero_grad()
                loss, outputs, labels = model.trainStep(batch_data, device)

                # backward
                loss.backward()

                # update
                optimizer.step()

                train_loss += loss.item()

                train_acc += (outputs.argmax(1) == torch.squeeze(labels.long())).sum().item()

                y_pred.append(outputs.argmax(1).cpu())
                y_true.append(labels.squeeze().long().cpu())

        pred, true = torch.cat(y_pred), torch.cat(y_true)

        train_loss = train_loss / len(pred)

        train_acc = train_acc / len(pred)

        print("TRAIN-(%s) (%d/%d)>> loss: %.4f train_acc: %.4f" % (
            'lf_dnn', epochs - best_epoch, epochs, train_loss, train_acc))

        val_acc, _, _ = do_test(model, val_loader, mode="VAL")
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epochs
            print(config.MODEL_PATH)
            if os.path.exists(config.MODEL_PATH):
                os.remove(config.MODEL_PATH)
            torch.save(model.cpu().state_dict(), config.MODEL_PATH)
            model.to(device)

        # early stop
        if epochs - best_epoch >= config.EARLY_STOPPING:
            print(f'the best epochs:{best_epoch},the best acc:{best_acc}')
            tmp = {
                'best_epoch': best_epoch,
                'best_acc': best_acc
            }
            return tmp
            # break


five_results = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for i in range(5):
    five_fold(i)
    tmp_dict = printResult(model_name=config.MODEL_NAME, fold = i)
    five_results.append(tmp_dict)

file_name = 'five_results'
with open(config.RESULT_FILE.format(file_name), 'w') as file:
    json.dump(five_results, file)
print('dump results into ', config.RESULT_FILE.format(file_name))

results = json.load(open(config.RESULT_FILE.format(file_name), "rb"))
precisions, recalls, f1s = [], [], []
for fold, result in enumerate(results):
    tmp1 = result['precision']
    tmp2 = result['recall']
    tmp3 = result['f1']
    precisions.append(tmp1)
    recalls.append(tmp2)
    f1s.append(tmp3)

print('five average: precision recall f1')
print(round(np.mean(precisions) * 100, 1), round(np.mean(recalls) * 100, 1), round(np.mean(f1s) * 100, 1))

tmp = {
    'precision:': np.mean(precisions),
    'recall': np.mean(recalls),
    'f1': np.mean(f1s)
}

file_name = 'five_results_average'
with open(config.RESULT_FILE.format(file_name), 'w') as file:
    json.dump(tmp, file)