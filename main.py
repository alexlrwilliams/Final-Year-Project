import json
import os

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

import config
from data_loader import MultiModalDataLoader
from network import WeightedMultiModalFusionNetwork
from utils import get_author_ind, evaluate


def five_fold(cur_time):
    results = []
    dataloader_gen = MultiModalDataLoader()

    for fold, (train_index, test_index) in enumerate(dataloader_gen.split_indices):
        print('-' * 25, " Fold ", fold, '-' * 25)

        author_ind = get_author_ind(train_index, dataloader_gen.data_input)
        speakers_num = len(author_ind)

        train_loader = dataloader_gen.get_data_loader(train_index, author_ind)
        val_loader = dataloader_gen.get_data_loader(test_index, author_ind)
        test_loader = dataloader_gen.get_data_loader(test_index, author_ind)

        # Initialize the model and move to the device
        model = WeightedMultiModalFusionNetwork(speakers_num)
        model = model.to(config.DEVICE)

        best_result = model.fit(train_loader, val_loader)

        model.load_state_dict(torch.load(config.MODEL_PATH)['model_state_dict'])
        model.to(config.DEVICE)

        test_output = evaluate(model, test_loader)
        print('Test: ', test_output['acc'])

        result_string = classification_report(test_output['true'], test_output['pred'], digits=3)
        print('confusion_matrix(y_true, y_pred)')
        print(confusion_matrix(test_output['true'], test_output['pred']))
        print(result_string)

        result_dict = classification_report(test_output['true'], test_output['pred'], digits=3, output_dict=True)
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


five_results = []

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