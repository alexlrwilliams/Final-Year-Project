import json

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

import config
from data_loader import MultiModalDataLoader
from network import WeightedMultiModalFusionNetwork
from results import Result, IterationResults
from utils import get_author_ind, evaluate


def k_fold_cross_validation():
    dataloader_gen = MultiModalDataLoader()
    iteration_results = IterationResults()

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

        result = Result(result_dict)
        iteration_results.add_result(result)
    return iteration_results


def print_result(results):
    for i, result in enumerate(results):
        print("Fold ", i + 1)
        print("Weighted Precision: {}  Weighted Recall: {}  Weighted F score: {}".format(
            result.precision,
            result.recall,
            result.f1_score))
        print(f'best_epoch:{result.best_epoch}, best_acc:{result.best_acc}')

    avg_precision = np.mean([result.precision for result in results])
    avg_recall = np.mean([result.recall for result in results])
    avg_f1_score = np.mean([result.f1_score for result in results])

    print("#" * 20)
    print("Avg :")
    print("Weighted Precision: {:.3f}  Weighted Recall: {:.3f}  Weighted F score: {:.3f}".format(
        avg_precision, avg_recall, avg_f1_score))

    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1_score
    }


five_results = []

for i in range(5):
    iteration_results = k_fold_cross_validation()
    tmp_dict = print_result(iteration_results.fold_results)
    five_results.append(tmp_dict)

avg_precision = np.mean([result['precision'] for result in five_results])
avg_recall = np.mean([result['recall'] for result in five_results])
avg_f1_score = np.mean([result['f1_score'] for result in five_results])

print('five average: precision recall f1')
print(round(avg_precision * 100, 1), round(avg_recall * 100, 1), round(avg_f1_score * 100, 1))

file_name = 'five_results_average'
with open(config.RESULT_FILE.format(file_name), 'w') as file:
    json.dump({
        'precision:': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1_score
    }, file)
