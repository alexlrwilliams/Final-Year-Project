import numpy as np


class Result:
    def __init__(self, classification_report=None):
        self.best_acc = 0
        self.best_epoch = 0
        self.precision = 0
        self.recall = 0
        self.f1_score = 0
        if classification_report is not None:
            self.set_values_from_classification_report(classification_report)

    def set_values_from_classification_report(self, classification_report):
        self.f1_score = classification_report["weighted avg"]["f1-score"]
        self.precision = classification_report["weighted avg"]["precision"]
        self.recall = classification_report["weighted avg"]["recall"]
        self.best_acc = classification_report["best_result"]["best_acc"]
        self.best_epoch = classification_report["best_result"]["best_epoch"]


class IterationResults:
    def __init__(self):
        self.fold_results = []

    def add_result(self, result):
        self.fold_results.append(result)

    def print_result(self):
        for i, result in enumerate(self.fold_results):
            print("Fold ", i + 1)
            print("Weighted Precision: {}  Weighted Recall: {}  Weighted F score: {}".format(
                result.precision,
                result.recall,
                result.f1_score))
            print(f'best_epoch:{result.best_epoch}, best_acc:{result.best_acc}')

        avg_precision = np.mean([result.precision for result in self.fold_results])
        avg_recall = np.mean([result.recall for result in self.fold_results])
        avg_f1_score = np.mean([result.f1_score for result in self.fold_results])

        print("#" * 20)
        print("Avg :")
        print("Weighted Precision: {:.3f}  Weighted Recall: {:.3f}  Weighted F score: {:.3f}".format(
            avg_precision, avg_recall, avg_f1_score))

        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1_score
        }
