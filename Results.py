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
