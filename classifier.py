from typing import Any, Tuple, Dict, Union, Optional

import torch
import torch.nn.functional as F
from torch import optim, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CONFIG
from utils import accuracy, SaveBestModel


def extract_data(data: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return \
        data['vision'].to(CONFIG.DEVICE) if CONFIG.USE_VISUAL else None, \
            data['audio'].to(CONFIG.DEVICE) if CONFIG.USE_AUDIO else None, \
            data['text'].to(CONFIG.DEVICE) if CONFIG.USE_TEXT else None, \
            data['text_c'].to(CONFIG.DEVICE) if CONFIG.USE_CONTEXT and CONFIG.USE_TEXT else None, \
            data['audio_c'].to(CONFIG.DEVICE) if CONFIG.USE_CONTEXT and CONFIG.USE_AUDIO else None, \
            data['speaker'].to(CONFIG.DEVICE) if CONFIG.USE_SPEAKER else None, \
            data['labels'].to(CONFIG.DEVICE)


def epoch_end(epoch: int, max_epochs: int, val_result: Dict[str, Any], result: Dict[str, Any]) -> None:
    print("Epoch [{}/{}], loss: {:.4f}, acc: {:.4f} val_loss: {:.4f}, val_acc: {:.4f}".format(
        epoch + 1, max_epochs, result['loss'], result['acc'], val_result['loss'], val_result['acc']
    ))


class Classifier(torch.nn.Module):
    """
        A classifier which the network inherits from, defines standard training and evaluating functions
    """

    def step(self, data: Dict[str, Tensor]) -> Tuple[Tensor, float, Tensor, Tensor]:
        """
            Pass input through model and calculate loss and accuracy

            :param data: input data from dataloader

            :return: loss, accuracy, outputs and labels
        """
        vision, audio, text, text_c, audio_c, speaker, labels = extract_data(data)

        output = self(text, vision, audio, speaker, text_c, audio_c)
        acc = accuracy(output, labels)
        loss = F.cross_entropy(output, labels.squeeze())
        return loss, acc, output, labels

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            optimizer: Optional[Optimizer] = None) -> Dict[str, Union[int, float]]:
        """
            Use gradient descent with the adam optimiser and backpropagation to train the model

            :param optimizer: optional optimiser, defaults to Adam
            :param val_loader: the validation set dataloader
            :param train_loader: the training set dataloader

            :return: best accuracy and its epoch
        """
        if optimizer is None:
            optimizer = optim.Adam(self.parameters(), lr=CONFIG.LEARNING_RATE, weight_decay=CONFIG.WEIGHT_DECAY)

        best_model = SaveBestModel()
        self.train()

        for epoch in range(CONFIG.EPOCHS):

            with tqdm(train_loader) as td:
                for batch_data in td:
                    optimizer.zero_grad()
                    loss, _, _, _ = self.step(batch_data)
                    loss.backward()
                    optimizer.step()

            # Validation phase
            val_result = self.evaluate(val_loader)
            best_model.save_if_best_model(val_result["acc"], epoch, self, CONFIG.MODEL_PATH)
 
            if (epoch + 1) % 10 == 0:
                train_result = self.evaluate(train_loader)
                epoch_end(epoch, CONFIG.EPOCHS, val_result, train_result)

            if (epoch - best_model.epoch) >= CONFIG.EARLY_STOPPING:
                break

        print(f'the best epochs:{best_model.epoch},the best acc:{best_model.acc}')
        return {
            'best_epoch': best_model.epoch,
            'best_acc': best_model.acc
        }

    def evaluate(self, data_loader: DataLoader) -> Dict[str, Union[float, Tensor]]:
        """
            Use gradient descent with the adam optimiser and backpropagation to train the model

            :param data_loader: the dataloader used to evaluate loss and accuracy

            :return: loss, accuracy, predictions, true values
        """
        self.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0
        eval_acc = 0.0
        with torch.no_grad():
            with tqdm(data_loader) as td:
                for batch_data in td:
                    val_loss, val_acc, outputs, labels = self.step(batch_data)

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
