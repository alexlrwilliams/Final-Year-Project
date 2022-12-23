import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

import config
from utils import accuracy, SaveBestModel, evaluate


def extract_data(data):
    return \
        data['vision'].to(config.DEVICE), \
            data['audio'].to(config.DEVICE), \
            data['text'].to(config.DEVICE), \
            data['context'].to(config.DEVICE), \
            data['speaker'].to(config.DEVICE), \
            data['labels'].to(config.DEVICE)


class Classifier(torch.nn.Module):
    def step(self, data):
        vision, audio, text, context, speaker, labels = extract_data(data)

        output = self(text, vision, audio, speaker, context)
        acc = accuracy(output, labels)
        loss = F.cross_entropy(output, labels.squeeze())
        return loss, acc, output, labels

    def train_epoch_end(self, outputs):
        for x in outputs:
            losses = [x['loss']]
            accs = [x['acc']]
        epoch_loss = torch.stack(losses).mean()
        epoch_acc = torch.stack(accs).mean()

        return {'loss': epoch_loss.item(), 'acc': epoch_acc.item()}

    def epoch_end(self, epoch, max_epochs, val_result, result):
        print("Epoch [{}/{}], loss: {:.4f}, acc: {:.4f} val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch + 1, max_epochs, result['loss'], result['acc'], val_result['loss'], val_result['acc']
        ))

    def fit(self, train_loader, val_loader):
        optimizer = optim.Adam(self.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        best_model = SaveBestModel()
        for epoch in range(config.EPOCHS):
            self.train()

            with tqdm(train_loader) as td:
                for batch_data in td:
                    optimizer.zero_grad()
                    loss, _, _, _ = self.step(batch_data)
                    loss.backward()
                    optimizer.step()

            # Validation phase
            val_result = evaluate(self, val_loader)
            best_model.save_if_best_model(val_result["acc"], epoch, self, config.MODEL_PATH)

            if (epoch + 1) % 10 == 0:
                train_result = evaluate(self, train_loader)
                self.epoch_end(epoch, config.EPOCHS, val_result, train_result)

            if (epoch - best_model.epoch) >= config.EARLY_STOPPING:
                break

        print(f'the best epochs:{best_model.epoch},the best acc:{best_model.acc}')
        return {
            'best_epoch': best_model.epoch,
            'best_acc': best_model.acc
        }
