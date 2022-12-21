import torch
from utils import accuracy
import torch.nn.functional as F

class Classifier(torch.nn.Module):
    def trainStep(self, data, device):
        vision = data['vision'].to(device)
        audio = data['audio'].to(device)
        text = data['text'].to(device)
        context = data['context'].to(device)
        speaker = data['speaker'].to(device)
        labels = data['labels'].to(device)

        output = self(text, vision, audio, speaker, context)
        loss = F.cross_entropy(output, labels.squeeze())
        return loss, output, labels

    def valStep(self, data, device):
        vision = data['vision'].to(device)
        audio = data['audio'].to(device)
        text = data['text'].to(device)
        context = data['context'].to(device)
        speaker = data['speaker'].to(device)
        labels = data['labels'].to(device)

        output = self(text, vision, audio, speaker, context)
        acc = accuracy(output, labels)
        loss = F.cross_entropy(output, labels.squeeze())
        return loss, acc, output, labels

    def trainEvalStep(self, data):
        vision, text, audio, labels = data
        output = self(text, vision, audio)
        acc = accuracy(output, labels)
        loss = F.cross_entropy(output, labels)
        return {"loss": loss.detach(), "acc": acc}

    def validation_epoch_end(self, outputs):
        for x in outputs:
            data_losses = [x['val_loss']]
            data_accs = [x['val_acc']]

        epoch_loss = torch.stack(data_losses).mean()
        epoch_acc = torch.stack(data_accs).mean()

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def train_epoch_end(self, outputs):
        for x in outputs:
            losses = [x['loss']]
            accs = [x['acc']]
        epoch_loss = torch.stack(losses).mean()
        epoch_acc = torch.stack(accs).mean()

        return {'loss': epoch_loss.item(), 'acc': epoch_acc.item()}

    def epoch_end(self, epoch, max_epochs, val_result, result):
        print("Epoch [{}/{}], loss: {:.4f}, acc: {:.4f} val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch + 1, max_epochs, result['loss'], result['acc'], val_result['val_loss'], val_result['val_acc']
        )
        )