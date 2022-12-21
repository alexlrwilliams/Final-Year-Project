import torch.utils.data


class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, text, video, audio, speaker, context, label):
        self.vision = video
        self.text = text
        self.audio = audio
        self.speaker = speaker
        self.context = context
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return {
            'text': torch.Tensor(self.text[index]),
            'vision': torch.Tensor(self.vision[index]),
            'audio': torch.Tensor(self.audio[index]),
            'speaker': torch.Tensor(self.speaker[index]),
            'context': torch.Tensor(self.context[index]),
            'labels': torch.Tensor(self.label[index]).type(torch.LongTensor)
        }
