import torch

if __name__ == '__main__':
    test = torch.load("../data/features/visual/visual-context-features.pt")
    print(len(test))