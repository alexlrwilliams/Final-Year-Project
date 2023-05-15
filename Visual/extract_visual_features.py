import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from Visual.vision_dataset import VisionDataset
from config import CONFIG


def pretrained_efficientnet() -> torch.nn.Module:
    efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
    efficientnet.eval()
    for param in efficientnet.parameters():
        param.requires_grad = False
    return efficientnet

"""Implementation adapted from original MUStARD paper https://github.com/soujanyaporia/MUStARD/blob/master/visual/extract_features.py"""
def save_efficientnet_features() -> None:
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = VisionDataset(transform=transforms, is_utterance=False)

    efficientnet = pretrained_efficientnet().to(CONFIG.DEVICE)

    class Identity(torch.nn.Module):
        def forward(self, input_: torch.Tensor) -> torch.Tensor:
            return input_

    efficientnet.fc = Identity()

    total_frame_count = sum(dataset.frame_count_by_video_id[video_id] for video_id in dataset.video_ids)

    with tqdm(total=total_frame_count, desc="Extracting EfficientNet features") as progress_bar:
        instances = []
        for instance in DataLoader(dataset):
            video_id = instance["id"]
            frames = instance["frames"]
            embeddings = []

            batch_size = 32
            for start_index in range(0, len(frames), batch_size):
                end_index = min(start_index + batch_size, len(frames))
                frame_ids_range = range(start_index, end_index)
                frame_batch = torch.stack([frames[frame_id].squeeze() for frame_id in frame_ids_range]).to(CONFIG.DEVICE)

                classifier_output = efficientnet(frame_batch)
                embeddings[start_index:end_index] = classifier_output.cpu()

                progress_bar.update(len(frame_ids_range))
            instances.append({
                'id': video_id,
                'embeddings': torch.mean(torch.stack(embeddings), dim=0).cpu()
            })
    torch.save(instances, '../data/features/visual/visual_context_features.pt')


if __name__ == '__main__':
    save_efficientnet_features()