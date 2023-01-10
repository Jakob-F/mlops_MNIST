import torch
import numpy as np
from torch.utils.data import Dataset
import wget
import os


class CorruptMnist(Dataset):
    def __init__(self, train):
        self.download_data(train)
        if train:
            content = [ ]
            for i in range(5):
                content.append(np.load(f"data/raw/train_{i}.npz", allow_pickle=True))
            data = torch.tensor(np.concatenate([c['images'] for c in content])).reshape(-1, 1, 28, 28)
            targets = torch.tensor(np.concatenate([c['labels'] for c in content]))
        else:
            content = np.load("data/raw/test.npz", allow_pickle=True)
            data = torch.tensor(content['images']).reshape(-1, 1, 28, 28)
            targets = torch.tensor(content['labels'])
            
        self.data = data
        self.targets = targets
    
    def download_data(self, train):
        files = os.listdir("data/raw")
        print(files)
        if train:
            for file_idx in range(5):
                if f'train_{file_idx}.npz' not in files:
                    wget.download(f"https://raw.githubusercontent.com/SkafteNicki/dtu_mlops/main/data/corruptmnist/train_{file_idx}.npz", "data/raw")
        else:
            if "test.npz" not in files:    
                wget.download("https://raw.githubusercontent.com/SkafteNicki/dtu_mlops/main/data/corruptmnist/test.npz", "data/raw")
    
    def __len__(self):
        return self.targets.numel()
    
    def __getitem__(self, idx):
        return self.data[idx].float(), self.targets[idx]


if __name__ == "__main__":
    dataset_train = CorruptMnist(train=True)
    dataset_test = CorruptMnist(train=False)
    print(dataset_train.data.shape)
    print(dataset_train.targets.shape)
    print(dataset_test.data.shape)
    print(dataset_test.targets.shape)
    torch.save(dataset_train, 'data/processed/train.npz')
    torch.save(dataset_test, 'data/processed/test.npz')