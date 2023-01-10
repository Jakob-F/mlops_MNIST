import torch
import numpy as np


content = [ ]
for i in range(5):
    content.append(np.load(f"data/raw/train_{i}.npz", allow_pickle=True))
train = torch.tensor(np.concatenate([c['images'] for c in content])).reshape(-1, 1, 28, 28)
train_targets = torch.tensor(np.concatenate([c['labels'] for c in content]))

torch.save(train, 'data/processed/train.npz')


content = np.load("data/raw/test.npz", allow_pickle=True)
test = torch.tensor(content['images']).reshape(-1, 1, 28, 28)
test_targets = torch.tensor(content['labels'])

torch.save(test, 'data/processed/test.npz')