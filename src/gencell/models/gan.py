import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class Generator(nn.Module):
    def __init__(self,input_dim: int, latent_dim: int, num_classes: int, label_emb_dim:int, hidden_dim: int):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, label_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + label_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            # no activation: raw log1p gene space
        )

    def forward(self, z, labels):
        y = self.label_emb(labels)
        x = torch.cat([z, y], dim=1)
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim: int, num_classes: int,label_emb_dim:int, hidden_dim: int):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.net = nn.Sequential(
            nn.Linear(input_dim + label_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, labels):
        y = self.label_emb(labels)
        inp = torch.cat([x, y], dim=1)
        return self.net(inp)

class GanDataset(Dataset):
    def __init__(self, X, labels):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.labels[idx]


def prepare_gan_dataset(X, labels, batch_size: int):
    ds = GanDataset(X, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=True)