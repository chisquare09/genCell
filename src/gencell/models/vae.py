import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class VAE(nn.Module):
    def __init__(self, input_dim, num_classes, latent_dim=128, hidden_dim=64):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, hidden_dim)
        # Encoder: input concatenated with label embedding
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder: latent + label embedding
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        y_emb = self.label_emb(y)
        h = torch.cat([x, y_emb], dim=1)
        h = self.encoder(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        zcat = torch.cat([z, y_emb], dim=1)
        x_recon = self.decoder(zcat)
        return x_recon, mu, logvar

class VaeDataset(Dataset):
    def __init__(self, X, labels):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.labels[idx]


def prepare_vae_dataset(X, labels, batch_size: int):
    ds = VaeDataset(X, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=True)