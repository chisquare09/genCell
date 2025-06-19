import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.distributions import Normal, LogNormal, StudentT
from gencell.networks.pos_embedding import SinusoidalPosEmb
from gencell.networks.blocks import DenseBatchNorm, ResidualBlock

class StableDiffusion(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim, time_emb_dim, label_emb_dim, num_res_blocks):
        super().__init__()
        self.time_embedding = SinusoidalPosEmb(time_emb_dim)
        self.time_mlp = DenseBatchNorm(time_emb_dim, hidden_dim, activation='relu')
        self.label_embedding = nn.Embedding(num_classes, label_emb_dim)
        self.init_proj = DenseBatchNorm(input_dim + hidden_dim + label_emb_dim,
                                             hidden_dim, activation='relu')
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_res_blocks)])
        self.final_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t, labels):
        t_emb = self.time_embedding(t)
        t_emb = self.time_mlp(t_emb)
        label_emb = self.label_embedding(labels)
        h = torch.cat([x, t_emb, label_emb], dim=-1)
        h = self.init_proj(h)
        for block in self.res_blocks:
            h = block(h)
        return self.final_layer(h)


class DiffusionProcess:
    def __init__(self,input_dim,num_timesteps,beta_start,
                 beta_end, schedule_type,prior_type='normal', prior_params=None):
        # Set noise schedule
        beta = torch.linspace(beta_start, beta_end, num_timesteps) \
               if schedule_type=='linear' else self._cosine_schedule(num_timesteps)
        self.beta = beta
        self.alpha = 1 - beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.num_timesteps = num_timesteps
        # Configure prior distribution
        self.prior_type = prior_type
        self.prior_params = prior_params or {}

        # Initialize distribution
        if prior_type == 'normal':
            self.dist = Normal(loc=0.0, scale=1.0)
        elif prior_type == 'lognormal':
            mu = self.prior_params.get('mu', 0.0)
            sigma = self.prior_params.get('sigma', 1.0)
            self.dist = LogNormal(loc=mu, scale=sigma)
        elif prior_type == 'student':
            df = self.prior_params.get('df', 3.0)
            self.dist = StudentT(df=df)
        else:
            raise ValueError(f"Unknown prior_type '{prior_type}'")


    def _cosine_schedule(self, num_timesteps, s=0.008):
        steps = torch.arange(num_timesteps + 1)/ num_timesteps
        alpha_bar = torch.cos((steps + s) / (1 + s) * torch.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return torch.clamp(beta, 1e-4, 0.9999)
    
    def _sample_noise(self, shape, device=None):
        device = device or self.alpha_bar.device
        sample = self.dist.sample(shape).to(device)
        # Optionally clip for lognormal
        if self.prior_type == 'lognormal':
            clip = self.prior_params.get('clip_std', None)
            if clip is not None:
                sample = torch.clamp(sample, -clip, clip)
        return sample

    def add_noise(self, x, t):
        # Sample noise from chosen prior
        noise = self._sample_noise(x.shape, device=x.device)
        a_bar = self.alpha_bar[t]
        noisy_x = torch.sqrt(a_bar) * x + torch.sqrt(1 - a_bar) * noise, noise
        return noisy_x,noise


class DiffusionDataset(Dataset):
    def __init__(self, X, labels, diffusion: DiffusionProcess):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.diffusion = diffusion

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        t = torch.randint(0, self.diffusion.num_timesteps, ()).long()
        y = self.labels[idx]
        x_noisy, noise = self.diffusion.add_noise(x, t)
        return (x_noisy, t, y), noise

def prepare_dataset(X,labels,diffusion: DiffusionProcess,batch_size,num_timesteps=1000):
    dataset = DiffusionDataset(X, labels,diffusion, num_timesteps)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True)
    return dataloader