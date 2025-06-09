# %% [markdown]
# # Data Preprocessing

# %%
import scanpy as sc
import anndata as ad
import scipy
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import pairwise_distances, rbf_kernel
from scipy.spatial.distance import pdist,squareform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader, TensorDataset
from torch.distributions import LogNormal
from torch.optim.lr_scheduler import StepLR



import matplotlib.pyplot as plt
import umap.umap_ as umap
from matplotlib.colors import ListedColormap
from matplotlib import cm
import scipy.sparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

import warnings
warnings.filterwarnings('ignore')




# %%
adata = ad.read_h5ad('./data/tabula_muris.h5ad')
print(adata)
print(adata.X.min())
print(adata.X.max())
top_10_classes = adata.obs['cell_ontology_class'].value_counts().sort_values(ascending=False).head(10).index.tolist()
# subseting anndata base on cell_ontology_class
adata_sub = adata[adata.obs['cell_ontology_class'].isin(top_10_classes)].copy()

adata_sub.layers['raw_counts'] = adata_sub.X.copy()


# %% [markdown]
# # Distribution of gene expressions for each cell types

# %% [markdown]
# # Amount of samples for each classes

# %%
top_10_class_counts = adata_sub.obs['cell_ontology_class'].value_counts().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
top_10_class_counts.plot(kind='barh')
plt.xlabel('Number of Samples')
plt.ylabel('Cell Type')
plt.title('Number of Samples for 10 Cell Types')
plt.gca().invert_yaxis()
plt.show()

# %%
# run pca to select highly variable genes
sc.pp.pca(adata_sub, n_comps=50)
# plot pca
sc.pl.pca(adata_sub, color='cell_ontology_class', title='Tabular Muris',show=True)
# save pca layers
adata_sub.obsm['X_pca'] = adata_sub.obsm['X_pca'].copy()


# %%
# running umap
sc.pp.neighbors(adata_sub)
sc.tl.umap(adata_sub)
# plot umap
sc.pl.umap(adata_sub, color='cell_ontology_class',title='Tabular Muris', show=True)


# %% [markdown]
# Interpretation: The data seems cluster together, with B cell, T cell and mesenchymal are in separate cluster. Meanwhile basal and keratino are in the same cluster.

# %% [markdown]
# # Implementing Stable Diffusion model

# %%
##############################################
# 1. CONFIGURATION / PARAMETERS
##############################################

class Config:
    # Data params
    batch_size = 320
    num_timesteps = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model params
    input_dim = None  # to be set after loading data
    num_classes = None  # to be set after encoding labels
    hidden_dim = 512
    time_emb_dim = 128
    label_emb_dim = 128
    num_res_blocks = 8
    lr = 1e-3
    max_epochs = 1000

    # Early stopping
    patience = 20

    # Sampling params for log-normal prior
    log_prior_mu = 0
    log_prior_sigma = 0.25
    eps = 1e-6

    # Noise scheduler
    schedule_type = 'linear'

config = Config()


##############################################
# 2. UTILITIES
##############################################

def compute_pdist(X, Y=None, metric='euclidean'):
    if Y is None:
        Y = X
    return pairwise_distances(X, Y, metric=metric, n_jobs=-1)

def energy_distance(X, Y):
    XX = compute_pdist(X)
    YY = compute_pdist(Y)
    XY = compute_pdist(X, Y)
    return np.sqrt(2*np.mean(XY) - np.mean(XX) - np.mean(YY))

def classwise_energy_distance(real_data, generated_data, labels_encoded, generated_labels):
    energy_distances = []
    print("Energy Distance (per class):")
    for cls in np.unique(labels_encoded):
        real_cls_data = real_data[labels_encoded == cls]
        gen_cls_data = generated_data[generated_labels == cls]

        # Calculate energy distance for this class
        ed_cls = energy_distance(real_cls_data, gen_cls_data)
        energy_distances.append(ed_cls)

        # Print result for each class
        class_name = le.inverse_transform([cls])[0]
        print(f"Class {class_name}: {ed_cls:.4f}")

    # Compute and return average Energy Distance
    avg_ed = np.mean(energy_distances)
    return avg_ed

# need to develop further
def mmd_rbf(X, Y, gamma=None):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    XX = rbf_kernel(X, X, gamma)
    YY = rbf_kernel(Y, Y, gamma)
    XY = rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

def classwise_mmd(real_data, generated_data, labels_encoded, generated_labels):
    mmd_scores = []
    print("MMD (per class):")
    for cls in np.unique(labels_encoded):
        real_cls_data = real_data[labels_encoded == cls]
        gen_cls_data = generated_data[generated_labels == cls]

        # Calculate MMD for this class
        mmd_cls = mmd_rbf(real_cls_data, gen_cls_data)
        mmd_scores.append(mmd_cls)

        # Print result for each class
        class_name = le.inverse_transform([cls])[0]
        print(f"Class {class_name}: {mmd_cls:.4f}")

    # Compute and return average MMD
    avg_mmd = np.mean(mmd_scores)
    return avg_mmd

def classwise_spearman_correlation(real_data, generated_data, labels_encoded, generated_labels):
    spearman_scores = []
    print("Spearman Correlation (per class):")
    for cls in np.unique(labels_encoded):
        real_cls_data = real_data[labels_encoded == cls]
        gen_cls_data = generated_data[generated_labels == cls]

        # Ensure that the number of real and generated samples are equal
        if len(real_cls_data) != len(gen_cls_data):
            min_len = min(len(real_cls_data), len(gen_cls_data))
            real_cls_data = real_cls_data[:min_len]
            gen_cls_data = gen_cls_data[:min_len]

        # Calculate Spearman correlation for this class
        correlation, _ = spearmanr(real_cls_data, gen_cls_data, axis=0)  # axis=0 for column-wise correlation

        # If the result is a matrix, we need to average it
        if isinstance(correlation, np.ndarray):
            correlation = np.mean(correlation)

        spearman_scores.append(correlation)

        # Print result for each class
        class_name = le.inverse_transform([cls])[0]
        print(f"Class {class_name}: Spearman Correlation = {correlation:.4f}")

    # Compute and return average Spearman Correlation across all classes
    avg_spearman = np.mean(spearman_scores)
    return avg_spearman

def classwise_lisi(real_data, generated_data, labels_encoded, generated_labels, k=10):
    lisi_scores = []
    for cls in np.unique(labels_encoded):
        real_cls_data = real_data[labels_encoded == cls]
        gen_cls_data = generated_data[generated_labels == cls]

        # Ensure that the number of real and generated samples are equal
        if len(real_cls_data) < len(gen_cls_data):
            gen_cls_data = gen_cls_data[np.random.choice(len(gen_cls_data), len(real_cls_data), replace=False)]
        elif len(gen_cls_data) < len(real_cls_data):
            real_cls_data = real_cls_data[np.random.choice(len(real_cls_data), len(gen_cls_data), replace=False)]

        # Combine real and generated data for Nearest Neighbors computation
        combined_data = np.vstack([real_cls_data, gen_cls_data])

        # Fit the Nearest Neighbors model
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(combined_data)
        distances, indices = nn.kneighbors(combined_data)

        # Calculate LISI
        lisi_score = np.mean(np.linalg.norm(real_cls_data - gen_cls_data, axis=1))
        lisi_scores.append(lisi_score)

        # Print result for each class
        class_name = le.inverse_transform([cls])[0]
        print(f"Class {class_name}: LISI Score = {lisi_score:.4f}")

    # Compute and return average LISI
    avg_lisi = np.mean(lisi_scores)
    return avg_lisi


def classwise_rf_auc(real_data, generated_data, labels_encoded, generated_labels):
    auc_scores = []
    print("Random Forest AUC (per class):")
    for cls in np.unique(labels_encoded):
        real_cls_data = real_data[labels_encoded == cls]
        gen_cls_data = generated_data[generated_labels == cls]

        real_labels = np.ones(len(real_cls_data))
        gen_labels = np.zeros(len(gen_cls_data))

        combined_data = np.vstack([real_cls_data, gen_cls_data])
        combined_labels = np.concatenate([real_labels, gen_labels])

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(combined_data, combined_labels)

        prob_real = rf.predict_proba(combined_data)[:, 1]
        auc_score = roc_auc_score(combined_labels, prob_real)
        auc_scores.append(auc_score)

        # Print result for each class
        class_name = le.inverse_transform([cls])[0]
        print(f"Class {class_name}: RF AUC = {auc_score:.4f}")

    avg_rf_auc = np.mean(auc_scores)
    return avg_rf_auc

def classwise_spearman_and_pearson_correlation(real_data, generated_data, labels_encoded, generated_labels):
    for cls in np.unique(labels_encoded):
        real_cls_data = real_data[labels_encoded == cls]
        gen_cls_data = generated_data[generated_labels == cls]

        # Compute Spearman correlation
        spearman_corr, _ = stats.spearmanr(real_cls_data.mean(axis=0), gen_cls_data.mean(axis=0))

        # Compute Pearson correlation
        pearson_corr = np.corrcoef(real_cls_data.mean(axis=0), gen_cls_data.mean(axis=0))[0][1]

        class_name = le.inverse_transform([cls])[0]
        print(f"Class {class_name}: Spearman Correlation = {spearman_corr:.4f}, Pearson Correlation = {pearson_corr:.4f}")

def knn_classify(real_data, generated_data, labels_encoded, generated_labels):
    # Create labels: 1 for real data, 0 for generated data
    real_labels = np.ones(len(real_data))
    gen_labels = np.zeros(len(generated_data))

    # Combine the real and generated data
    combined_data = np.vstack([real_data, generated_data])
    combined_labels = np.concatenate([real_labels, gen_labels])

    # Train KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    X_train, X_val, y_train, y_val = train_test_split(combined_data, combined_labels, test_size=0.3, random_state=1)
    knn_classifier.fit(X_train, y_train)
    predicted_label = knn_classifier.predict(X_val)

    # Accuracy
    accuracy = accuracy_score(y_val, predicted_label)

    # AUC
    predicted_probabilities = knn_classifier.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, predicted_probabilities)

    return accuracy, auc

def evaluate_metrics(real_data, generated_data, labels_encoded, generated_labels, num_classes=config.num_classes, k=config.num_classes):
    avg_ed = classwise_energy_distance(real_data, generated_data, labels_encoded, generated_labels)
    avg_mmd = classwise_mmd(real_data, generated_data, labels_encoded, generated_labels)
    avg_scc = classwise_spearman_correlation(real_data, generated_data, labels_encoded, generated_labels)
    avg_lisi = classwise_lisi(real_data, generated_data, labels_encoded, generated_labels, k)
    avg_rf_auc = classwise_rf_auc(real_data, generated_data, labels_encoded, generated_labels)

    print("\nAverage Results across all classes:")
    print(f"Average Energy Distance: {avg_ed:.4f}")
    print(f"Average MMD: {avg_mmd:.4f}")
    print(f"Average Silhouette Score: {avg_scc:.4f}")
    print(f"Average LISI Score: {avg_lisi:.4f}")
    print(f"Average RF AUC Score: {avg_rf_auc:.4f}")

    avg_spearman = classwise_spearman_and_pearson_correlation(real_data, generated_data, labels_encoded, generated_labels)
    avg_lisi = classwise_lisi(real_data, generated_data, labels_encoded, generated_labels)
    accuracy, auc = knn_classify(real_data, generated_data, labels_encoded, generated_labels)

    print(f"Overall AUC: {auc:.4f}, Overall Accuracy: {accuracy:.4f}")
    print(f"Overall Spearman: {avg_spearman:.4f}, Overall LISI: {avg_lisi:.4f}")

    return avg_ed, avg_mmd, avg_scc, avg_lisi, avg_rf_auc




##############################################
# 3. MODEL IMPLEMENTATION
##############################################
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )
    def forward(self, x):
        return x + self.net(x)

class DenseBatchNorm(nn.Module):
    def __init__(self,in_units,out_units,activation=None):
        super().__init__()
        layers = [
            nn.Linear(in_units,out_units),
            nn.BatchNorm1d(out_units)
        ]
        if activation == 'relu':
            layers.append(nn.ReLU())
        self.dense_bn_act = nn.Sequential(*layers)

    def forward(self,x):
        return self.dense_bn_act(x)

class StableDiffusion(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, time_emb_dim=128, label_emb_dim=128, num_res_blocks=4):
        super().__init__()
        self.time_embedding = SinusoidalPosEmb(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.ReLU()
        )
        self.label_embedding = nn.Embedding(num_classes, label_emb_dim)
        self.init_proj = nn.Linear(input_dim + hidden_dim + label_emb_dim, hidden_dim)
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
    def __init__(self,input_dim,num_timesteps=1000,beta_start=1e-5,
                 beta_end=0.02, schedule_type=config.schedule_type):
        self.input_dim = input_dim
        self.num_timesteps = num_timesteps

        if schedule_type == 'linear':
            beta = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == 'cosine':
            beta = self._cosine_schedule(num_timesteps)
        else:
            raise ValueError("Unsupported schedule type. Use 'linear' or 'cosine'.")

        self.beta = beta
        self.alpha = 1 - beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def _cosine_schedule(self, num_timesteps, s=0.008):
        steps = torch.arange(num_timesteps + 1)/ num_timesteps
        alpha_bar = torch.cos((steps + s) / (1 + s) * torch.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return torch.clamp(beta, 1e-4, 0.9999)

    def add_noise(self, x, t):
        noise = torch.randn_like(x)
        alpha_bar_t = self.alpha_bar[t]
        noisy_x = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise
        return noisy_x, noise

class DiffusionProcessLogNormal:
    def __init__(self, input_dim, num_timesteps=1000, beta_start=1e-5, 
                 beta_end=0.02, schedule_type=config.schedule_type, mu=config.log_prior_mu, sigma=config.log_prior_sigma):
        self.input_dim = input_dim
        self.num_timesteps = num_timesteps

        if schedule_type == 'linear':
            beta = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == 'cosine':
            beta = self._cosine_schedule(num_timesteps)
        else:
            raise ValueError("Unsupported schedule type. Use 'linear' or 'cosine'.")

        self.beta = beta
        self.alpha = 1 - beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def _cosine_schedule(self, num_timesteps, s=0.008):
        steps = torch.arange(num_timesteps + 1)/ num_timesteps
        alpha_bar = torch.cos((steps + s) / (1 + s) * torch.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return torch.clamp(beta, 1e-4, 0.9999)

    def add_noise(self, x, t):
       # Log-normal noise sampling using PyTorch's LogNormal
        normal_dist = torch.distributions.LogNormal(loc=config.log_prior_mu, scale=config.log_prior_sigma)
        log_normal_noise = torch.exp(normal_dist.sample(x.size()))  # Log-normal noise

        alpha_bar_t = self.alpha_bar[t]
        noisy_x = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * log_normal_noise
        return noisy_x, log_normal_noise

class DiffusionDataset(Dataset):
    def __init__(self,X,labels,diffusion,num_timesteps):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.diffusion = diffusion
        self.num_timesteps = num_timesteps

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        t = torch.randint(0, self.diffusion.num_timesteps, ()).long()
        label = self.labels[idx]
        x_noisy, noise = self.diffusion.add_noise(x, t)
        return (x_noisy,t,label), noise

def prepare_dataset(X,labels,diffusion,batch_size,num_timesteps=1000):
    dataset = DiffusionDataset(X, labels,diffusion, num_timesteps)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True)
    return dataloader


class DiffusionDatasetLog(Dataset):
    def __init__(self, X, labels, diffusion, num_timesteps, eps=1e-6):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.diffusion = diffusion
        self.num_timesteps = num_timesteps

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_log = self.X[idx]
        t = torch.randint(0, self.diffusion.num_timesteps, ()).long()
        label = self.labels[idx]
        x_noisy, noise = self.diffusion.add_noise(x_log, t)
        return (x_noisy, t, label), noise

def prepare_dataset_log(X, labels, diffusion, batch_size, num_timesteps=1000):
    dataset = DiffusionDatasetLog(X, labels, diffusion, num_timesteps)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    return dataloader



##############################################
# 4. SAMPLING
##############################################

def sample(models,diffusion,labels,num_samples,device='cuda'):
    if not isinstance(models, list):
        models = [models]

    samples = torch.randn(num_samples, diffusion.input_dim).to(device)

    for t in reversed(range(diffusion.num_timesteps)):
        timestep = torch.full((num_samples,), t, dtype=torch.long).to(device)

        pred_noise = torch.zeros_like(samples)
        for model in models:
            model.eval()
            with torch.no_grad():
                pred = model(samples, timestep, labels)
            pred_noise += pred
        pred_noise /= len(models)

        alpha_bar_t = diffusion.alpha_bar[t].to(device)
        alpha_bar_prev = diffusion.alpha_bar[t-1].to(device) if t > 0 else torch.tensor(1.0).to(device)
        alpha_t = alpha_bar_t/alpha_bar_prev

        coef = diffusion.beta[t].to(device)/torch.sqrt(1 - alpha_bar_t)

        samples = (samples - coef * pred_noise) / torch.sqrt(alpha_t)
        if t > 0:
            noise = torch.randn_like(samples)
            samples += torch.sqrt(diffusion.beta[t]) * noise

    return samples.cpu().numpy()


def sample_log_normal(model, diffusion, labels, num_samples, device='cuda', mu=config.log_prior_mu, sigma=config.log_prior_sigma, eps=1e-6):
    # Use the LogNormal distribution from PyTorch for sampling
    log_normal_dist = torch.distributions.LogNormal(mu, sigma)
    
    # Sample from log-normal distribution
    log_normal_samples = log_normal_dist.sample((num_samples, diffusion.input_dim)).to(device)
    
    samples_log = log_normal_samples  

    for t in reversed(range(diffusion.num_timesteps)):
        timestep = torch.full((num_samples,), t, dtype=torch.long).to(device)
        pred_noise = torch.zeros_like(samples_log)

        # Use the single model (no need for a list of models here)
        model.eval()
        with torch.no_grad():
            pred = model(samples_log, timestep, labels)
        pred_noise += pred

        alpha_bar_t = diffusion.alpha_bar[t].to(device)
        alpha_bar_prev = diffusion.alpha_bar[t-1].to(device) if t > 0 else torch.tensor(1.0).to(device)
        alpha_t = alpha_bar_t / alpha_bar_prev

        coef = diffusion.beta[t].to(device) / torch.sqrt(1 - alpha_bar_t)

        samples_log = (samples_log - coef * pred_noise) / torch.sqrt(alpha_t)
        if t > 0:
            noise = torch.randn_like(samples_log)
            samples_log += torch.sqrt(diffusion.beta[t]) * noise

    # The model is predicting in log-space, so no need to apply exp() at the end
    samples_log = samples_log.cpu()  
    samples_np = samples_log.numpy()

    # Clean up values that are NaN, infinite, or beyond acceptable ranges
    samples_np = np.nan_to_num(samples_np, nan=eps, posinf=1e6, neginf=eps)
    samples_tensor = torch.tensor(samples_np, dtype=torch.float32, device=device)
    samples_tensor = torch.clamp(samples_tensor, min=1e-6, max=10)

    return samples_tensor



##############################################
# 5. GENERATE SYNTHETIC DATA
##############################################

def generate_synthetic_data(model, diffusion, labels_encoded, num_samples_per_class, device):
    model.eval()
    unique_labels = np.unique(labels_encoded)
    generated_data = []
    generated_labels = []

    for label in unique_labels:
        label_tensor = torch.full((num_samples_per_class,), label, dtype=torch.long).to(device)
        samples = sample(model, diffusion, label_tensor, num_samples_per_class, device=device)
        generated_data.append(samples)
        generated_labels.extend([label] * num_samples_per_class)

    generated_data = np.vstack(generated_data)
    generated_labels = np.array(generated_labels)
    return generated_data, generated_labels

def generate_synthetic_data_log(model, diffusion, labels_encoded, num_samples_per_class, device):
    model.eval()
    unique_labels = np.unique(labels_encoded)
    generated_data = []
    generated_labels = []

    for label in unique_labels:
        label_tensor = torch.full((num_samples_per_class,), label, dtype=torch.long).to(device)

        # Sample from the log-normal prior distribution
        samples = sample_log_normal(model, diffusion, label_tensor, num_samples_per_class, device=device)

        # Append the generated samples and corresponding labels
        generated_data.append(samples.cpu().numpy())
        generated_labels.extend([label] * num_samples_per_class)

    generated_data = np.vstack(generated_data)
    generated_labels = np.array(generated_labels)

    return generated_data, generated_labels


##############################################
# 6. TRAINING LOOP
##############################################

def train_loop(config, model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    best_val_loss = float('inf')
    trigger_times = 0

    # Lists to store loss values for plotting
    train_losses = []
    val_losses = []

    for epoch in range(config.max_epochs):
        model.train()
        train_loss_accum = 0.0
        num_batches = 0

        for (x_noisy, t, label_batch), noise in train_loader:
            x_noisy = x_noisy.to(config.device)
            t = t.to(config.device)
            label_batch = label_batch.to(config.device)
            noise = noise.to(config.device)

            pred_noise = model(x_noisy, t, label_batch)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()
            num_batches += 1

        avg_train_loss = train_loss_accum / num_batches
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (x_noisy, t, label_batch), noise in val_loader:
                x_noisy = x_noisy.to(config.device)
                t = t.to(config.device)
                label_batch = label_batch.to(config.device)
                noise = noise.to(config.device)

                pred_noise = model(x_noisy, t, label_batch)
                val_loss += F.mse_loss(pred_noise, noise).item()
        val_loss /= len(train_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{config.max_epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {val_loss:.6f}")

        # Early stopping logic (optional)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= config.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        scheduler.step()

def log_normal_loss(pred_noise, true_noise, eps=1e-6):
    # Ensure both pred_noise and noise are within a reasonable range
    pred_noise = torch.clamp(pred_noise, min=eps)
    true_noise = torch.clamp(true_noise, min=eps)

    # MSE in log space for the log-normal prior
    log_pred_noise = torch.log(pred_noise)
    log_true_noise = torch.log(true_noise)
    recon_loss = F.mse_loss(log_pred_noise, log_true_noise)
    return recon_loss


def train_loop_log(config, model, train_loader, val_loader):
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # AdamW optimizer for better performance
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    best_val_loss = float('inf')
    trigger_times = 0

    # Lists to store loss values for plotting
    train_losses = []
    val_losses = []

    for epoch in range(config.max_epochs):
        model.train()
        train_loss_accum = 0.0
        num_batches = 0

        for (x_noisy, t, label_batch), noise in train_loader:
            x_noisy = x_noisy.to(config.device)
            t = t.to(config.device)
            label_batch = label_batch.to(config.device)
            noise = noise.to(config.device)

            # Forward pass: Get the predicted noise (pred_noise), mean and logvar
            pred_noise = model(x_noisy, t, label_batch)

            # Compute log-normal loss
            # loss = F.mse_loss(pred_noise, noise)
            loss = log_normal_loss(pred_noise,noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            train_loss_accum += loss.item()
            num_batches += 1

        avg_train_loss = train_loss_accum / num_batches
        train_losses.append(avg_train_loss)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (x_noisy, t, label_batch), noise in val_loader:
                x_noisy = x_noisy.to(config.device)
                t = t.to(config.device)
                label_batch = label_batch.to(config.device)
                noise = noise.to(config.device)

                # Forward pass for validation
                pred_noise = model(x_noisy, t, label_batch)

                val_loss += log_normal_loss(pred_noise, noise).item()

        val_loss /= len(train_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{config.max_epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {val_loss:.6f}")

        # Early stopping logic (optional)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= config.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Step the learning rate scheduler
        scheduler.step()


# %%
# Convert sparse matrix to dense
X = adata_sub.X
if scipy.sparse.issparse(X):
    X = X.toarray()

labels = adata_sub.obs['cell_ontology_class'].values
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# parameter for config
config.input_dim = X.shape[1]
config.num_classes = len(np.unique(labels_encoded))

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, labels_encoded,
    test_size=0.2,
    random_state=42,
    stratify=labels_encoded
)

# %% [markdown]
# # Training for normal distribution prior

# %%
if __name__ == "__main__":
    # Initialize diffusion process (normal prior)

    diffusion = DiffusionProcess(config.input_dim,
                                 num_timesteps=config.num_timesteps,
                                 schedule_type='linear')
    train_loader = prepare_dataset(X_train, y_train, diffusion, batch_size=config.batch_size)
    val_loader   = prepare_dataset(X_val,   y_val,   diffusion, batch_size=config.batch_size)

    # Initialize model
    model = StableDiffusion(
        input_dim=config.input_dim,
        num_classes=config.num_classes,
        hidden_dim=config.hidden_dim,
        time_emb_dim=config.time_emb_dim,
        label_emb_dim=config.label_emb_dim,
        num_res_blocks=config.num_res_blocks
    ).to(config.device)

    # Train with normal prior data preparation
    train_loop(
    config=config,
    model=model,
    train_loader=train_loader,
    val_loader=val_loader)

    # Save the trained model weights
    torch.save(model.state_dict(), './weight/stable_diffusion_normal_prior.pth')
    print("Model trained and saved with normal prior.")

# %%
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize diffusion and model (make sure to match training config)
    diffusion = DiffusionProcess(X.shape[1], num_timesteps=1000)

    num_classes = len(np.unique(labels_encoded))
    model = StableDiffusion(
        input_dim=config.input_dim,
        num_classes=config.num_classes,
        hidden_dim=config.hidden_dim,
        time_emb_dim=config.time_emb_dim,
        label_emb_dim=config.label_emb_dim,
        num_res_blocks=config.num_res_blocks
    ).to(config.device)
    # Load your trained model weights (normal prior model)
    model.load_state_dict(torch.load('./weight/stable_diffusion_normal_prior.pth', map_location=device))
    model.eval()

    # Generate synthetic data per class
    num_samples_per_class = 5000
    generated_data_normal, generated_labels_normal = generate_synthetic_data(model, diffusion, labels_encoded, num_samples_per_class, device)



# %%
classwise_energy_distance(real_data=X, generated_data=generated_data_normal, labels_encoded=labels_encoded, generated_labels=generated_labels_normal)


# %%
classwise_mmd(real_data=X, generated_data=generated_data_normal, labels_encoded=labels_encoded, generated_labels=generated_labels_normal)

# %% [markdown]
# # Plotting the generated and real data

# %%
generated_labels_text = le.inverse_transform(generated_labels_normal)
adata_plot = np.concatenate((X,generated_data_normal),axis=0)
adata_plot = ad.AnnData(adata_plot, dtype=np.float32)
adata_plot.obs['celltype'] = np.concatenate((labels,generated_labels_text),axis=0)
adata_plot.obs['cell_name'] = [f'Real cell' for i in range(X.shape[0])] + [f'Generated cell' for i in range(generated_data_normal.shape[0])]


# %%
sc.pp.highly_variable_genes(adata_plot, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata_plot.raw = adata_plot
adata_plot = adata_plot[:, adata_plot.var.highly_variable]

sc.pp.scale(adata_plot)
sc.tl.pca(adata_plot, svd_solver='arpack')

sc.pp.neighbors(adata_plot, n_neighbors=10, n_pcs=20)
sc.tl.umap(adata_plot)
sc.pl.umap(adata=adata_plot,color="cell_name",size=6,title='Comparision between generated and real scRNA data')

# %% [markdown]
# Comparison for each cell type

# %%
celltypes = adata_plot.obs['celltype'].unique()
custom_palette = {
    'Real cell': '#e57c20',       
    'Generated cell': '#2778b1', 
    'Other cell': '#d8d8d8'     
}
for celltype in celltypes:
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Filter data for the current cell type (plot only real and generated of this type)
    subset = adata_plot[adata_plot.obs['celltype'] == celltype]
    
    # Create a mask for other cell types (all data except current cell type)
    other_cells = adata_plot[adata_plot.obs['celltype'] != celltype].copy()
    other_cells.obs['cell_name'] = 'Other cell'  # Label other cells as 'Other cell'
    
    # Combine the current subset (real + generated) and 'Other' cells
    combined_data = adata_plot[adata_plot.obs['celltype'] == celltype].concatenate(other_cells)
    
    # Plot UMAP for the current cell type, highlighting real and generated cells as well as other cells in black
    sc.pl.umap(combined_data, color='cell_name', size=6, title=f'Comparison for Cell Type: {celltype}', ax=ax,palette=custom_palette)
    
    # Show the plot
    plt.show()

# %% [markdown]
# # Training for log-normal distribution prior

# %%
if __name__ == "__main__":
  # Convert sparse matrix to dense
    X = adata_sub.X
    if scipy.sparse.issparse(X):
        X = X.toarray()

    labels = adata_sub.obs['cell_ontology_class'].values
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    # parameter for config
    config.input_dim = X.shape[1]
    config.num_classes = len(np.unique(labels_encoded))

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, labels_encoded,
        test_size=0.2,
        random_state=42,
        stratify=labels_encoded
    )
    # Assuming X, labels_encoded and le are already defined and prepared
    config.input_dim = X.shape[1]
    config.num_classes = len(np.unique(labels_encoded))

    # Initialize diffusion process (normal prior)
    diffusion = DiffusionProcessLogNormal(config.input_dim, 
                                            num_timesteps=1000,
                                            mu=config.log_prior_mu, 
                                            sigma=config.log_prior_sigma,
                                            schedule_type='cosine')

    # Initialize model
    model = StableDiffusion(
        input_dim=config.input_dim,
        num_classes=config.num_classes,
        hidden_dim=config.hidden_dim,
        time_emb_dim=config.time_emb_dim,
        label_emb_dim=config.label_emb_dim,
        num_res_blocks=config.num_res_blocks
    ).to(config.device)

    # Train with log prior data preparation
    train_loop_log(
    config=config,
    model=model,
    train_loader=train_loader,
    val_loader=val_loader)

    # Save the trained model weights
    torch.save(model.state_dict(), './weight/stable_diffusion_log_prior.pth')
    print("Model trained and saved with log prior.")

# %%
# Convert sparse matrix to dense
X = adata_sub.X
if scipy.sparse.issparse(X):
    X = X.toarray()

labels = adata_sub.obs['cell_ontology_class'].values
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
# Assuming X, labels_encoded and le are already defined and prepared
config.input_dim = X.shape[1]
config.num_classes = len(np.unique(labels_encoded))

# %%
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize diffusion and model (make sure to match training config)
    diffusion = DiffusionProcess(X.shape[1], num_timesteps=1000)

    num_classes = len(np.unique(labels_encoded))
    model = StableDiffusion(
        input_dim=config.input_dim,
        num_classes=config.num_classes,
        hidden_dim=config.hidden_dim,
        time_emb_dim=config.time_emb_dim,
        label_emb_dim=config.label_emb_dim,
        num_res_blocks=config.num_res_blocks
    ).to(config.device)
    # Load your trained model weights (normal prior model)
    model.load_state_dict(torch.load('./weight/stable_diffusion_log_prior.pth', map_location=device))
    model.eval()

    # Generate synthetic data per class
    num_samples_per_class = 5000
    generated_data_log, generated_labels_log = generate_synthetic_data_log(model, diffusion, labels_encoded, num_samples_per_class, device)




# %%
generated_data_log

# %%
classwise_energy_distance(real_data=X, generated_data=generated_data_log, labels_encoded=labels_encoded, generated_labels=generated_labels_log)

# %%
classwise_mmd(real_data=X, generated_data=generated_data_log, labels_encoded=labels_encoded, generated_labels=generated_labels_log)

# %%
generated_labels_text = le.inverse_transform(generated_labels_log)
adata_plot_log = np.concatenate((X,generated_data_log),axis=0)
adata_plot_log = ad.AnnData(adata_plot, dtype=np.float32)
adata_plot_log.obs['celltype'] = np.concatenate((labels,generated_labels_text),axis=0)
adata_plot_log.obs['cell_name'] = [f'Real cell' for i in range(X.shape[0])] + [f'Generated cell' for i in range(generated_data_log.shape[0])]


# %%
sc.pp.highly_variable_genes(adata_plot_log, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata_plot_log.raw = adata_plot_log
adata_plot_log = adata_plot_log[:, adata_plot_log.var.highly_variable]

sc.pp.scale(adata_plot_log)
sc.tl.pca(adata_plot_log, svd_solver='arpack')

sc.pp.neighbors(adata_plot_log, n_neighbors=10, n_pcs=20)
sc.tl.umap(adata_plot_log)
sc.pl.umap(adata=adata_plot_log,color="cell_name",size=6,title='Comparision between generated and real scRNA data')

# %%
celltypes = adata_plot_log.obs['celltype'].unique()
custom_palette = {
    'Real cell': '#e57c20',       
    'Generated cell': '#2778b1', 
    'Other cell': '#d8d8d8'     
}
for celltype in celltypes:
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Filter data for the current cell type (plot only real and generated of this type)
    subset = adata_plot_log[adata_plot_log.obs['celltype'] == celltype]
    
    # Create a mask for other cell types (all data except current cell type)
    other_cells = adata_plot_log[adata_plot_log.obs['celltype'] != celltype].copy()
    other_cells.obs['cell_name'] = 'Other cell'  # Label other cells as 'Other cell'
    
    # Combine the current subset (real + generated) and 'Other' cells
    combined_data = adata_plot_log[adata_plot_log.obs['celltype'] == celltype].concatenate(other_cells)
    
    # Plot UMAP for the current cell type, highlighting real and generated cells as well as other cells in black
    sc.pl.umap(combined_data, color='cell_name', size=6, title=f'Comparison for Cell Type: {celltype}', ax=ax,palette=custom_palette)
    
    # Show the plot
    plt.show()

# %% [markdown]
# # Plot the noise process

# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
from joypy import joyplot

# Select a sample from your original dataset for demonstration
sample_data = adata_sub.X[0, :]  # For example, the first cell (a 1D array of gene expressions)
sample_data = sample_data.toarray().flatten()  # Flatten if it's a sparse matrix

# Initialize the diffusion process
diffusion = DiffusionProcess(input_dim=sample_data.shape[0], num_timesteps=1000)

# Create a list to store noisy data at each timestep
noisy_data = [sample_data]

# Apply noise at different timesteps
for t in range(1, 1000,50):  # Add noise for the first 5 timesteps
    noisy_sample, _ = diffusion.add_noise(torch.tensor(sample_data), t)
    noisy_data.append(noisy_sample.numpy().flatten())  # Store the noisy data

# Prepare data for joypy
# joypy expects a list of lists/arrays, where each item is the data for a single distribution.
data_for_joypy = noisy_data

# Create a joyplot visualization
# plt.figure(figsize=(12, 8))
fig, axes = joyplot(
    data_for_joypy,
    labels=[f"Timestep {t}" for t in range(len(data_for_joypy))],  # Labels for each timestep
    grid=False,
    ylim='auto',
    alpha=0.6
)

# Customize the plot with title and labels
plt.title('Visualization of the Noising Process from Original Data to Noised Data')
plt.xlabel('Gene Expression Level')
plt.ylabel('Density')
plt.show()



# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
from joypy import joyplot

# Select a sample from your original dataset for demonstration
sample_data = adata_sub.X[0, :]  # For example, the first cell (a 1D array of gene expressions)
sample_data = sample_data.toarray().flatten()  # Flatten if it's a sparse matrix

# Initialize the diffusion process
diffusion_log = DiffusionProcessLogNormal(input_dim=sample_data.shape[0], num_timesteps=1000)

# Create a list to store noisy data at each timestep
noisy_data = [sample_data]

# Apply noise at different timesteps
for t in range(1, 1000,50):  # Add noise for the first 5 timesteps
    noisy_sample, _ = diffusion_log.add_noise(torch.tensor(sample_data), t)
    noisy_data.append(noisy_sample.numpy().flatten())  # Store the noisy data

# Prepare data for joypy
# joypy expects a list of lists/arrays, where each item is the data for a single distribution.
data_for_joypy = noisy_data

# Create a joyplot visualization
# plt.figure(figsize=(12, 8))
fig, axes = joyplot(
    data_for_joypy,
    labels=[f"Timestep {t}" for t in range(len(data_for_joypy))],  # Labels for each timestep
    grid=False,
    ylim='auto',
    alpha=0.6
)

# Customize the plot with title and labels
plt.title('Visualization of the Noising Process from Original Data to Noised Data')
plt.xlabel('Gene Expression Level')
plt.ylabel('Density')
plt.show()




