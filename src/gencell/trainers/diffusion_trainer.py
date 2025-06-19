import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from gencell.models.diffusion import DiffusionProcess, StableDiffusion, prepare_dataset


def sample(models, diffusion: DiffusionProcess, labels: torch.Tensor,
           num_samples: int, device: torch.device = 'cuda') -> np.ndarray:
    """
    Generic sampling from one or multiple models with Gaussian, lognormal, or Student-T priors.
    Returns numpy array of shape [num_samples, input_dim].
    """
    if not isinstance(models, list):
        models = [models]
    samples = diffusion._sample_noise((num_samples, diffusion.input_dim), device=device)
    for t in reversed(range(diffusion.num_timesteps)):
        timestep = torch.full((num_samples,), t, dtype=torch.long, device=device)
        pred_noise = torch.zeros_like(samples, device=device)
        for model in models:
            model.eval()
            with torch.no_grad():
                pred_noise += model(samples, timestep, labels)
        pred_noise /= len(models)
        alpha_bar_t = diffusion.alpha_bar[t].to(device)
        alpha_bar_prev = diffusion.alpha_bar[t-1].to(device) if t > 0 else torch.tensor(1.0, device=device)
        alpha_t = alpha_bar_t / alpha_bar_prev
        coef = diffusion.beta[t].to(device) / torch.sqrt(1 - alpha_bar_t)
        samples = (samples - coef * pred_noise) / torch.sqrt(alpha_t)
        if t > 0:
            noise = diffusion._sample_noise(samples.shape, device=device)
            samples += torch.sqrt(diffusion.beta[t]) * noise
    return samples.cpu().numpy()


def generate_synthetic_data(model: StableDiffusion,
                            diffusion: DiffusionProcess,
                            labels_encoded: np.ndarray,
                            num_per_class: int,
                            device: torch.device = 'cuda') -> tuple[np.ndarray, np.ndarray]:
    """
    For each unique label, generate num_per_class samples via reverse diffusion.
    Returns tuple (generated_data, generated_labels).
    """
    unique_labels = np.unique(labels_encoded)
    all_samples = []
    all_labels = []
    for cls in unique_labels:
        label_tensor = torch.full((num_per_class,), cls, dtype=torch.long, device=device)
        samples = sample(model, diffusion, label_tensor, num_per_class, device)
        all_samples.append(samples)
        all_labels.extend([cls] * num_per_class)
    generated_data = np.vstack(all_samples)
    generated_labels = np.array(all_labels, dtype=int)
    return generated_data, generated_labels


def train_diffusion(cfg, X, labels, checkpoint_path=None, logdir=None):
    """
    Train a diffusion model using MSE noise prediction.
    """
    # 1) Setup diffusion
    diffusion = DiffusionProcess(
        input_dim=cfg.input_dim,
        num_timesteps=cfg.num_timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        schedule_type=cfg.schedule_type,
        prior_type=cfg.prior_type,
        prior_params={
            "mu": cfg.log_prior_mu,
            "sigma": cfg.log_prior_sigma,
            "clip_std": cfg.clip_std
        }
    )

    # 2) Prepare data loader
    loader = prepare_dataset(X, labels, diffusion, cfg.batch_size)
    device = cfg.device

    # 3) Model, optimizer, loss
    model = StableDiffusion(
        input_dim=cfg.input_dim,
        num_classes=cfg.num_classes,
        hidden_dim=cfg.hidden_dim,
        time_emb_dim=cfg.time_emb_dim,
        label_emb_dim=cfg.label_emb_dim,
        num_res_blocks=cfg.num_res_blocks
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = torch.nn.MSELoss()

    # 4) Resume from checkpoint
    start_epoch = 1
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["opt_state"])
        start_epoch = ckpt["epoch"] + 1

    # 5) Logging
    writer = SummaryWriter(log_dir=logdir or "logs/diffusion")

    # 6) Training loop
    for epoch in range(start_epoch, cfg.max_epochs + 1):
        model.train()
        total_loss = 0.0
        for (x_noisy, t, y), noise in loader:
            x_noisy, t, y, noise = [v.to(device) for v in (x_noisy, t, y, noise)]
            optimizer.zero_grad()
            pred_noise = model(x_noisy, t, y)
            loss = criterion(pred_noise, noise)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_noisy.size(0)

        avg_loss = total_loss / len(loader.dataset)
        writer.add_scalar("train/loss", avg_loss, epoch)
        print(f"Epoch {epoch}/{cfg.max_epochs} - Loss: {avg_loss:.4f}")

        if logdir:
            path = f"{logdir}/diffusion_epoch{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "opt_state": optimizer.state_dict()
            }, path)

    writer.close()
    return model