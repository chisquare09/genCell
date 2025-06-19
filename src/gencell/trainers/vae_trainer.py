import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from gencell.models.vae import VAE, prepare_vae_dataset

# Loss for VAE
def vae_loss(x_recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='mean')
    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld

# Training loop for VAE
def train_vae(cfg, X, labels, checkpoint_path=None, logdir=None):
    # data
    loader = prepare_vae_dataset(X, labels, cfg.batch_size)
    # model
    model = VAE(cfg.input_dim, cfg.hidden_dim, cfg.latent_dim, cfg.num_classes).to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    # resume
    start_epoch = 1
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=cfg.device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['opt_state'])
        start_epoch = ckpt['epoch'] + 1
    # logging
    writer = SummaryWriter(log_dir=logdir or 'logs/vae')
    # loop
    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        total_loss = 0
        for x, y in loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            recon, mu, logvar = model(x, y)
            loss = vae_loss(recon, x, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(loader.dataset)
        writer.add_scalar('train/loss', avg_loss, epoch)
        print(f'Epoch {epoch}/{cfg.epochs} VAE Loss: {avg_loss:.4f}')
        # save
        if logdir:
            path = f"{logdir}/vae_epoch{epoch}.pt"
            torch.save({'epoch': epoch,
                        'model_state': model.state_dict(),
                        'opt_state': optimizer.state_dict()}, path)
    writer.close()
    return model

# Generate data from VAE
def generate_vae(model, num_samples,device='cuda'):
    model.eval()
    C = model.label_emb.num_embeddings
    D = model.decoder[-1].out_features  # final output dim

    all_samples = []
    all_labels  = []

    with torch.no_grad():
        for cls in range(C):
            # build a label‐tensor of size [num_per_class]
            y_cls = torch.full(
                (num_samples,), 
                cls, 
                dtype=torch.long, 
                device=device
            )
            # embed and sample
            y_emb = model.label_emb(y_cls)
            z     = torch.randn(num_samples, model.fc_mu.out_features, device=device)
            zin   = torch.cat([z, y_emb], dim=1)
            Xc    = model.decoder(zin)               # torch.Tensor [num_per_class, D]
            all_samples.append(Xc.cpu().numpy())
            all_labels.extend([cls] * num_samples)

    X_gen = np.vstack(all_samples)               # (C*num_per_class, D)
    y_gen = np.array(all_labels, dtype=int)      # (C*num_per_class,)
    return X_gen, y_gen