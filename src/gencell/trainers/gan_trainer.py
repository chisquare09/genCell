import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from gencell.models.gan import Generator, Discriminator, prepare_gan_dataset


def train_gan(cfg, X, labels, checkpoint_path=None, logdir=None):
    loader = prepare_gan_dataset(X, labels, cfg.batch_size)
    device = cfg.device
    G = Generator(cfg.latent_dim, cfg.num_classes, cfg.hidden_dim, cfg.input_dim).to(device)
    D = Discriminator(cfg.input_dim, cfg.num_classes, cfg.hidden_dim).to(device)
    optG = optim.Adam(G.parameters(), lr=cfg.lr, betas=tuple(cfg.betas))
    optD = optim.Adam(D.parameters(), lr=cfg.lr, betas=tuple(cfg.betas))
    criterion = nn.BCELoss()
    writer = SummaryWriter(log_dir=logdir or 'logs/gan')
    start_epoch = 1
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device)
        G.load_state_dict(ckpt['G_state'])
        D.load_state_dict(ckpt['D_state'])
        optG.load_state_dict(ckpt['optG'])
        optD.load_state_dict(ckpt['optD'])
        start_epoch = ckpt['epoch'] + 1
    for epoch in range(start_epoch, cfg.epochs + 1):
        G.train(); D.train()
        lossG_total, lossD_total = 0, 0
        for x_real, y_real in loader:
            bs = x_real.size(0)
            x_real = x_real.to(device); y_real = y_real.to(device)
            valid = torch.ones(bs,1, device=device); fake = torch.zeros(bs,1, device=device)
            # Train D
            optD.zero_grad()
            z = torch.randn(bs, cfg.latent_dim, device=device)
            x_fake = G(z, y_real)
            lossD = criterion(D(x_real, y_real), valid) + criterion(D(x_fake.detach(), y_real), fake)
            lossD.backward(); optD.step()
            # Train G
            optG.zero_grad()
            lossG = criterion(D(x_fake, y_real), valid)
            lossG.backward(); optG.step()
            lossD_total += lossD.item() * bs
            lossG_total += lossG.item() * bs
        avgD = lossD_total/len(loader.dataset); avgG = lossG_total/len(loader.dataset)
        writer.add_scalar('train/D_loss', avgD, epoch)
        writer.add_scalar('train/G_loss', avgG, epoch)
        print(f'Epoch {epoch} GAN D_loss: {avgD:.4f}, G_loss: {avgG:.4f}')
        if logdir:
            path = f"{logdir}/gan_epoch{epoch}.pt"
            torch.save({'epoch': epoch,
                        'G_state': G.state_dict(),
                        'D_state': D.state_dict(),
                        'optG': optG.state_dict(),
                        'optD': optD.state_dict()}, path)
    writer.close()
    return G, D