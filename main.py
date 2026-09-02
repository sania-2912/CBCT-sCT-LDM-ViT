import random
import numpy as np
import torch

import config
from dataset import get_dataloaders
from models.autoencoder import Autoencoder, autoencoder_loss
from models.unet import ConditionalDiffusionModel
from models.diffusion import GaussianDiffusion

def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_autoencoder(autoencoder, train_loader, device=config.DEVICE):
    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=config.AE_LR)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(config.AE_EPOCHS):
        autoencoder.train()
        total_loss = 0.0

        for _, ct in train_loader:
            ct = ct.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                reconstruction, _ = autoencoder(ct)
                loss = autoencoder_loss(reconstruction, ct)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f"[Autoencoder] Epoch {epoch + 1}/{config.AE_EPOCHS} | Loss: {total_loss / len(train_loader):.4f}")

    torch.save(autoencoder.state_dict(), config.AE_CHECKPOINT)
    print(f"Saved autoencoder checkpoint to {config.AE_CHECKPOINT}")


def train_diffusion(diffusion_model, autoencoder, diffusion, train_loader, device=config.DEVICE):
    for param in autoencoder.parameters():
        param.requires_grad = False
    autoencoder.eval()

    optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=config.DIFFUSION_LR)

    for epoch in range(config.DIFFUSION_EPOCHS):
        diffusion_model.train()
        epoch_loss = 0.0

        for cbct, ct in train_loader:
            cbct = cbct.to(device)
            ct = ct.to(device)

            with torch.no_grad():
                z = autoencoder.encoder(ct)

            loss = diffusion.training_loss(diffusion_model, z, cbct)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        print(f"[Diffusion] Epoch {epoch + 1}/{config.DIFFUSION_EPOCHS} | Loss: {epoch_loss / len(train_loader):.6f}")

    torch.save(diffusion_model.state_dict(), config.DIFFUSION_CHECKPOINT)
    print(f"Saved diffusion model checkpoint to {config.DIFFUSION_CHECKPOINT}")

def main():
    set_seed()
    torch.backends.cudnn.benchmark = True

    print("Device:", config.DEVICE)

    train_loader, val_loader, test_loader = get_dataloaders()

    autoencoder = Autoencoder(config.LATENT_CHANNELS).to(config.DEVICE)
    train_autoencoder(autoencoder, train_loader)

    diffusion_model = ConditionalDiffusionModel(config.LATENT_CHANNELS).to(config.DEVICE)
    diffusion = GaussianDiffusion()
    train_diffusion(diffusion_model, autoencoder, diffusion, train_loader)

    print("Training complete.")


if __name__ == "__main__":
    main()