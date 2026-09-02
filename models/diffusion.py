import torch
import torch.nn.functional as F

import config

class GaussianDiffusion:
    """DDPM forward process (noise schedule + noising) and reverse sampling,
    operating in the autoencoder's latent space."""

    def __init__(
        self,
        num_timesteps=config.NUM_TIMESTEPS,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
        device=config.DEVICE,
    ):
        self.num_timesteps = num_timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, z, t):
        """Forward diffusion: sample z_t from z_0 given timestep t."""
        noise = torch.randn_like(z)
        alpha_t = self.alpha_hat[t].view(-1, 1, 1, 1)
        noisy_z = torch.sqrt(alpha_t) * z + torch.sqrt(1 - alpha_t) * noise
        return noisy_z, noise

    def training_loss(self, model, z, cbct):
        """Sample a random timestep, noise the latent, predict the noise,
        and return the MSE loss for one training step."""
        t = torch.randint(0, self.num_timesteps, (z.shape[0],), device=self.device)
        noisy_z, noise = self.add_noise(z, t)
        predicted_noise = model(noisy_z, t, cbct)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, model, cbct, autoencoder, latent_shape=(4, 32, 32)):
        """Reverse diffusion (DDPM ancestral sampling) conditioned on a CBCT
        image, then decode the resulting latent back into image space."""
        model.eval()
        B = cbct.shape[0]

        z = torch.randn(B, *latent_shape, device=self.device)

        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((B,), t, device=self.device, dtype=torch.long)

            predicted_noise = model(z, t_tensor, cbct)

            alpha = self.alphas[t]
            alpha_bar = self.alpha_hat[t]
            beta = self.betas[t]

            z = (1 / torch.sqrt(alpha)) * (
                z - (beta / torch.sqrt(1 - alpha_bar)) * predicted_noise
            )

            if t > 0:
                noise = torch.randn_like(z)
                z += torch.sqrt(beta) * noise

        sct = autoencoder.decoder(z)
        return sct