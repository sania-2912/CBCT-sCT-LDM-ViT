import lpips
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchmetrics.image.fid import FrechetInceptionDistance

import config
from dataset import get_dataloaders
from models.autoencoder import Autoencoder
from models.unet import ConditionalDiffusionModel
from models.diffusion import GaussianDiffusion


def evaluate_psnr_ssim(prediction, target):
    pred = prediction.squeeze()
    gt = target.squeeze()
    psnr = peak_signal_noise_ratio(gt, pred, data_range=2)
    ssim = structural_similarity(gt, pred, data_range=2)
    return psnr, ssim


def dice_score(prediction, target, threshold=0.5):
    prediction = (prediction > threshold).float()
    target = (target > threshold).float()
    intersection = (prediction * target).sum()
    return 2 * intersection / (prediction.sum() + target.sum() + 1e-8)


def run_inference(diffusion_model, autoencoder, diffusion, test_loader, device=config.DEVICE):
    predictions, targets = [], []

    diffusion_model.eval()
    for cbct, ct in test_loader:
        cbct = cbct.to(device)
        with torch.no_grad():
            sct = diffusion.sample(diffusion_model, cbct, autoencoder)
        predictions.append(sct.cpu())
        targets.append(ct)

    return predictions, targets


def compute_metrics(predictions, targets, device=config.DEVICE):
    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    per_slice_rows = []
    real_imgs_for_fid, fake_imgs_for_fid = [], []

    for pred, tgt in zip(predictions, targets):
        pred = pred.clamp(0, 1)
        tgt = tgt.clamp(0, 1)

        for i in range(pred.shape[0]):
            p = pred[i:i + 1]
            g = tgt[i:i + 1]

            psnr_val, ssim_val = evaluate_psnr_ssim(p.numpy(), g.numpy())

            with torch.no_grad():
                lpips_val = lpips_model(
                    p.to(device).repeat(1, 3, 1, 1), g.to(device).repeat(1, 3, 1, 1)
                ).mean().item()

            dsc_val = dice_score(p, g).item()

            per_slice_rows.append({"psnr": psnr_val, "ssim": ssim_val, "lpips": lpips_val, "dsc": dsc_val})

            real_imgs_for_fid.append(g.repeat(1, 3, 1, 1))
            fake_imgs_for_fid.append(p.repeat(1, 3, 1, 1))

    fid_batch = 16
    for i in range(0, len(real_imgs_for_fid), fid_batch):
        real_batch = torch.cat(real_imgs_for_fid[i:i + fid_batch]).to(device)
        fake_batch = torch.cat(fake_imgs_for_fid[i:i + fid_batch]).to(device)
        fid.update(real_batch, real=True)
        fid.update(fake_batch, real=False)

    fid_score = fid.compute().item()

    results_df = pd.DataFrame(per_slice_rows)

    summary = {
        "model": "LDM_ViT_proposed",
        "n_test_slices": len(results_df),
        "psnr_mean": results_df["psnr"].mean(),
        "psnr_std": results_df["psnr"].std(),
        "ssim_mean": results_df["ssim"].mean(),
        "ssim_std": results_df["ssim"].std(),
        "lpips_mean": results_df["lpips"].mean(),
        "lpips_std": results_df["lpips"].std(),
        "dsc_mean": results_df["dsc"].mean(),
        "dsc_std": results_df["dsc"].std(),
        "fid": fid_score,
    }

    return results_df, summary


def main():
    print("Device:", config.DEVICE)

    _, _, test_loader = get_dataloaders()

    autoencoder = Autoencoder(config.LATENT_CHANNELS).to(config.DEVICE)
    autoencoder.load_state_dict(torch.load(config.AE_CHECKPOINT, map_location=config.DEVICE))
    autoencoder.eval()

    diffusion_model = ConditionalDiffusionModel(config.LATENT_CHANNELS).to(config.DEVICE)
    diffusion_model.load_state_dict(torch.load(config.DIFFUSION_CHECKPOINT, map_location=config.DEVICE))
    diffusion_model.eval()

    diffusion = GaussianDiffusion()

    predictions, targets = run_inference(diffusion_model, autoencoder, diffusion, test_loader)
    results_df, summary = compute_metrics(predictions, targets)

    print(pd.Series(summary))

    per_slice_path = f"{config.RESULTS_DIR}/ldm_vit_test_per_slice.csv"
    summary_path = f"{config.RESULTS_DIR}/ldm_vit_summary.csv"

    results_df.to_csv(per_slice_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print(f"\nSaved: {per_slice_path}, {summary_path}")


if __name__ == "__main__":
    main()