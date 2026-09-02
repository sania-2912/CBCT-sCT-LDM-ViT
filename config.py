import os
import torch

DATA_ROOT = os.environ.get("DATA_ROOT", "./data/SynthRAD2025_Task2")
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR = "./results"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FRACTION = 0.5          
IMAGE_SIZE = 256
STRIDE = 1
HU_MIN = -1000
HU_MAX = 2000

BATCH_SIZE = 8
NUM_WORKERS = min(4, os.cpu_count() or 1)

IN_CHANNELS = 1
LATENT_CHANNELS = 4
BASE_CHANNELS = 64
AE_EPOCHS = 25
AE_LR = 2e-4

NUM_TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02
DIFFUSION_EPOCHS = 50
DIFFUSION_LR = 1e-4

AE_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "autoencoder.pt")
DIFFUSION_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "diffusion_model.pt")