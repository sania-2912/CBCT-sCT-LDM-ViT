# CBCT-to-sCT: Latent Diffusion Model with ViT Bottleneck

Synthetic CT (sCT) generation from Cone-Beam CT (CBCT) using a two-stage
pipeline:

1. **Autoencoder** — compresses 256×256 CT slices into a compact latent
   space (4×32×32).
2. **Conditional Latent Diffusion Model** — a DDPM with a ViT bottleneck
   that learns to denoise CT latents conditioned on the paired CBCT slice.
   At inference, it generates a CT latent from pure noise conditioned on a
   real CBCT slice, which the autoencoder's decoder turns into an sCT image.

Trained and evaluated on the [SynthRAD2025 Task 2](https://zenodo.org/records/15373853) dataset.


## 1. File structure

CBCT-to-sCT/
│
├── models/
│   ├── __init__.py          # empty, makes `models` a package
│   ├── autoencoder.py       # ResBlock, Encoder, Decoder, Autoencoder
│   ├── unet.py               # ConditionEncoder, TimeEmbedding, ConditionalDiffusionModel
│   ├── vit.py                 # TransformerBlock, ViTBottleneck
│   └── diffusion.py         # GaussianDiffusion: noise schedule, add_noise, training loss, sampling
│
├── dataset.py                # NIfTI loading, normalization, slicing, CBCTCTDataset, get_dataloaders
├── config.py                  # all paths, hyperparameters, device
├── main.py                    # trains autoencoder, then diffusion model; saves checkpoints
├── evaluate.py                # runs inference on test set, computes PSNR/SSIM/LPIPS/FID/DSC
├── requirements.txt
├── README.md
│
├── data/                       # you create this — raw dataset lives here
│   └── SynthRAD2025_Task2/
│       └── <patient folders with CT + CBCT volumes>
│
├── checkpoints/               # auto-created — trained model weights land here
│   ├── autoencoder.pt
│   └── diffusion_model.pt
│
└── results/                    # auto-created — evaluation CSVs land here
    ├── ldm_vit_test_per_slice.csv
    └── ldm_vit_summary.csv

## 2. Prerequisites

- Python 3.9–3.11
- A CUDA-capable GPU is strongly recommended (the diffusion model does
  1000-step sampling per batch at eval time — this is slow on CPU)
- ~15–30 GB free disk space for the dataset, depending on how much you
  extract
- VS Code with the Python extension (optional but assumed, given your setup)

## 3. Installation

```bash
# from inside CBCT-to-sCT/
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

If you have a CUDA GPU, install the CUDA build of PyTorch instead of the
default CPU wheel that `requirements.txt` may pull in — check
https://pytorch.org/get-started/locally/ for the right command for your
CUDA version, e.g.:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 4. Get the data


```bash
mkdir -p data
cd data
#upload the data as downloaded from the pc directly no need to preprocess it separately
wget -c "https://zenodo.org/records/15373853/files/synthRAD2025_Task2_Train.zip?download=1" \
  -O synthRAD2025_Task2_Train.zip

mkdir -p SynthRAD2025_Task2
unzip -q synthRAD2025_Task2_Train.zip -d SynthRAD2025_Task2

cd ..
```

(No `wget`/`unzip` on Windows: download the zip from the Zenodo link in a
browser and extract it into `data/SynthRAD2025_Task2/` with File Explorer.)

**Verify pairing works** before a full run — from the project root:

```bash
python -c "from dataset import find_pairs; p = find_pairs(); print(len(p), 'paired patients'); print(p[0] if p else 'NO PAIRS FOUND')"
```

If this prints `0 paired patients`, your folder structure doesn't match
what `find_pairs()` expects (CT filename starting with `ct` or containing
`ct_`, CBCT filename containing `cbct`, both inside the same per-patient
folder). Adjust `DATA_ROOT` in `config.py` or the matching logic in
`dataset.py` to fit however the archive actually unzipped for you — check
with:

```bash
find data/SynthRAD2025_Task2 -maxdepth 4 | head -50
```

---

## 5. Configure

Open `config.py` and check/adjust before your first run:

| Setting | Notebook default | Notes |
|---|---|---|
| `DATA_ROOT` | `./data/SynthRAD2025_Task2` | or set env var `DATA_ROOT=...` instead of editing the file |
| `FRACTION` | `0.5` | fraction of patients used — raise for a full run, lower for a quick smoke test |
| `IMAGE_SIZE` | `256` | |
| `BATCH_SIZE` | `8` | lower this first if you hit CUDA OOM |
| `AE_EPOCHS` | `25` | autoencoder training epochs |
| `DIFFUSION_EPOCHS` | `50` | diffusion model training epochs |
| `NUM_TIMESTEPS` | `1000` | DDPM steps — also controls how slow sampling/eval is |

For a first smoke test, it's worth temporarily setting `FRACTION = 0.05` or
so and `AE_EPOCHS = 2`, `DIFFUSION_EPOCHS = 2`, just to confirm the whole
pipeline runs end-to-end before committing to a full multi-hour run.

---

## 6. Run

### Step 1 — Train

```bash
python main.py
```

This will:
1. Build train/val/test DataLoaders from `DATA_ROOT` (this step is slow the
   first time — it extracts and resizes every 2D slice from every volume;
   grab a coffee).
2. Train the autoencoder for `AE_EPOCHS`, printing loss per epoch, then
   save `checkpoints/autoencoder.pt`.
3. Freeze the autoencoder, train the diffusion model for
   `DIFFUSION_EPOCHS`, printing loss per epoch, then save
   `checkpoints/diffusion_model.pt`.

Expect this to take hours on GPU depending on `FRACTION`, dataset size, and
epoch counts — the dataset construction alone (slicing + resizing every
volume) can take a while on first run.

### Step 2 — Evaluate

```bash
python evaluate.py
```

This will:
1. Rebuild the test DataLoader.
2. Load both checkpoints from `checkpoints/`.
3. Run full 1000-step DDPM sampling on every test batch (this is the
   slowest part — each slice requires 1000 forward passes through the
   diffusion model).
4. Compute PSNR, SSIM, LPIPS, FID, and Dice score per slice.
5. Print a summary and save:
   - `results/ldm_vit_test_per_slice.csv` — per-slice metrics
   - `results/ldm_vit_summary.csv` — mean/std summary + FID

---

## 7. Running in VS Code

- Open the `CBCT-to-sCT/` folder as the workspace root.
- Select the `.venv` interpreter (Command Palette → "Python: Select
  Interpreter").
- Run `main.py` and `evaluate.py` either via the Run button or a terminal
  inside VS Code (`python main.py`) — no notebook cells, no Colab-specific
  code paths, so this should run identically to a normal Python script.
- If you want cell-by-cell interactivity while debugging, you can still add
  `# %%` markers to turn sections of `main.py` into a VS Code interactive
  Python file, but the scripts are written to run straight through as-is.
