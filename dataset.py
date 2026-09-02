import os
import glob
import random
from functools import partial

import numpy as np
import cv2
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.contrib.concurrent import process_map
from sklearn.model_selection import train_test_split

import config


def find_pairs(data_root=config.DATA_ROOT):
    """Locate matching CBCT/CT volume files, keyed by parent folder (patient)."""
    ct_files, cbct_files = [], []

    for ext in ["*.nii.gz", "*.nii", "*.mha", "*.nrrd"]:
        ct_files.extend(glob.glob(os.path.join(data_root, "**", ext), recursive=True))
        cbct_files.extend(glob.glob(os.path.join(data_root, "**", ext), recursive=True))

    ct_files = [
        f for f in ct_files
        if os.path.basename(f).lower().startswith("ct") or "ct_" in os.path.basename(f).lower()
    ]
    cbct_files = [f for f in cbct_files if "cbct" in os.path.basename(f).lower()]

    def patient_key(path):
        return os.path.dirname(path)

    cbct_dict = {patient_key(f): f for f in cbct_files}
    ct_dict = {patient_key(f): f for f in ct_files}

    common = sorted(set(cbct_dict.keys()) & set(ct_dict.keys()))
    pairs = [{"cbct": cbct_dict[k], "ct": ct_dict[k], "patient": k} for k in common]
    return pairs


def split_pairs(pairs, fraction=config.FRACTION, seed=config.SEED):
    """Subsample patients and split into train/val/test (80/10/10 of the subset)."""
    rng = random.Random(seed)
    pairs_subset = pairs.copy()
    rng.shuffle(pairs_subset)
    n_keep = max(3, int(len(pairs_subset) * fraction))
    pairs_subset = pairs_subset[:n_keep]

    train_pairs, temp_pairs = train_test_split(pairs_subset, test_size=0.20, random_state=seed, shuffle=True)
    val_pairs, test_pairs = train_test_split(temp_pairs, test_size=0.50, random_state=seed, shuffle=True)

    return train_pairs, val_pairs, test_pairs


def load_nifti(path):
    image = sitk.ReadImage(path)
    volume = sitk.GetArrayFromImage(image)
    return volume.astype(np.float32)


def normalize_image(image, hu_min=config.HU_MIN, hu_max=config.HU_MAX):
    image = np.clip(image, hu_min, hu_max)
    image = (image - hu_min) / (hu_max - hu_min)
    return image.astype(np.float32)


def extract_slice_pairs(pair, image_size=config.IMAGE_SIZE, stride=config.STRIDE):
    """Load a CBCT/CT volume pair, normalize, resize, and split into 2D slices."""
    cbct = normalize_image(load_nifti(pair["cbct"]))
    ct = normalize_image(load_nifti(pair["ct"]))

    depth = min(cbct.shape[0], ct.shape[0])
    slices = []

    for i in range(0, depth, stride):
        cbct_slice = cbct[i]
        ct_slice = ct[i]

        if np.std(ct_slice) < 0.01:
            continue

        if cbct_slice.shape != (image_size, image_size):
            cbct_slice = cv2.resize(cbct_slice, (image_size, image_size), interpolation=cv2.INTER_AREA)
        if ct_slice.shape != (image_size, image_size):
            ct_slice = cv2.resize(ct_slice, (image_size, image_size), interpolation=cv2.INTER_AREA)

        slices.append((cbct_slice.astype(np.float16), ct_slice.astype(np.float16)))

    return slices


class CBCTCTDataset(Dataset):
    """2D slice-pair dataset built from paired CBCT/CT volumes."""

    def __init__(self, patient_pairs, image_size=config.IMAGE_SIZE, stride=config.STRIDE, max_workers=None):
        self.image_size = image_size
        max_workers = max_workers or max(1, (os.cpu_count() or 2) - 1)

        print(f"Loading & resizing slices with {max_workers} worker processes...")

        results = process_map(
            partial(extract_slice_pairs, image_size=image_size, stride=stride),
            patient_pairs,
            max_workers=max_workers,
            chunksize=1,
        )

        self.samples = [s for patient_slices in results for s in patient_slices]
        print(f"Total slices loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cbct, ct = self.samples[idx]
        cbct = torch.from_numpy(cbct.astype(np.float32)).unsqueeze(0)
        ct = torch.from_numpy(ct.astype(np.float32)).unsqueeze(0)
        return cbct, ct


def get_dataloaders(data_root=config.DATA_ROOT, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS):
    """Build train/val/test DataLoaders end-to-end from the raw data root."""
    pairs = find_pairs(data_root)
    print(f"Paired patients: {len(pairs)}")

    train_pairs, val_pairs, test_pairs = split_pairs(pairs)
    print(f"Train patients: {len(train_pairs)} | Val patients: {len(val_pairs)} | Test patients: {len(test_pairs)}")

    train_dataset = CBCTCTDataset(train_pairs)
    val_dataset = CBCTCTDataset(val_pairs)
    test_dataset = CBCTCTDataset(test_pairs)

    loader_kwargs = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader