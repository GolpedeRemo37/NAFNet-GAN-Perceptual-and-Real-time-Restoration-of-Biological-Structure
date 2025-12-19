import torch
from torch.utils.data import Dataset
import numpy as np
from skimage import io
import random
import os

class DenoisingDataset2D(Dataset):
    def __init__(self, noisy_paths, gt_paths, crop_size=None, augment=True, p=0.5):
        assert len(noisy_paths) == len(gt_paths), "Noisy and GT paths must have the same length"
        self.noisy_paths = noisy_paths
        self.gt_paths = gt_paths
        self.crop_size = crop_size
        self.augment = augment
        self.p = p
        self.crop_coords = []

    def __getitem__(self, idx):
        noisy = io.imread(self.noisy_paths[idx]).astype(np.float32)
        gt = io.imread(self.gt_paths[idx]).astype(np.float32)
    
        # Normalize
        noisy = (noisy - noisy.min()) / (noisy.max() - noisy.min() + 1e-8)
        gt = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
        
        # Augment (random crop and flips)
        if self.augment:
            h, w = noisy.shape
            crop_h = crop_w = self.crop_size  # Desired crop size for augmentation
            if h < crop_h or w < crop_w:
                raise ValueError(f"Image too small for augmentation crop: ({h}, {w}) at index {idx}")
            
            # Compute safe random crop coordinates
            max_x = h - crop_h
            max_y = w - crop_w
            x = random.randint(0, max_x) if max_x > 0 else 0
            y = random.randint(0, max_y) if max_y > 0 else 0
            noisy = noisy[x:x+crop_h, y:y+crop_w]
            gt = gt[x:x+crop_h, y:y+crop_w]
    
            # Random flips
            if random.random() < self.p:
                noisy = np.fliplr(noisy).copy()
                gt = np.fliplr(gt).copy()
            if random.random() < self.p:
                noisy = np.flipud(noisy).copy()
                gt = np.flipud(gt).copy()
    
        # Convert to tensors
        noisy = torch.from_numpy(noisy.copy()).unsqueeze(0)
        gt = torch.from_numpy(gt.copy()).unsqueeze(0)
        return noisy, gt

    def __len__(self):
        return len(self.noisy_paths)