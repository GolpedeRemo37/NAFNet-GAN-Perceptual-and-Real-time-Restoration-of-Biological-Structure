import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from skimage import io, filters, util
import random, os

class DenoisingDataset2D(Dataset):
    def __init__(self, gt_paths, crop_size=None, augment=True, mode="train"):
        """
        mode: 'train' or 'val'
        - train: random augmentations
        - val: fixed, cyclical augmentations (blur, noise, none)
        """
        self.gt_paths = gt_paths
        self.crop_size = crop_size
        self.augment = augment
        self.mode = mode

        # Pre-assign augmentation types for validation
        if self.mode == "val":
            n = len(gt_paths)
            # Example: 40% blur, 40% noise, 20% clean
            n_blur = int(0.4 * n)
            n_noise = int(0.4 * n)
            n_clean = n - n_blur - n_noise
            self.val_aug_types = (["blur"] * n_blur +
                                  ["noise"] * n_noise +
                                  ["none"] * n_clean)
            random.shuffle(self.val_aug_types)  # fixed once per dataset init

    def __getitem__(self, idx):
        gt = io.imread(self.gt_paths[idx]).astype(np.float32)
        gt = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)

        noisy = gt.copy()

        if self.mode == "train":
            # Random augmentation each call
            aug_type = random.choice(["blur", "noise", "none"])
            if aug_type == "blur":
                sigma = random.uniform(1, 15)
                noisy = filters.gaussian(noisy, sigma=sigma)
            elif aug_type == "noise":
                var = random.uniform(0.01, 0.1)
                noisy = util.random_noise(noisy, mode="gaussian", var=var)
            
            h, w = noisy.shape
            crop_h = crop_w = self.crop_size  # Desired crop size for augmentation
            if h < crop_h or w < crop_w:
                raise ValueError(f"Image too small for augmentation crop: ({h}, {w}) at index {idx}")

            max_x = h - crop_h
            max_y = w - crop_w
            x = random.randint(0, max_x) if max_x > 0 else 0
            y = random.randint(0, max_y) if max_y > 0 else 0
            noisy = noisy[x:x+crop_h, y:y+crop_w]
            gt = gt[x:x+crop_h, y:y+crop_w]

                    
        elif self.mode == "val":
            # Fixed augmentation assignment
            aug_type = self.val_aug_types[idx]
            if aug_type == "blur":
                noisy = filters.gaussian(noisy, sigma=10)  # fixed sigma
            elif aug_type == "noise":
                noisy = util.random_noise(noisy, mode="gaussian", var=0.1)
            # "none" → leave as is

        # Convert to tensors
        noisy = torch.from_numpy(noisy.copy()).unsqueeze(0).float()
        gt = torch.from_numpy(gt.copy()).unsqueeze(0).float()
        return noisy, gt

    def __len__(self):
        return len(self.gt_paths)
