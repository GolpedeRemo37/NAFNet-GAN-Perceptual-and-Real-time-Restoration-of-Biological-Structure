"""
Utility functions for data loading, visualization, and model management.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import pandas as pd
from torch.utils.data import DataLoader
from skimage import io


def plot_images(noisy, pred, target, epoch, save_dir="training_visuals", show=True):
    """
    Saves and displays input, predicted, and ground truth images with colorbars.
    
    Args:
        noisy: Noisy input tensor
        pred: Model prediction tensor
        target: Ground truth tensor
        epoch: Current epoch number
        save_dir: Directory to save visualization
        show: Whether to display the plot
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Noisy Input", "Denoised Output", "Ground Truth"]
    
    # Move to CPU and detach for plotting
    images = [
        noisy.squeeze().cpu().detach().numpy(),
        pred.squeeze().cpu().detach().numpy(),
        target.squeeze().cpu().detach().numpy()
    ]
    
    for i, img in enumerate(images):
        im = axs[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axs[i].set_title(f"{titles[i]} (Epoch {epoch})")
        axs[i].axis("off")
        fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    
    # Save
    save_path = os.path.join(save_dir, f"epoch_{epoch:03d}_viz.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    plt.close()


def visualize_dataloader(loader, title="Sample", save_path=None):
    """
    Visualize a sample from a dataloader.
    
    Args:
        loader: DataLoader instance
        title: Title for the visualization
        save_path: Optional path to save the figure
    """
    noisy_img_batch, gt_img_batch = next(iter(loader))
    print(f"Noisy batch shape: {noisy_img_batch.shape}, GT batch shape: {gt_img_batch.shape}")

    noisy_sample = noisy_img_batch[0].squeeze().cpu().numpy()
    gt_sample = gt_img_batch[0].squeeze().cpu().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    im0 = axs[0].imshow(noisy_sample, cmap='gray')
    axs[0].set_title(f"Noisy Input - {title}")
    axs[0].axis("off")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    
    im1 = axs[1].imshow(gt_sample, cmap='gray')
    axs[1].set_title(f"Ground Truth - {title}")
    axs[1].axis("off")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    plt.close()


def load_checkpoint(model, checkpoint_path, device, discriminator=None, 
                    g_optimizer=None, d_optimizer=None, strict=True):
    """
    Load model checkpoint with optional discriminator and optimizers.
    
    Args:
        model: Generator model
        checkpoint_path: Path to checkpoint file
        device: Device to load on
        discriminator: Optional discriminator model
        g_optimizer: Optional generator optimizer
        d_optimizer: Optional discriminator optimizer
        strict: Whether to strictly enforce state dict keys match
        
    Returns:
        Loaded epoch number and validation loss
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return 0, float('inf')
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load generator
    if 'generator_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['generator_state_dict'], strict=strict)
    else:
        model.load_state_dict(checkpoint, strict=strict)
    
    # Load discriminator if provided
    if discriminator and 'discriminator_state_dict' in checkpoint:
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'], strict=strict)
    
    # Load optimizers if provided
    if g_optimizer and 'g_optimizer_state_dict' in checkpoint:
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    
    if d_optimizer and 'd_optimizer_state_dict' in checkpoint:
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
    
    # Get metadata
    epoch = checkpoint.get('epoch', 0)
    val_loss = checkpoint.get('val_loss', float('inf'))
    
    print(f"Loaded checkpoint from epoch {epoch} with val_loss: {val_loss:.4f}")
    
    return epoch, val_loss


def save_checkpoint(model, checkpoint_path, discriminator=None, 
                    g_optimizer=None, d_optimizer=None, 
                    epoch=0, val_loss=float('inf'), **kwargs):
    """
    Save model checkpoint with optional discriminator and optimizers.
    
    Args:
        model: Generator model
        checkpoint_path: Path to save checkpoint
        discriminator: Optional discriminator model
        g_optimizer: Optional generator optimizer
        d_optimizer: Optional discriminator optimizer
        epoch: Current epoch number
        val_loss: Current validation loss
        **kwargs: Additional items to save
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    checkpoint = {
        'generator_state_dict': model.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss,
    }
    
    if discriminator:
        checkpoint['discriminator_state_dict'] = discriminator.state_dict()
    
    if g_optimizer:
        checkpoint['g_optimizer_state_dict'] = g_optimizer.state_dict()
    
    if d_optimizer:
        checkpoint['d_optimizer_state_dict'] = d_optimizer.state_dict()
    
    # Add any additional items
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, checkpoint_path)


def save_loss_history(train_g_losses, train_d_losses, val_g_losses, 
                      num_epochs, output_path):
    """
    Save training history to CSV.
    
    Args:
        train_g_losses: List of training generator losses
        train_d_losses: List of training discriminator losses
        val_g_losses: List of validation generator losses
        num_epochs: Total number of epochs
        output_path: Path to save CSV file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    loss_df = pd.DataFrame({
        'Epoch': range(1, num_epochs + 1),
        'Train_Generator_Loss': train_g_losses,
        'Train_Discriminator_Loss': train_d_losses,
        'Val_Generator_Loss': val_g_losses
    })
    
    loss_df.to_csv(output_path, index=False)
    print(f"Loss history saved to: {output_path}")


def get_device(use_cuda=True):
    """
    Get available device (CUDA/CPU).
    
    Args:
        use_cuda: Whether to use CUDA if available
        
    Returns:
        torch.device object
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device


def count_parameters(model):
    """Count trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total:,}")
    return total


def pad_to_multiple(x, multiple=32):
    """
    Pads image tensor to be divisible by 'multiple' (required for NAFNet).
    
    Args:
        x: Input tensor (B, C, H, W)
        multiple: Padding multiple (default: 32)
        
    Returns:
        Padded tensor and padding amounts (pad_h, pad_w)
    """
    h, w = x.shape[2], x.shape[3]
    H = ((h + multiple - 1) // multiple) * multiple
    W = ((w + multiple - 1) // multiple) * multiple
    pad_h = H - h
    pad_w = W - w
    
    # Pad using reflection to minimize edge artifacts
    x_padded = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    
    return x_padded, pad_h, pad_w


def check_nan_inf(tensor, name="tensor"):
    """Check if tensor contains NaN or Inf values."""
    if torch.isnan(tensor).any():
        print(f"WARNING: {name} contains NaN values!")
        return True
    if torch.isinf(tensor).any():
        print(f"WARNING: {name} contains Inf values!")
        return True
    return False


def get_lr(optimizer):
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_time(seconds):
    """Format seconds into readable time string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
