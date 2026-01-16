"""
Main training script for NAFNet GAN image denoising.
Supports both supervised and self-supervised training modes.
"""

import os
import sys
import argparse
import warnings
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Local imports
from config import Config, LossWeightsConfig
from models import NAFNet, Deep_Discriminator, Discriminator
from dataloader import DenoisingDataset2D
from dataloader_selfsup import DenoisingDataset2D as SelfSupDataset
from losses import MasterLoss
from utils import (
    plot_images, save_checkpoint, save_loss_history, 
    get_device, count_parameters, get_lr, AverageMeter, format_time
)

warnings.filterwarnings('ignore')


def load_data(config):
    """
    Load training and validation dataloaders.
    
    Args:
        config: Configuration object
        
    Returns:
        train_loader, val_loader
    """
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    # Select dataset based on supervised/self-supervised mode
    if config.training.supervised:
        print("Mode: SUPERVISED")
        DatasetClass = DenoisingDataset2D
        
        # Load paths
        train_noisy = sorted([
            os.path.join(config.data.train_dir, "RAW", f) 
            for f in os.listdir(os.path.join(config.data.train_dir, "RAW")) 
            if f.endswith('.tif')
        ])
        train_gt = sorted([
            os.path.join(config.data.train_dir, "GT", f) 
            for f in os.listdir(os.path.join(config.data.train_dir, "GT")) 
            if f.endswith('.tif')
        ])
        val_noisy = sorted([
            os.path.join(config.data.val_dir, "RAW", f) 
            for f in os.listdir(os.path.join(config.data.val_dir, "RAW")) 
            if f.endswith('.tif')
        ])
        val_gt = sorted([
            os.path.join(config.data.val_dir, "GT", f) 
            for f in os.listdir(os.path.join(config.data.val_dir, "GT")) 
            if f.endswith('.tif')
        ])
        
        print(f"Training: {len(train_noisy)} image pairs")
        print(f"Validation: {len(val_noisy)} image pairs")
        
        # Create datasets
        train_ds = DatasetClass(
            train_noisy, train_gt, 
            crop_size=config.data.crop_size, 
            augment=True,
            p=config.data.augmentation_prob
        )
        val_ds = DatasetClass(
            val_noisy, val_gt, 
            crop_size=None, 
            augment=False
        )
        
    else:
        print("Mode: SELF-SUPERVISED")
        DatasetClass = SelfSupDataset
        
        # Load only GT paths for self-supervised
        train_gt = sorted([
            os.path.join(config.data.train_dir, "GT", f) 
            for f in os.listdir(os.path.join(config.data.train_dir, "GT")) 
            if f.endswith('.tif')
        ])
        val_gt = sorted([
            os.path.join(config.data.val_dir, "GT", f) 
            for f in os.listdir(os.path.join(config.data.val_dir, "GT")) 
            if f.endswith('.tif')
        ])
        
        print(f"Training: {len(train_gt)} images")
        print(f"Validation: {len(val_gt)} images")
        
        # Create datasets with synthetic degradation
        train_ds = DatasetClass(
            train_gt, 
            crop_size=config.data.crop_size, 
            augment=True,
            mode="train"
        )
        val_ds = DatasetClass(
            val_gt, 
            crop_size=None, 
            augment=True,
            mode="val"
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=config.data.batch_size_train, 
        shuffle=True, 
        num_workers=config.data.num_workers_train, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=config.data.batch_size_val, 
        shuffle=False, 
        num_workers=config.data.num_workers_val, 
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print("="*60 + "\n")
    
    return train_loader, val_loader


def setup_models(config, device):
    """
    Initialize generator and discriminator models.
    
    Args:
        config: Configuration object
        device: Device to load models on
        
    Returns:
        generator, discriminator (or None if GAN disabled)
    """
    print("\n" + "="*60)
    print("INITIALIZING MODELS")
    print("="*60)
    
    # Initialize Generator (NAFNet)
    generator = NAFNet(
        img_channel=config.model.img_channel,
        width=config.model.width,
        middle_blk_num=config.model.middle_blk_num,
        enc_blk_nums=config.model.enc_blks,
        dec_blk_nums=config.model.dec_blks
    ).to(device)
    
    print(f"Generator (NAFNet):")
    count_parameters(generator)
    
    # Initialize Discriminator (if using GAN)
    discriminator = None
    if config.training.use_gan:
        if config.model.discriminator_type == "deep":
            discriminator = Deep_Discriminator(in_channels=config.model.img_channel).to(device)
        else:
            discriminator = Discriminator(in_channels=config.model.img_channel).to(device)
        
        print(f"\nDiscriminator ({config.model.discriminator_type}):")
        count_parameters(discriminator)
    else:
        print("\nGAN training disabled - No discriminator")
    
    print("="*60 + "\n")
    
    return generator, discriminator


def train_epoch(train_loader, generator, discriminator, criterion, 
                g_optimizer, d_optimizer, scaler_g, scaler_d, 
                device, config, epoch):
    """
    Train for one epoch.
    
    Args:
        train_loader: Training dataloader
        generator: Generator model
        discriminator: Discriminator model (or None)
        criterion: Loss function
        g_optimizer: Generator optimizer
        d_optimizer: Discriminator optimizer (or None)
        scaler_g: Generator gradient scaler
        scaler_d: Discriminator gradient scaler (or None)
        device: Device
        config: Configuration object
        epoch: Current epoch number
        
    Returns:
        Average generator loss, average discriminator loss
    """
    generator.train()
    if discriminator:
        discriminator.train()
    
    g_loss_meter = AverageMeter()
    d_loss_meter = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{config.training.num_epochs} [Train]')
    
    for noisy_img, clean_img in pbar:
        noisy_img = noisy_img.to(device)
        clean_img = clean_img.to(device)
        
        # Skip batch if contains NaN/Inf
        if torch.isnan(noisy_img).any() or torch.isinf(noisy_img).any():
            continue
        
        batch_size = noisy_img.size(0)
        
        # ==================== Train Discriminator ====================
        if config.training.use_gan and discriminator:
            d_optimizer.zero_grad()
            
            with autocast(enabled=config.training.mixed_precision):
                fake_img = generator(noisy_img)
                fake_img = torch.clamp(fake_img, 0, 1)
                
                d_real = discriminator(clean_img)
                d_fake = discriminator(fake_img.detach())
                
                d_loss = criterion.forward_discriminator(d_real, d_fake)
            
            if not (torch.isnan(d_loss) or torch.isinf(d_loss)):
                scaler_d.scale(d_loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    discriminator.parameters(), 
                    max_norm=config.training.gradient_clip
                )
                scaler_d.step(d_optimizer)
                scaler_d.update()
                
                d_loss_meter.update(d_loss.item(), batch_size)
        
        # ==================== Train Generator ====================
        g_optimizer.zero_grad()
        
        with autocast(enabled=config.training.mixed_precision):
            fake_img = generator(noisy_img)
            fake_img = torch.clamp(fake_img, 0, 1)
            
            # Prepare loss inputs
            g_loss_inputs = {
                'pred_img': fake_img,
                'target_img': clean_img
            }
            
            # Add discriminator logits if using GAN
            if config.training.use_gan and discriminator:
                d_fake_for_g = discriminator(fake_img)
                g_loss_inputs['d_fake_logits'] = d_fake_for_g
            else:
                # Provide dummy discriminator output if not using GAN
                g_loss_inputs['d_fake_logits'] = torch.zeros(1, device=device)
            
            g_loss = criterion.forward_generator(g_loss_inputs)
        
        if not (torch.isnan(g_loss) or torch.isinf(g_loss)):
            scaler_g.scale(g_loss).backward()
            torch.nn.utils.clip_grad_norm_(
                generator.parameters(), 
                max_norm=config.training.gradient_clip
            )
            scaler_g.step(g_optimizer)
            scaler_g.update()
            
            g_loss_meter.update(g_loss.item(), batch_size)
        
        # Update progress bar
        pbar.set_postfix({
            'G_Loss': g_loss_meter.avg,
            'D_Loss': d_loss_meter.avg if config.training.use_gan else 0.0,
            'LR': f"{get_lr(g_optimizer):.2e}"
        })
    
    return g_loss_meter.avg, d_loss_meter.avg


def validate(val_loader, generator, discriminator, criterion, device, config):
    """
    Validate the model.
    
    Args:
        val_loader: Validation dataloader
        generator: Generator model
        discriminator: Discriminator model (or None)
        criterion: Loss function
        device: Device
        config: Configuration object
        
    Returns:
        Average validation generator loss
    """
    generator.eval()
    if discriminator:
        discriminator.eval()
    
    val_loss_meter = AverageMeter()
    
    with torch.no_grad():
        with autocast(enabled=config.training.mixed_precision):
            for val_input, val_target in tqdm(val_loader, desc='Validation'):
                val_input = val_input.to(device)
                val_target = val_target.to(device)
                
                val_output = generator(val_input)
                val_output = torch.clamp(val_output, 0, 1)
                
                # Prepare loss inputs
                val_inputs = {
                    'pred_img': val_output,
                    'target_img': val_target
                }
                
                if config.training.use_gan and discriminator:
                    d_fake_val = discriminator(val_output)
                    val_inputs['d_fake_logits'] = d_fake_val
                else:
                    val_inputs['d_fake_logits'] = torch.zeros(1, device=device)
                
                val_loss = criterion.forward_generator(val_inputs)
                
                if not (torch.isnan(val_loss) or torch.isinf(val_loss)):
                    val_loss_meter.update(val_loss.item(), val_input.size(0))
    
    return val_loss_meter.avg


def main(args):
    """Main training function."""
    
    # Load configuration
    config = Config(args.config)
    config.create_directories()
    config.print_config()
    
    # Get device
    device = get_device(use_cuda=not args.cpu)
    
    # Load data
    train_loader, val_loader = load_data(config)
    
    # Setup models
    generator, discriminator = setup_models(config, device)
    
    # Setup optimizers
    g_optimizer = optim.Adam(
        generator.parameters(), 
        lr=config.training.learning_rate_g,
        betas=(config.training.beta1, config.training.beta2)
    )
    
    d_optimizer = None
    if config.training.use_gan and discriminator:
        d_optimizer = optim.Adam(
            discriminator.parameters(),
            lr=config.training.learning_rate_d,
            betas=(config.training.beta1, config.training.beta2)
        )
    
    # Setup schedulers
    g_scheduler = CosineAnnealingLR(g_optimizer, T_max=config.training.scheduler_t_max)
    d_scheduler = None
    if d_optimizer:
        d_scheduler = CosineAnnealingLR(d_optimizer, T_max=config.training.scheduler_t_max)
    
    # Setup loss
    loss_weights = LossWeightsConfig(**config.loss_weights.__dict__)
    criterion = MasterLoss(
        loss_type=config.training.loss_type,
        weights=loss_weights,
        device=device
    )
    
    # Setup gradient scalers
    scaler_g = GradScaler(enabled=config.training.mixed_precision)
    scaler_d = GradScaler(enabled=config.training.mixed_precision) if config.training.use_gan else None
    
    # Load pretrained weights if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if config.paths.pretrained_path and os.path.exists(config.paths.pretrained_path):
        print(f"\nLoading pretrained weights from: {config.paths.pretrained_path}")
        checkpoint = torch.load(config.paths.pretrained_path, map_location=device)
        
        if 'generator_state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['generator_state_dict'])
        else:
            generator.load_state_dict(checkpoint)
        
        if discriminator and 'discriminator_state_dict' in checkpoint:
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        start_epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch} with best val loss: {best_val_loss:.4f}\n")
    
    # Create fixed validation batch for consistent visualization
    print("Creating fixed validation batch...")
    try:
        fixed_val_input, fixed_val_target = next(iter(val_loader))
        fixed_val_input = fixed_val_input.to(device)
        fixed_val_target = fixed_val_target.to(device)
    except StopIteration:
        print("ERROR: Validation loader is empty!")
        return
    
    # Training history
    train_g_losses = []
    train_d_losses = []
    val_g_losses = []
    
    # ==================== TRAINING LOOP ====================
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")
    
    start_time = time.time()
    
    for epoch in range(start_epoch + 1, config.training.num_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_g_loss, train_d_loss = train_epoch(
            train_loader, generator, discriminator, criterion,
            g_optimizer, d_optimizer, scaler_g, scaler_d,
            device, config, epoch
        )
        
        # Validate
        val_g_loss = validate(val_loader, generator, discriminator, criterion, device, config)
        
        # Record losses
        train_g_losses.append(train_g_loss)
        train_d_losses.append(train_d_loss)
        val_g_losses.append(val_g_loss)
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"\nEpoch {epoch:03d}/{config.training.num_epochs} | "
              f"Time: {format_time(epoch_time)} | "
              f"Train G: {train_g_loss:.4f} | "
              f"Train D: {train_d_loss:.4f} | "
              f"Val G: {val_g_loss:.4f}")
        
        # Visualize
        if epoch % config.training.visualization_interval == 0:
            generator.eval()
            with torch.no_grad():
                fixed_pred = generator(fixed_val_input)
                fixed_pred = torch.clamp(fixed_pred, 0, 1)
                plot_images(
                    fixed_val_input[0], fixed_pred[0], fixed_val_target[0],
                    epoch, config.paths.visualization_dir, show=False
                )
        
        # Save best model
        if val_g_loss < best_val_loss:
            best_val_loss = val_g_loss
            print(f">>> New Best Model! Val Loss: {best_val_loss:.4f}")
            save_checkpoint(
                generator, config.paths.checkpoint_path,
                discriminator=discriminator,
                g_optimizer=g_optimizer,
                d_optimizer=d_optimizer,
                epoch=epoch,
                val_loss=best_val_loss
            )
        
        # Save latest checkpoint
        save_checkpoint(
            generator, config.paths.latest_checkpoint,
            discriminator=discriminator,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
            epoch=epoch,
            val_loss=val_g_loss
        )
        
        # Step schedulers
        g_scheduler.step()
        if d_scheduler:
            d_scheduler.step()
    
    # ==================== TRAINING COMPLETE ====================
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total time: {format_time(total_time)}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Save loss history
    save_loss_history(
        train_g_losses, train_d_losses, val_g_losses,
        config.training.num_epochs, config.paths.log_file
    )
    
    print(f"\nCheckpoint saved: {config.paths.checkpoint_path}")
    print(f"Loss history saved: {config.paths.log_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NAFNet GAN for image denoising")
    parser.add_argument('--config', type=str, default='configs/default_config.json',
                        help='Path to configuration JSON file')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage even if CUDA is available')
    
    args = parser.parse_args()
    main(args)
