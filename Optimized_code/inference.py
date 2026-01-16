"""
Inference script for NAFNet GAN image denoising.
Processes a directory of noisy images and saves denoised outputs.
"""

import os
import sys
import argparse
import torch
import numpy as np
from skimage import io
from tqdm import tqdm
from pathlib import Path

# Local imports
from config import Config
from models import NAFNet
from utils import get_device, pad_to_multiple, count_parameters


def load_model(config, device):
    """
    Load trained model from checkpoint.
    
    Args:
        config: Configuration object
        device: Device to load model on
        
    Returns:
        Loaded generator model
    """
    print("\n" + "="*60)
    print("LOADING MODEL")
    print("="*60)
    
    # Initialize model
    model = NAFNet(
        img_channel=config.model.img_channel,
        width=config.model.width,
        middle_blk_num=config.model.middle_blk_num,
        enc_blk_nums=config.model.enc_blks,
        dec_blk_nums=config.model.dec_blks
    ).to(device)
    
    count_parameters(model)
    
    # Load checkpoint
    if not hasattr(config, 'inference') or not config.inference.model_path:
        raise ValueError("Inference config with model_path must be specified")
    
    if not os.path.exists(config.inference.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {config.inference.model_path}")
    
    print(f"\nLoading checkpoint: {config.inference.model_path}")
    checkpoint = torch.load(config.inference.model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'generator_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['generator_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        val_loss = checkpoint.get('val_loss', 'unknown')
        print(f"Loaded from epoch {epoch}, val_loss: {val_loss}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("="*60 + "\n")
    
    return model


def process_image(model, img_path, device, padding_multiple=32):
    """
    Process a single image through the model.
    
    Args:
        model: Trained generator model
        img_path: Path to input image
        device: Device
        padding_multiple: Padding multiple for model compatibility
        
    Returns:
        Denoised image as numpy array
    """
    # Load image
    img = io.imread(img_path).astype(np.float32)
    
    # Normalize to [0, 1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
    
    # Pad if necessary
    img_padded, pad_h, pad_w = pad_to_multiple(img_tensor, multiple=padding_multiple)
    
    # Forward pass
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda"):
            pred = model(img_padded)
    
    # Remove padding
    if pad_h > 0 or pad_w > 0:
        pred = pred[:, :, :pred.shape[2]-pad_h, :pred.shape[3]-pad_w]
    
    # Post-process
    pred = torch.clamp(pred, 0, 1)
    pred_np = pred.squeeze().cpu().numpy()
    
    return pred_np


def run_inference(config, device):
    """
    Run inference on all images in input directory.
    
    Args:
        config: Configuration object
        device: Device
    """
    print("\n" + "="*60)
    print("INFERENCE")
    print("="*60)
    
    # Check input directory
    if not os.path.exists(config.inference.input_dir):
        raise FileNotFoundError(f"Input directory not found: {config.inference.input_dir}")
    
    # Create output directory
    os.makedirs(config.inference.output_dir, exist_ok=True)
    
    # Get list of image files
    image_extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
    image_files = sorted([
        f for f in os.listdir(config.inference.input_dir)
        if Path(f).suffix.lower() in image_extensions
    ])
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {config.inference.input_dir}")
    
    print(f"Input directory: {config.inference.input_dir}")
    print(f"Output directory: {config.inference.output_dir}")
    print(f"Found {len(image_files)} images")
    print(f"Save format: {config.inference.save_format}")
    print("="*60 + "\n")
    
    # Load model
    model = load_model(config, device)
    
    # Process images
    print("Processing images...")
    for filename in tqdm(image_files, desc="Inference"):
        input_path = os.path.join(config.inference.input_dir, filename)
        
        try:
            # Process image
            denoised = process_image(
                model, input_path, device,
                padding_multiple=config.inference.padding_multiple
            )
            
            # Determine output filename
            base_name = Path(filename).stem
            output_filename = f"{base_name}.{config.inference.save_format}"
            output_path = os.path.join(config.inference.output_dir, output_filename)
            
            # Save result
            io.imsave(output_path, denoised, check_contrast=False)
            
        except Exception as e:
            print(f"\nError processing {filename}: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print("INFERENCE COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {config.inference.output_dir}\n")


def main(args):
    """Main inference function."""
    
    # Load configuration
    config = Config(args.config)
    
    # Override config with command line arguments if provided
    if args.input_dir:
        if not hasattr(config, 'inference'):
            from config import InferenceConfig
            config.inference = InferenceConfig(
                input_dir=args.input_dir,
                output_dir=args.output_dir or "outputs/predictions",
                model_path=args.model or config.paths.checkpoint_path
            )
        else:
            config.inference.input_dir = args.input_dir
            if args.output_dir:
                config.inference.output_dir = args.output_dir
            if args.model:
                config.inference.model_path = args.model
    
    # Validate inference config
    if not hasattr(config, 'inference'):
        raise ValueError(
            "Inference configuration not found. "
            "Either specify --input_dir or provide a config with inference section."
        )
    
    # Get device
    device = get_device(use_cuda=not args.cpu)
    
    # Run inference
    run_inference(config, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with trained NAFNet GAN")
    
    parser.add_argument('--config', type=str, default='configs/default_config.json',
                        help='Path to configuration JSON file')
    parser.add_argument('--input_dir', type=str,
                        help='Directory containing input images (overrides config)')
    parser.add_argument('--output_dir', type=str,
                        help='Directory to save denoised images (overrides config)')
    parser.add_argument('--model', type=str,
                        help='Path to model checkpoint (overrides config)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage even if CUDA is available')
    
    args = parser.parse_args()
    main(args)
