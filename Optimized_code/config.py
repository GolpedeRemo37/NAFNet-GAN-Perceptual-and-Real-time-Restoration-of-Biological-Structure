"""
Configuration management for NAFNet GAN training and inference.
Loads parameters from JSON and provides structured access.
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class DataConfig:
    """Data-related configuration."""
    train_dir: str
    val_dir: str
    test_dir: str
    crop_size: int = 128
    batch_size_train: int = 4
    batch_size_val: int = 4
    num_workers_train: int = 4
    num_workers_val: int = 2
    test_size: float = 0.04
    augmentation_prob: float = 0.5


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    img_channel: int = 1
    width: int = 16
    middle_blk_num: int = 12
    enc_blks: List[int] = field(default_factory=lambda: [2, 2, 4, 8])
    dec_blks: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    discriminator_type: str = "deep"  # "basic" or "deep"


@dataclass
class LossWeightsConfig:
    """Loss function weights."""
    lambda_lpips: float = 3.0
    lambda_vgg: float = 0.0
    lambda_charb: float = 0.5
    lambda_ssim: float = 0.0
    lambda_lap: float = 2.0
    lambda_edge: float = 0.0
    lambda_fft_cc: float = 0.5
    lambda_fft: float = 0.0
    lambda_gan: float = 0.5
    r1_gamma: float = 0.0


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    num_epochs: int = 30
    learning_rate_g: float = 1e-4
    learning_rate_d: float = 1e-4
    beta1: float = 0.5
    beta2: float = 0.999
    scheduler_t_max: int = 6  # For CosineAnnealingLR
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    visualization_interval: int = 1  # Visualize every N epochs
    
    # Training mode flags
    supervised: bool = True  # True for supervised, False for self-supervised
    use_gan: bool = True  # Enable GAN training
    loss_type: str = "enhanced_deblur"  # "enhanced_deblur" or "combined_criterion"


@dataclass
class PathsConfig:
    """File paths for models and outputs."""
    checkpoint_path: str = "checkpoints/NAFNet_GAN_best_model.pth"
    latest_checkpoint: str = "checkpoints/NAFNet_GAN_latest_model.pth"
    pretrained_path: Optional[str] = None
    output_dir: str = "outputs"
    visualization_dir: str = "training_visuals"
    log_file: str = "training_log.csv"


@dataclass
class InferenceConfig:
    """Inference-specific configuration."""
    input_dir: str
    output_dir: str
    model_path: str
    batch_size: int = 1
    padding_multiple: int = 32
    save_format: str = "tif"  # "tif" or "png"


class Config:
    """Master configuration class."""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path and os.path.exists(config_path):
            self.load_from_json(config_path)
        else:
            # Default configuration
            self.data = DataConfig(
                train_dir="data/train",
                val_dir="data/val",
                test_dir="data/test"
            )
            self.model = ModelConfig()
            self.loss_weights = LossWeightsConfig()
            self.training = TrainingConfig()
            self.paths = PathsConfig()
    
    def load_from_json(self, config_path: str):
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        self.data = DataConfig(**config_dict.get('data', {}))
        self.model = ModelConfig(**config_dict.get('model', {}))
        self.loss_weights = LossWeightsConfig(**config_dict.get('loss_weights', {}))
        self.training = TrainingConfig(**config_dict.get('training', {}))
        self.paths = PathsConfig(**config_dict.get('paths', {}))
        
        # Handle inference config if present
        if 'inference' in config_dict:
            self.inference = InferenceConfig(**config_dict['inference'])
    
    def save_to_json(self, config_path: str):
        """Save configuration to JSON file."""
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'loss_weights': self.loss_weights.__dict__,
            'training': self.training.__dict__,
            'paths': self.paths.__dict__
        }
        
        if hasattr(self, 'inference'):
            config_dict['inference'] = self.inference.__dict__
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    def create_directories(self):
        """Create necessary directories for training/inference."""
        dirs = [
            self.paths.output_dir,
            self.paths.visualization_dir,
            os.path.dirname(self.paths.checkpoint_path),
            os.path.dirname(self.paths.log_file)
        ]
        
        for dir_path in dirs:
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
    
    def print_config(self):
        """Print current configuration."""
        print("\n" + "="*60)
        print("CONFIGURATION")
        print("="*60)
        
        print("\n[DATA]")
        for key, value in self.data.__dict__.items():
            print(f"  {key}: {value}")
        
        print("\n[MODEL]")
        for key, value in self.model.__dict__.items():
            print(f"  {key}: {value}")
        
        print("\n[TRAINING]")
        for key, value in self.training.__dict__.items():
            print(f"  {key}: {value}")
        
        print("\n[LOSS WEIGHTS]")
        for key, value in self.loss_weights.__dict__.items():
            print(f"  {key}: {value}")
        
        print("\n[PATHS]")
        for key, value in self.paths.__dict__.items():
            print(f"  {key}: {value}")
        
        print("="*60 + "\n")


def create_default_config(output_path: str = "configs/default_config.json"):
    """Create a default configuration file."""
    config = Config()
    config.save_to_json(output_path)
    print(f"Default configuration saved to: {output_path}")
    return config


if __name__ == "__main__":
    # Create default config
    create_default_config()
