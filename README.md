# NAFNet-GAN: Perceptual and Real-time Restoration of Biological Structure

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange) ![Status](https://img.shields.io/badge/Status-Active-green)

## Description

A high-performance restoration framework for biological imagery using **NAFNet** (Nonlinear Activation-Free Network) combined with **GAN** and **Perceptual** losses. Designed for real-time inference and recovering fine structural details.

## Overview

This repository contains the implementation of a hybrid deep learning model for image restoration (denoising, deblurring). It integrates the efficiency of **NAFNet** as a generator with a **Deep Discriminator** and a comprehensive **MasterLoss** (combining Charbonnier, Laplacian, and VGG-based perceptual losses).

The project supports both **paired training** (Clean $\leftrightarrow$ Noisy) and **self-supervised training** (synthetic noise generation), making it adaptable to various biological datasets where ground truth might be scarce.

**Key Features**:
- **NAFNet Generator**: Uses `SimpleGate` and `LayerNorm2d` for activation-free, low-latency restoration.
- **Adversarial Training**: Hinge-loss based GAN training to preserve high-frequency textures.
- **Hybrid Loss Landscape**: Optimizes for pixel fidelity (Charbonnier), edge preservation (Laplacian), and perceptual quality (LPIPS/VGG).
- **Dual Data Pipelines**: 
  - `dataloader.py`: Standard paired loading.
  - `dataloader_selfsup.py`: On-the-fly synthetic noise generation (Gaussian blur/noise).
- **Inference Optimization**: Automated padding/unpadding to handle arbitrary image resolutions.

## Requirements

- **Python**: 3.8 or higher
- **Packages**:

```text
torch>=2.0.0
torchvision>=0.15.0
numpy
pandas
scikit-image
tqdm
matplotlib
torchmetrics
```

**Hardware**: Developed and tested on: AMD Ryzen 7 5800X / 32GB RAM / NVIDIA RTX 5060 Ti (16GB). CUDA support is strongly recommended for training.

**Directory Structure**: 
```
.
├── dataloader.py          # Dataset class for loading paired (Noisy, GT) images
├── dataloader_selfsup.py  # Dataset class for self-supervised learning (synthetic noise)
├── losses.py              # Custom loss modules (MasterLoss, Charbonnier, Laplacian, VGG/Perceptual)
├── models.py              # NAFNet Generator and Deep Discriminator architectures
├── Main_Notebook.ipynb    # Main training loop, model initialization, and visualization
├── Inference.ipynb        # Inference script for restoring new images (handles padding)
└── training_visuals/      # Directory for saving training progress and validation outputs
```
**Storage**: 10GB+ free space for outputs.

## Usage
### 1. Training
Training is primarily handled in `Main_Notebook.ipynb.`
Setup: Open the notebook and install necessary dependencies.

**Dataset Selection:**
- Use from `dataloader` import `DenoisingDataset2D` for datasets with existing Noisy/GT pairs.
- Use from `dataloader_selfsup` import `DenoisingDataset2D` if you only have clean images and want to train with synthetic degradations.
**Configuration:** Adjust paths to your data `(INPUT_DIR, TARGET_DIR)` in the setup cells.
**Execution:** Run the cells to initialize NAFNet, Deep_Discriminator, and MasterLoss.

The training loop automatically saves checkpoints and visual comparisons to ./training_visuals.

### 2. Inference
To run the model on new, unseen data, use `Inference.ipynb`.
**Load Model:** Specify the path to your trained .pth checkpoint.
**Set Directories:** Define `INPUT_DIR` (noisy images) and `OUTPUT_DIR`.
**Run:** Execute the notebook.
**Note on Padding:** The script includes a pad_to_multiple function to ensure input dimensions are multiples of 32 (required by NAFNet) and crops the output back to the original size automatically.

## Architecture Details
### Generator (NAFNet)
Defined in `models.py`. It replaces traditional non-linear activations with a logic-gate inspired mechanism: $$ \text{SimpleGate}(X, Y) = X \odot Y $$ This design (from Chu et al.) reduces computational complexity while maintaining global receptive fields.

### Loss Function
Defined in `losses.py` as MasterLoss. It aggregates:
- Pixel Loss: Charbonnier Loss (robust L1).
- Frequency Loss: Laplacian Pyramid Loss.
- Perceptual Loss: VGG16 features (via torchmetrics).
- Adversarial Loss: Hinge loss via the Deep_Discriminator.

## Acknowledgments
NAFNet: Architecture based on "NAFSSR: Stereo Image Super-Resolution Using NAFNet".
LPIPS/SSIM: Metrics calculated using torchmetrics.

**NAFNet original paper citation:**

```bibtex
@InProceedings{chu2022nafssr,
    author    = {Chu, Xiaojie and Chen, Liangyu and Yu, Wenqing},
    title     = {NAFSSR: Stereo Image Super-Resolution Using NAFNet},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {1239-1248}
}
