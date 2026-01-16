# NAFNet-GAN: Perceptual and Real-time Restoration of Biological Structure

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange) ![Status](https://img.shields.io/badge/Status-Active-green)

## Description

**NAFNet-GAN** is a high-performance deep learning framework for the restoration of biological and scientific imagery. It integrates the computational efficiency of **NAFNet (Nonlinear Activation-Free Network)** with **adversarial (GAN)** and **perceptual loss functions** to achieve high-fidelity structural recovery under real-time or near–real-time constraints.  
The framework is designed to effectively restore fine-grained biological structures while maintaining low latency and robust generalization across diverse imaging conditions.

## Overview

The proposed system is specifically optimized for:

- **Real-time or near–real-time inference**, suitable for high-throughput imaging pipelines  
- **High-fidelity structural restoration**, preserving biologically meaningful details  
- **Flexible training paradigms**, supporting both paired (supervised) and unpaired (self-supervised) datasets  

Both training and inference workflows are supported in two complementary modes:

- **Interactive execution via Jupyter Notebooks**, enabling rapid prototyping and qualitative analysis  
- **Reproducible execution via terminal-based scripts** (`train.py`, `inference.py`), suitable for large-scale experiments and deployment  

### Key Features

- **NAFNet Generator**  
  Employs `SimpleGate` and `LayerNorm2d` to eliminate traditional nonlinear activations, resulting in reduced computational overhead and low-latency inference.

- **Adversarial Training**  
  Utilizes hinge-loss-based GAN optimization to enhance high-frequency detail preservation and perceptual realism.

- **Hybrid Loss Formulation**  
  Combines complementary objectives to balance reconstruction accuracy and perceptual quality:
  - Charbonnier loss for pixel-level fidelity  
  - Laplacian pyramid loss for edge and frequency preservation  
  - Perceptual loss based on VGG/LPIPS feature representations  

- **Dual Data Pipelines**
  - `dataloader.py`: Standard supervised loading with paired noisy/clean images  
  - `dataloader_selfsup.py`: Self-supervised training with on-the-fly synthetic degradations (Gaussian noise and blur)  

- **Inference Optimization**  
  Automatic padding and unpadding mechanisms allow processing of arbitrary image resolutions while satisfying architectural constraints.


## Requirements

- **Python**: 3.8 or higher  

### Required Packages

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

## Hardware: 
The framework was developed and evaluated on the following system:
  - CPU: AMD Ryzen 7 5800X
  - Memory: 32 GB RAM
  - GPU: NVIDIA RTX 5060 Ti (16 GB VRAM)
CUDA-enabled GPUs are strongly recommended for efficient training.

## Directory Structure**: 
```
.
├── dataloader.py              # Supervised dataset loader (Noisy / GT pairs)
├── dataloader_selfsup.py      # Self-supervised dataset loader (synthetic degradations)
├── losses.py                  # MasterLoss and individual loss components
├── models.py                  # NAFNet generator and deep discriminator architectures
├── Main_Notebook.ipynb        # Jupyter-based training and visualization
├── Inference.ipynb            # Jupyter-based inference with automatic padding
└── Optimized code/
   ├── config.py               # Configuration management
   ├── dataloader.py
   ├── dataloader_selfsup.py
   ├── default_config.json     # Default experiment configuration
   ├── inference.py            # Terminal-based inference script
   ├── losses.py
   ├── models.py
   ├── train.py                # Terminal-based training script
   └── utils.py                # Padding, checkpointing, and utility functions
```

## Installation
### 1. Create Conda Environment

```
conda create -n nafnet_gan python=3.10 -y
conda activate nafnet_gan
```

### 2. Install PyTorch (CUDA example)
```
# CUDA 11.8
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# CPU only (optional)
# conda install pytorch torchvision cpuonly -c pytorch -y
```

### 3. Install Dependencies
```
pip install scikit-image pandas tqdm matplotlib
pip install torchmetrics lpips
```
### 4. Verify Installation
```
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Usage

### Training

Training is primarily conducted using **`Main_Notebook.ipynb`**, which provides an interactive and transparent environment for experimentation, debugging, and qualitative visualization of restoration results. This workflow is particularly suitable during model development and exploratory research phases.

#### Dataset Selection

Depending on data availability, two training paradigms are supported:

- **Supervised training**  
  Use `DenoisingDataset2D` from `dataloader.py` when paired noisy and clean (ground-truth) images are available. This setting enables direct optimization against known reference images.

- **Self-supervised training**  
  Use `DenoisingDataset2D` from `dataloader_selfsup.py` when only clean images are available. In this mode, synthetic degradations (e.g., Gaussian noise and blur) are generated on-the-fly to simulate realistic corruption processes.

#### Configuration

Within the notebook setup cells, specify:
- Dataset paths (`INPUT_DIR`, `TARGET_DIR`)
- Training hyperparameters (batch size, crop size, learning rates, loss weights)
- Optional GAN and mixed-precision settings

These parameters directly control the initialization and behavior of the generator, discriminator, and loss functions.

#### Execution

Execute the notebook cells sequentially to:
1. Initialize the **NAFNet generator**
2. Initialize the **deep discriminator** (if adversarial training is enabled)
3. Construct the **MasterLoss** objective
4. Run the training loop

During training, model checkpoints and qualitative visual comparisons are automatically saved, enabling continuous monitoring of convergence and perceptual quality.

---

### Training via Terminal (Reproducible / Production Mode)

For reproducible experiments and large-scale runs, training can also be performed directly from the terminal using the optimized codebase.

```bash
python train.py --config default_config.json
```

This mode is recommended for:
  - Batch experiments
  - Hyperparameter sweeps
  - Headless or server-based training environments
The configuration file fully defines the experiment, ensuring reproducibility across runs.

## Inference

Inference on previously unseen data can be performed either **interactively via Jupyter Notebook** or **programmatically from the terminal**, depending on the intended use case (qualitative analysis vs. batch processing or deployment).

---

### Inference via Jupyter Notebook

Inference can be executed using **`Inference.ipynb`**, which is particularly well suited for qualitative inspection, visualization, and rapid validation of model performance.

#### Procedure

1. Specify the path to a trained `.pth` model checkpoint  
2. Define `INPUT_DIR`, the directory containing the noisy or degraded images  
3. Define `OUTPUT_DIR`, the directory where restored images will be saved  
4. Execute the notebook cells sequentially  

This workflow enables interactive exploration of restoration results and facilitates direct comparison between input and output images.

---

### Inference via Terminal

For efficient batch inference and deployment-oriented scenarios, inference can be performed using the terminal-based script:

```bash
python inference.py --input_dir test_images --output_dir results
```

An optional trained model checkpoint can be explicitly specified:

```bash
python inference.py \
  --input_dir test_images \
  --output_dir results \
  --model checkpoints/best_model.pth
```

This mode is recommended for large datasets, automated pipelines, and headless execution environments.

### Padding Note

During inference, all input images are automatically padded so that their spatial dimensions are multiples of **32**, as required by the NAFNet architecture. After restoration, the outputs are cropped back to their original spatial dimensions, ensuring exact spatial correspondence with the input images and preventing boundary artifacts.

---

## Architecture Details

### Generator (NAFNet)

The generator, implemented in `models.py`, replaces conventional nonlinear activation functions with a lightweight gating mechanism defined as:

\[
\text{SimpleGate}(X, Y) = X \odot Y
\]

This design, introduced by Chu *et al.*, significantly reduces computational complexity while preserving large receptive fields and expressive capacity. As a result, the generator achieves high-quality image restoration with low inference latency, making it well suited for real-time and high-throughput applications.

---

## Loss Function

The **MasterLoss**, defined in `losses.py`, integrates multiple complementary objectives to balance numerical accuracy and perceptual realism:

- **Pixel Loss**  
  Charbonnier loss, a robust L1 formulation, enforces pixel-wise fidelity and stabilizes optimization.

- **Frequency Loss**  
  Laplacian pyramid loss emphasizes edge consistency and the preservation of high-frequency structural details.

- **Perceptual Loss**  
  VGG16-based feature loss computed via `torchmetrics`, encouraging similarity in deep feature space and improving perceptual quality.

- **Adversarial Loss**  
  Hinge loss computed using the deep discriminator, promoting realistic texture reconstruction and sharper outputs.

---

## Acknowledgments

- **NAFNet / NAFSSR**  
  The generator architecture is inspired by *NAFSSR: Stereo Image Super-Resolution Using NAFNet*.

- **LPIPS / SSIM**  
  Perceptual metrics and feature representations are computed using `torchmetrics`.

---

## Original NAFNet Citation

```bibtex
@InProceedings{chu2022nafssr,
    author    = {Chu, Xiaojie and Chen, Liangyu and Yu, Wenqing},
    title     = {NAFSSR: Stereo Image Super-Resolution Using NAFNet},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {1239--1248}
}
