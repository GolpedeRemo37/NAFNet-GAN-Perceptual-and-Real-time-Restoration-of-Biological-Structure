import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
from torchmetrics.image import StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
from torch.cuda.amp import autocast


# ========================================================================================
# === 🧩 AMALGAMATED LOSS MODULE (NaN-PROOF for [mean=0, std=1] input) ===
# ========================================================================================

# ---------- Global Helper (Renamed) ----------
def normalize_to_minus_one_one(x: torch.Tensor) -> torch.Tensor:
    """
    Normalizes a tensor from any range (e.g., standardized) 
    back to [-1, 1] for perceptual losses.
    """
    x = ((x-x.min())/(x.max()-x.min() + 1e-9))*2-1 # Use 1e-9 for safety
    return x


# ---------- Helpers (Stabilized) ----------
def has_bad_values(t):
    return not torch.isfinite(t).all().item() if t is not None else True


def charbonnier(pred, target, eps=1e-3):  # Increased eps for fp16 stability
    # This loss now operates on standardized (mean=0, std=1) data
    diff = (pred - target).to(torch.float32)
    return torch.mean(torch.sqrt(diff * diff + eps**2))


def laplacian_loss(pred, target):
    # This loss now operates on standardized (mean=0, std=1) data
    kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                          device=pred.device, dtype=pred.dtype).unsqueeze(0).unsqueeze(0)
    channels = pred.shape[1]
    kernel = kernel.repeat(channels, 1, 1, 1)
    p = F.conv2d(pred, kernel, padding=1, groups=channels)
    t = F.conv2d(target, kernel, padding=1, groups=channels)
    return F.l1_loss(p, t)


def fft_highfreq_loss(pred, target, eps: float = 1e-6):
    # This loss now operates on standardized (mean=0, std=1) data
    p = pred[:, 0].float()    # Force float32
    t = target[:, 0].float()

    mag_p = torch.abs(torch.fft.fftshift(torch.fft.fft2(p)))
    mag_t = torch.abs(torch.fft.fftshift(torch.fft.fft2(t)))

    # Clamp magnitude to prevent inf
    mag_p = torch.clamp(mag_p, min=eps)
    mag_t = torch.clamp(mag_t, min=eps)

    B, H, W = mag_p.shape
    yy = torch.arange(H, device=p.device) - H / 2
    xx = torch.arange(W, device=p.device) - W / 2
    Y, X = torch.meshgrid(yy, xx, indexing='ij')
    dist = torch.sqrt(X**2 + Y**2)
    mask = dist / (dist.max() + 1e-9)
    mask = mask.unsqueeze(0)

    return F.l1_loss(mask * mag_p, mask * mag_t)


def discriminator_hinge_loss(d_real, d_fake):
    loss_real = torch.mean(F.relu(1.0 - d_real))
    loss_fake = torch.mean(F.relu(1.0 + d_fake))
    return 0.5 * (loss_real + loss_fake)


def generator_hinge_loss(d_fake):
    return F.softplus(-d_fake).mean()

def r1_penalty(d_out, real_images):
    # real_images are standardized, which is fine.
    grads = torch.autograd.grad(outputs=d_out.sum(), inputs=real_images,
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
    return (grads.view(grads.size(0), -1).pow(2).sum(1)).mean()


# ---------- VGG / LPIPS Feature Extractors (NaN-PROOF) ----------
class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval().to(device)
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    def _prep(self, x):
        # MODIFICATION: Explicitly re-normalize from (mean=0, std=1) to [-1, 1]
        x = normalize_to_minus_one_one(x) 
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # Now convert from [-1, 1] to [0, 1] for ImageNet stats
        x = (x + 1.0) / 2.0 
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        return F.l1_loss(self.vgg(self._prep(pred)), self.vgg(self._prep(target)))


class LPIPSLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        # LPIPS (AlexNet) expects input in [-1, 1] range
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=False).eval().to(device)
        for param in self.lpips.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        # MODIFICATION: Explicitly re-normalize from (mean=0, std=1) to [-1, 1]
        pred = normalize_to_minus_one_one(pred)
        target = normalize_to_minus_one_one(target)

        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        # Compute in float32, let autocast handle downcast if needed
        return self.lpips(pred, target).mean()


# ---------- Individual Loss Modules (Stabilized) ----------
class EnhancedDeblurLoss(nn.Module):
    def __init__(self, device, weights_obj):
        super().__init__()
        self.weights = weights_obj
        self.vgg = VGGPerceptualLoss(device)
        # SSIM is initialized with data_range=2.0, so it *must* get [-1, 1] data
        self.ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)

    def forward_discriminator(self, d_real, d_fake):
        return discriminator_hinge_loss(d_real, d_fake)

    def calculate_r1_penalty(self, d_real_logits, real_imgs):
        if self.weights.r1_gamma == 0:
            return torch.tensor(0.0, device=d_real_logits.device)
        penalty = r1_penalty(d_real_logits, real_imgs)
        return 0.5 * self.weights.r1_gamma * penalty

    def forward_generator(self, inputs):
        pred, target = inputs['pred_img'], inputs['target_img']
        d_fake = inputs['d_fake_logits']

        # These losses operate on standardized data
        loss_c = charbonnier(pred, target) * self.weights.lambda_charb
        loss_l = laplacian_loss(pred, target) * self.weights.lambda_lap
        loss_f = fft_highfreq_loss(pred, target) * self.weights.lambda_fft

        # These losses require re-normalization to [-1, 1]
        loss_v = self.vgg(pred, target) * self.weights.lambda_vgg
        
        # MODIFICATION: Explicitly re-normalize for SSIM
        pred_norm_ssim = normalize_to_minus_one_one(pred)
        target_norm_ssim = normalize_to_minus_one_one(target)
        loss_s = (1.0 - self.ssim(pred_norm_ssim, target_norm_ssim)) * self.weights.lambda_ssim
        
        # This loss operates on discriminator logits
        loss_g = generator_hinge_loss(d_fake) * self.weights.lambda_gan

        return loss_c + loss_v + loss_s + loss_l + loss_f + loss_g


# ========================================================================
# === 💡 NEW MODIFIED CombinedCriterion (No Noise) =======================
# ========================================================================
# Replace the old CombinedCriterion class in your loss script with this one.
# This version removes the dependency on 'pred_noise' and 'target_noise'.
# ========================================================================
class CombinedCriterion(nn.Module):
    def __init__(self, device, weights_obj):
        super().__init__()
        self.weights = weights_obj
        self.device_type = device.type

        self.lpips = LPIPSLoss(device)
        self.vgg = VGGPerceptualLoss(device)
        # SSIM is initialized with data_range=2.0, so it *must* get [-1, 1] data
        self.ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
        print("CombinedCriterion (All-in-One) initialized with LPIPS, VGG, and SSIM modules.")
        print("✅ NOTE: This is the MODIFIED version without 'loss_mse_noise' for NAFNet.")

    def forward_generator(self, inputs):
        # === These lines are the only ones that need to be provided ===
        pred_img, target_img = inputs['pred_img'], inputs['target_img']
        d_fake = inputs['d_fake_logits']

        # --- 'pred_noise' and 'target_noise' inputs are no longer required ---
        
        # --- Core DDPM Noise Loss (REMOVED) ---
        # loss_mse_noise = F.mse_loss(pred_noise, target_noise) # <-- REMOVED

        # Perceptual / Realism (these classes handle re-normalization internally)
        loss_lpips = self.lpips(pred_img, target_img) * self.weights.lambda_lpips
        loss_vgg = self.vgg(pred_img, target_img) * self.weights.lambda_vgg

        # Pixel / Structural (Charbonnier operates on standardized data)
        loss_charb_img = charbonnier(pred_img, target_img) * self.weights.lambda_charb
        
        # Explicitly re-normalize for SSIM
        pred_norm_ssim = normalize_to_minus_one_one(pred_img)
        target_norm_ssim = normalize_to_minus_one_one(target_img)
        loss_ssim = (1.0 - self.ssim(pred_norm_ssim, target_norm_ssim)) * self.weights.lambda_ssim

        # High-Frequency / Edge (these operate on standardized data)
        total_lap_weight = self.weights.lambda_lap + self.weights.lambda_edge
        loss_lap = laplacian_loss(pred_img, target_img) * total_lap_weight
        loss_fft_highfreq = fft_highfreq_loss(pred_img, target_img) * self.weights.lambda_fft

        # RFFT L1 Loss
        fft_pred = torch.fft.rfft2(normalize_to_minus_one_one(pred_img))
        fft_target = torch.fft.rfft2(normalize_to_minus_one_one(target_img))
        loss_fft_rfft = F.l1_loss(torch.abs(fft_pred), torch.abs(fft_target)) * self.weights.lambda_fft_cc

        # GAN (operates on logits)
        loss_gan = generator_hinge_loss(d_fake) * self.weights.lambda_gan

        # Total (without noise loss)
        total_loss = (
            # loss_mse_noise +  # <-- REMOVED
            loss_lpips +
            loss_vgg +
            loss_ssim +
            loss_charb_img +
            loss_lap +
            loss_fft_highfreq +
            loss_fft_rfft +
            loss_gan
        )

        return total_loss

    def forward_discriminator(self, d_real, d_fake):
        return discriminator_hinge_loss(d_real, d_fake)

    def calculate_r1_penalty(self, d_real_logits, real_imgs):
        if self.weights.r1_gamma == 0:
            return torch.tensor(0.0, device=d_real_logits.device)
        penalty = r1_penalty(d_real_logits, real_imgs)
        return 0.5 * self.weights.r1_gamma * penalty

# ========================================================================
# === MASTER LOSS (Unchanged) ===
# ========================================================================
class MasterLoss(nn.Module):
    def __init__(self, loss_type, weights, device):
        super().__init__()
        self.loss_type = loss_type
        if self.loss_type == 'enhanced_deblur':
            self.criterion = EnhancedDeblurLoss(device, weights)
        elif self.loss_type == 'combined_criterion':
            self.criterion = CombinedCriterion(device, weights)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        print(f"Initialized MasterLoss with '{self.loss_type}' criterion.")

    def forward_generator(self, inputs):
        loss = self.criterion.forward_generator(inputs)
        if has_bad_values(loss):
            raise RuntimeError(f"NaN/Inf in generator loss ({self.loss_type})")
        return loss

    def forward_discriminator(self, d_real, d_fake):
        loss = self.criterion.forward_discriminator(d_real, d_fake)
        if has_bad_values(loss):
            raise RuntimeError(f"NaN/Inf in discriminator hinge loss ({self.loss_type})")
        return loss

    def calculate_r1_penalty(self, d_real_logits, real_imgs):
        loss = self.criterion.calculate_r1_penalty(d_real_logits, real_imgs)
        if has_bad_values(loss):
            raise RuntimeError(f"NaN/Inf in R1 penalty ({self.loss_type})")
        return loss