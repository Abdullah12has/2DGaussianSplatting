#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:,1:-1, :-2] + disp[:,1:-1,2:] - 2 * disp[:,1:-1,1:-1])
    grad_disp_y = torch.abs(disp[:,:-2, 1:-1] + disp[:,2:,1:-1] - 2 * disp[:,1:-1,1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def gradient_loss(prediction, target, mask=None, scales=1):
    """
    Multi-scale gradient loss (MonoSDF style).
    Penalizes differences in spatial gradients between prediction and target.
    
    Args:
        prediction: [H, W] depth prediction
        target: [H, W] depth target
        mask: Optional [H, W] valid region mask
        scales: Number of scales (default 1, MonoSDF uses 4 for full)
        
    Returns:
        Gradient loss value
    """
    total = 0.0
    
    for scale in range(scales):
        step = 2 ** scale
        
        # Downsample
        pred_scaled = prediction[::step, ::step]
        target_scaled = target[::step, ::step]
        mask_scaled = mask[::step, ::step] if mask is not None else None
        
        # Compute difference
        diff = pred_scaled - target_scaled
        if mask_scaled is not None:
            diff = diff * mask_scaled
        
        # X gradients (horizontal)
        grad_x = torch.abs(diff[:, 1:] - diff[:, :-1])
        if mask_scaled is not None:
            mask_x = mask_scaled[:, 1:] * mask_scaled[:, :-1]
            grad_x = grad_x * mask_x
            grad_x_sum = grad_x.sum()
        else:
            grad_x_sum = grad_x.sum()
        
        # Y gradients (vertical)
        grad_y = torch.abs(diff[1:, :] - diff[:-1, :])
        if mask_scaled is not None:
            mask_y = mask_scaled[1:, :] * mask_scaled[:-1, :]
            grad_y = grad_y * mask_y
            grad_y_sum = grad_y.sum()
        else:
            grad_y_sum = grad_y.sum()
        
        total += grad_x_sum + grad_y_sum
    
    # Normalize by number of valid pixels
    if mask is not None:
        num_valid = mask.sum().clamp(min=1)
        total = total / num_valid
    else:
        total = total / (prediction.numel())
    
    return total


def scale_invariant_depth_loss(pred_depth, mono_depth, mask=None, alpha=0.5):
    """
    Scale-invariant depth loss with gradient regularization (MonoSDF style).
    
    Combines MSE loss (after scale-shift alignment) with multi-scale gradient regularization.
    
    Args:
        pred_depth: Rendered depth [1, H, W] or [H, W]
        mono_depth: Monocular prior [1, H, W] or [H, W]
        mask: Optional valid region mask [H, W]
        alpha: Weight for gradient regularization (default 0.5, MonoSDF default)
        
    Returns:
        Loss value (scalar)
    """
    # Flatten tensors
    if pred_depth.dim() == 3:
        pred_depth = pred_depth.squeeze(0)
    if mono_depth.dim() == 3:
        mono_depth = mono_depth.squeeze(0)
    
    H, W = pred_depth.shape
    pred_flat = pred_depth.flatten()
    mono_flat = mono_depth.flatten()
    
    # Apply mask if provided
    if mask is not None:
        if mask.dim() == 3:
            mask = mask.squeeze(0)
        mask_flat = mask.flatten()
        valid = mask_flat & (pred_flat > 0) & ~torch.isnan(pred_flat) & ~torch.isnan(mono_flat)
    else:
        valid = (pred_flat > 0) & ~torch.isnan(pred_flat) & ~torch.isnan(mono_flat)
    
    if valid.sum() < 10:
        return torch.tensor(0.0, device=pred_depth.device)
    
    pred_valid = pred_flat[valid]
    mono_valid = mono_flat[valid]
    
    # Solve least squares for scale and shift: [w, q] = argmin ||w*pred + q - mono||^2
    n = pred_valid.shape[0]
    sum_pred = pred_valid.sum()
    sum_pred_sq = (pred_valid ** 2).sum()
    sum_mono = mono_valid.sum()
    sum_pred_mono = (pred_valid * mono_valid).sum()
    
    # Solve 2x2 system
    det = sum_pred_sq * n - sum_pred * sum_pred
    if det.abs() < 1e-8:
        # Degenerate case, use simple L1 loss
        return torch.abs(pred_valid - mono_valid).mean()
    
    w = (n * sum_pred_mono - sum_pred * sum_mono) / det
    q = (sum_pred_sq * sum_mono - sum_pred * sum_pred_mono) / det
    
    # Compute aligned prediction
    aligned_pred = w * pred_valid + q
    
    # MSE loss (data term)
    mse_loss = ((aligned_pred - mono_valid) ** 2).mean()
    
    # Gradient regularization term
    if alpha > 0:
        # Apply scale-shift to full 2D depth map for gradient computation
        pred_aligned_2d = w * pred_depth + q
        
        # Compute gradient loss
        grad_loss = gradient_loss(pred_aligned_2d, mono_depth, mask, scales=1)
        
        total_loss = mse_loss + alpha * grad_loss
    else:
        total_loss = mse_loss
    
    return total_loss


def mono_normal_loss(pred_normal, mono_normal, mask=None):
    """
    Normal consistency loss (MonoSDF style).
    Returns L1 and cosine losses separately for independent weighting.
    
    Args:
        pred_normal: Rendered normal [3, H, W]
        mono_normal: Monocular prior [3, H, W]
        mask: Optional valid region mask [H, W]
        
    Returns:
        l1_loss: L1 distance between normalized normals
        cos_loss: 1 - cosine similarity (angular distance)
    """
    # Explicit normalization (critical for numerical stability!)
    pred_normal = torch.nn.functional.normalize(pred_normal, p=2, dim=0)
    mono_normal = torch.nn.functional.normalize(mono_normal, p=2, dim=0)
    
    if mask is not None:
        # Expand mask to [3, H, W]
        mask_3d = mask.unsqueeze(0).expand_as(pred_normal)
        
        # Apply mask
        pred_masked = pred_normal * mask_3d
        mono_masked = mono_normal * mask_3d
        
        num_valid = mask.sum().clamp(min=1)
        
        # L1 term: |pred - target| summed over channels, averaged over pixels
        l1_per_pixel = torch.abs(pred_masked - mono_masked).sum(dim=0)  # [H, W]
        l1_loss = (l1_per_pixel * mask).sum() / num_valid
        
        # Cosine term: 1 - dot product (0 when aligned, 2 when opposite)
        cos_sim = (pred_masked * mono_masked).sum(dim=0)  # [H, W]
        cos_loss = ((1.0 - cos_sim) * mask).sum() / num_valid
    else:
        # L1 term
        l1_per_pixel = torch.abs(pred_normal - mono_normal).sum(dim=0)
        l1_loss = l1_per_pixel.mean()
        
        # Cosine term
        cos_sim = (pred_normal * mono_normal).sum(dim=0)
        cos_loss = (1.0 - cos_sim).mean()
    
    return l1_loss, cos_loss

