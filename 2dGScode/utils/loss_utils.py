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


def scale_invariant_depth_loss(pred_depth, mono_depth, mask=None):
    """
    Scale-invariant depth loss (MonoSDF style).
    Estimates optimal scale & shift via least-squares.
    
    Since monocular depth is relative (up to scale and shift),
    we solve: min_w,q ||w * pred + q - mono||^2
    
    Args:
        pred_depth: Rendered depth [1, H, W] or [H, W]
        mono_depth: Monocular prior [1, H, W] or [H, W]
        mask: Optional valid region mask [H, W]
        
    Returns:
        Loss value (scalar)
    """
    # Flatten tensors
    if pred_depth.dim() == 3:
        pred_depth = pred_depth.squeeze(0)
    if mono_depth.dim() == 3:
        mono_depth = mono_depth.squeeze(0)
    
    pred_flat = pred_depth.flatten()
    mono_flat = mono_depth.flatten()
    
    # Apply mask if provided
    if mask is not None:
        mask_flat = mask.flatten()
        valid = mask_flat & (pred_flat > 0) & ~torch.isnan(pred_flat) & ~torch.isnan(mono_flat)
    else:
        valid = (pred_flat > 0) & ~torch.isnan(pred_flat) & ~torch.isnan(mono_flat)
    
    if valid.sum() < 10:
        return torch.tensor(0.0, device=pred_depth.device)
    
    pred_valid = pred_flat[valid]
    mono_valid = mono_flat[valid]
    
    # Solve least squares: [w, q] = argmin ||A @ [w, q]^T - b||^2
    # where A = [pred, 1] and b = mono
    # Normal equations: A^T A @ x = A^T b
    
    # A^T A = [[sum(pred^2), sum(pred)], [sum(pred), n]]
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
    
    # L2 loss
    loss = ((aligned_pred - mono_valid) ** 2).mean()
    
    return loss


def mono_normal_loss(pred_normal, mono_normal, mask=None):
    """
    Normal consistency loss (MonoSDF style).
    Combines L1 and angular (cosine) loss.
    
    Args:
        pred_normal: Rendered normal [3, H, W]
        mono_normal: Monocular prior [3, H, W]
        mask: Optional valid region mask [H, W]
        
    Returns:
        Loss value (scalar)
    """
    if mask is not None:
        # Expand mask to match normal dimensions
        mask = mask.unsqueeze(0).expand_as(pred_normal)
        pred_masked = pred_normal * mask
        mono_masked = mono_normal * mask
        
        # Count valid pixels for normalization
        num_valid = mask[0].sum().clamp(min=1)
        
        # L1 term
        l1_term = torch.abs(pred_masked - mono_masked).sum() / (3 * num_valid)
        
        # Angular term (1 - cos similarity)
        cos_sim = (pred_masked * mono_masked).sum(dim=0)  # [H, W]
        angular_term = ((1 - cos_sim) * mask[0]).sum() / num_valid
    else:
        # L1 term  
        l1_term = torch.abs(pred_normal - mono_normal).mean()
        
        # Angular term: 1 - dot product (should be 0 when aligned)
        cos_sim = (pred_normal * mono_normal).sum(dim=0)  # [H, W]
        angular_term = (1 - cos_sim).mean()
    
    return l1_term + angular_term

