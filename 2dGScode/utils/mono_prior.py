#
# Monocular Depth and Normal Prior Estimation
# For 2D Gaussian Splatting enhancement
#
import sys

import logging

# 1. Force sys.stdout to behave like a terminal-friendly object
if not hasattr(sys.stdout, 'isatty'):
    sys.stdout.isatty = lambda: False
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
import os

# Global model instances (lazy-loaded)
_depth_model = None
_normal_model = None
_depth_processor = None


def get_depth_model(device="cuda"):
    """
    Load Depth-Anything-V2 model for monocular depth estimation.
    Uses HuggingFace transformers.
    """
    global _depth_model, _depth_processor
    
    if _depth_model is None:
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            
            model_name = "depth-anything/Depth-Anything-V2-Small-hf"
            print(f"Loading depth model: {model_name}")
            
            _depth_processor = AutoImageProcessor.from_pretrained(model_name)
            _depth_model = AutoModelForDepthEstimation.from_pretrained(model_name)
            _depth_model = _depth_model.to(device)
            _depth_model.eval()
            
        except ImportError:
            print("Warning: transformers not installed. Install with: pip install transformers")
    
    return _depth_model, _depth_processor




def get_normal_model(device="cuda"):
    """
    Load Omnidata DPT-Hybrid normal estimation model.

    Tries two approaches:
    1. Official torch.hub entrypoint (auto-downloads weights)
    2. MiDaS DPT-Hybrid architecture + local Omnidata weights file

    Local weights path: ./pretrained_models/omnidata_dpt_normal_v2.ckpt
    Download:
        pip install huggingface_hub
        huggingface-cli download clay3d/omnidata omnidata_dpt_normal_v2.ckpt --local-dir pretrained_models/
    """
    global _normal_model

    if _normal_model is None:
        # Approach 1: Official Omnidata torch.hub (auto-downloads weights)
        try:
            model = torch.hub.load('alexsax/omnidata_models', 'surface_normal_dpt_hybrid_384')
            model.to(device)
            model.eval()
            _normal_model = model
            print("[Normal Model] Loaded Omnidata via torch.hub (auto-downloaded)")
            return _normal_model
        except Exception as e:
            print(f"[Normal Model] torch.hub auto-download failed: {e}")

        # Approach 2: MiDaS architecture + local weights file
        weights_path = Path(__file__).parent.parent / "pretrained_models" / "omnidata_dpt_normal_v2.ckpt"

        if not weights_path.exists():
            print(f"[Normal Model] Weights not found at {weights_path}")
            print("  Download with:")
            print("    pip install huggingface_hub")
            print(f"    huggingface-cli download clay3d/omnidata omnidata_dpt_normal_v2.ckpt --local-dir {weights_path.parent}")
            print("  Falling back to depth-derived normals.")
            return None

        try:
            import torch.nn as nn

            # MiDaS DPT-Hybrid shares the same architecture as Omnidata
            model = torch.hub.load("isl-org/MiDaS", "DPT_Hybrid", pretrained=False)

            # Replace 1-channel depth head with 3-channel normal head
            head = list(model.scratch.output_conv.children())
            model.scratch.output_conv = nn.Sequential(
                head[0],  # Conv2d(256, 128, 3, 1, 1)
                head[1],  # Interpolate(scale_factor=2)
                head[2],  # Conv2d(128, 32, 3, 1, 1)
                head[3],  # ReLU
                nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            )

            checkpoint = torch.load(str(weights_path), map_location="cpu")
            model.load_state_dict(checkpoint)
            model.to(device)
            model.eval()
            _normal_model = model
            print("[Normal Model] Loaded Omnidata DPT-Hybrid from local weights")

        except Exception as e:
            print(f"[Normal Model] Failed to load: {e}")
            print("  Falling back to depth-derived normals.")
            return None

    return _normal_model


def estimate_depth(image, device="cuda"):
    """
    Estimate monocular depth from an image.
    
    Args:
        image: PIL Image or numpy array [H, W, 3] (0-255)
        device: torch device
        
    Returns:
        depth: numpy array [H, W] with relative depth values
    """
    model, processor = get_depth_model(device)
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    
    # Process image
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    # Interpolate to original size
    prediction = F.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],  # PIL .size is (W, H), we need (H, W)
        mode="bicubic",
        align_corners=False,
    )
    
    depth = prediction.squeeze()
    return depth


def estimate_normal_from_depth(depth, mask=None, fovx=None, fovy=None):
    """
    Estimate surface normals from depth map using gradients.
    Uses perspective-correct formula when FoV is provided.

    Args:
        depth: torch tensor [H, W]
        mask: optional valid region mask
        fovx: horizontal field of view in radians (for perspective correction)
        fovy: vertical field of view in radians (for perspective correction)

    Returns:
        normal: torch tensor [3, H, W] with xyz normal components
    """
    H, W = depth.shape
    device = depth.device

    # Compute depth gradients
    dz_dx = torch.gradient(depth, dim=1)[0]
    dz_dy = torch.gradient(depth, dim=0)[0]

    if fovx is not None and fovy is not None:
        # Perspective-correct normal estimation
        fx = W / (2 * np.tan(fovx / 2))
        fy = H / (2 * np.tan(fovy / 2))

        # Pixel coordinate grids
        y, x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        cx, cy = W / 2.0, H / 2.0

        # For perspective cameras, the surface normal in camera space is:
        # n = (-fx * dz/dx, -fy * dz/dy, z + (x-cx)*dz/dx + (y-cy)*dz/dy)
        nx = -fx * dz_dx
        ny = -fy * dz_dy
        nz = depth + (x - cx) * dz_dx + (y - cy) * dz_dy
        normal = torch.stack([nx, ny, nz], dim=-1)
    else:
        # Orthographic fallback: n = (-dz/dx, -dz/dy, 1)
        normal = torch.stack([-dz_dx, -dz_dy, torch.ones_like(depth)], dim=-1)

    # Normalize
    norm = torch.norm(normal, dim=-1, keepdim=True)
    normal = normal / (norm + 1e-8)

    return normal.movedim(-1, 0)


def estimate_normal(image, device="cuda", fovx=None, fovy=None):
    """
    Estimate surface normals from an image using Omnidata or depth fallback.

    Args:
        image: PIL Image or numpy array [H, W, 3] (0-255)
        device: torch device
        fovx: horizontal FoV in radians (used only for depth-derived fallback)
        fovy: vertical FoV in radians (used only for depth-derived fallback)

    Returns:
        normal: torch tensor [3, H, W] with xyz normal components in [-1, 1]
    """
    model = get_normal_model(device)

    if model is None:
        depth = estimate_depth(image, device)
        return estimate_normal_from_depth(depth, fovx=fovx, fovy=fovy)

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))

    from torchvision import transforms

    orig_w, orig_h = image.size

    # Omnidata preprocessing: resize to 384x384, normalize with mean=0.5 std=0.5
    transform = transforms.Compose([
        transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        normal_pred = model(input_tensor)

    if normal_pred.dim() == 3:
        normal_pred = normal_pred.unsqueeze(0)

    # Resize back to original resolution
    normal_pred = F.interpolate(
        normal_pred,
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    )

    # Map from [0,1] to [-1,1]
    normal = normal_pred.squeeze(0).clamp(0, 1)  # [3, H, W]
    normal = normal * 2 - 1

    # L2 normalize
    norm = torch.norm(normal, dim=0, keepdim=True)
    normal = normal / (norm + 1e-8)

    return normal


class MonoPriorProcessor:
    """
    Process and cache monocular priors for a dataset.
    """
    
    def __init__(self, device="cuda"):
        self.device = device
    
    def compute_priors(self, image_paths, output_dir, force_recompute=False):
        """
        Compute and save depth/normal priors for a list of images.
        
        Args:
            image_paths: List of image file paths
            output_dir: Directory to save priors (creates mono_depth/, mono_normal/ subdirs)
            force_recompute: If True, recompute even if files exist
        """
        output_dir = Path(output_dir)
        depth_dir = output_dir / "mono_depth"
        normal_dir = output_dir / "mono_normal"
        
        depth_dir.mkdir(parents=True, exist_ok=True)
        normal_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Computing monocular priors for {len(image_paths)} images...")
        
        for img_path in image_paths:
            img_path = Path(img_path)
            img_name = img_path.stem
            
            depth_path = depth_dir / f"{img_name}.npy"
            normal_path = normal_dir / f"{img_name}.npy"
            
            # Skip if exists and not forcing recompute
            if not force_recompute and depth_path.exists() and normal_path.exists():
                continue
            
            # Load image
            image = Image.open(img_path).convert("RGB")
            
            # Compute depth
            if not depth_path.exists() or force_recompute:
                depth = estimate_depth(image, self.device)
                np.save(depth_path, depth.cpu().numpy().astype(np.float32))

            # Compute normal
            if not normal_path.exists() or force_recompute:
                normal = estimate_normal(image, self.device)
                np.save(normal_path, normal.cpu().numpy().astype(np.float32))
        
        print(f"Priors saved to: {output_dir}")
    
    @staticmethod
    def load_depth(path):
        """Load cached depth prior."""
        if Path(path).exists():
            return np.load(path)
        return None
    
    @staticmethod
    def load_normal(path):
        """Load cached normal prior."""
        if Path(path).exists():
            return np.load(path)
        return None
    
    @staticmethod
    def get_prior_paths(image_path, prior_dir):
        """
        Get paths to depth/normal priors for an image.
        
        Args:
            image_path: Path to the original image
            prior_dir: Base directory containing mono_depth/ and mono_normal/
            
        Returns:
            (depth_path, normal_path) tuple
        """
        prior_dir = Path(prior_dir)
        img_name = Path(image_path).stem
        
        depth_path = prior_dir / "mono_depth" / f"{img_name}.npy"
        normal_path = prior_dir / "mono_normal" / f"{img_name}.npy"
        
        return depth_path, normal_path
